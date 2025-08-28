# app.py — Sales Progression Tracker (UK, Multi-case)
# - Home page: create & list cases
# - Per-case dashboards: upload docs / paste text / update same case
# - Gumroad license gate + admin override
# - Corporate UI, progress bar, timeline, actions/waiting summary
# - Robust JSON extraction + stage normalization (no schema crashes)
# - SQLite persistence across sessions (per user key)

import os, re, io, json, time, hashlib, sqlite3, datetime as dt
import urllib.parse, urllib.request
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# Optional readers (loaded only if needed)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# ================== Config & Secrets ==================
st.set_page_config(page_title="Sales Progression Tracker", page_icon="📈", layout="wide")

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
DEFAULT_MODEL  = get_secret("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

# Licensing
GUMROAD_PRODUCT_PERMALINK = get_secret("GUMROAD_PRODUCT_PERMALINK", "")
ADMIN_BYPASS              = get_secret("ADMIN_BYPASS", "")

# Usage controls
USAGE_DAILY_LIMIT      = int(get_secret("USAGE_DAILY_LIMIT", "150"))
USAGE_COOLDOWN_SECONDS = int(get_secret("USAGE_COOLDOWN_SECONDS", "5"))

# SLA highlight
DEFAULT_SLA_PENDING_DAYS = int(get_secret("SLA_PENDING_DAYS", "10"))

if not OPENAI_API_KEY:
    st.error("Server misconfigured: missing OPENAI_API_KEY in Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ================== Styles (high-contrast, light & dark) ==================
st.markdown("""
<style>
@media (prefers-color-scheme: light) {
  :root {
    --fg:#111827; --muted:#6b7280; --card:#ffffff; --border:#e5e7eb;
    --shadow: 0 6px 24px rgba(0,0,0,.06); --accent:#2563eb;
  }
}
@media (prefers-color-scheme: dark) {
  :root {
    --fg:#e5e7eb; --muted:#94a3b8; --card:#0f172a; --border:#1f2937;
    --shadow: 0 8px 28px rgba(0,0,0,.45); --accent:#3b82f6;
  }
}
.block-container { padding-top: 1rem; }
.card, .kpi { border:1px solid var(--border); background:var(--card); border-radius:16px; padding:16px; box-shadow:var(--shadow); }
.card, .card * { color:var(--fg) !important; text-shadow:none !important; opacity:1 !important; }
.kpi, .kpi * { color:var(--fg) !important; text-shadow:none !important; opacity:1 !important; }
.small { color:var(--muted) !important; font-size:.9rem }
div.stButton>button, div.stDownloadButton>button { border-radius:10px; padding:.6rem 1rem; }
hr { border:none; border-top:1px solid var(--border); margin: 18px 0; }
.progress-label { display:flex; justify-content:space-between; font-size:.9rem; color:var(--muted); margin-top:6px;}
.sidebar-case { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.sidebar-pill { background:transparent; border:1px solid var(--border); border-radius:12px; padding:.35rem .6rem; font-size:.85rem; }
a.case-link { text-decoration:none; color:var(--fg); }
</style>
""", unsafe_allow_html=True)

# ================== Session State ==================
if "licensed" not in st.session_state:
    st.session_state.licensed = False
if "usage" not in st.session_state:
    st.session_state.usage = {"date": dt.date.today().isoformat(), "count": 0, "last_ts": 0.0, "bypass": False}
if "uid" not in st.session_state:
    st.session_state.uid = None  # set after unlock

# ================== Gumroad gate ==================
def verify_gumroad_license(license_key: str, product_permalink: str) -> bool:
    try:
        data = urllib.parse.urlencode({
            "product_permalink": product_permalink,
            "license_key": license_key,
            "increment_uses_count": "false",
        }).encode("utf-8")
        req = urllib.request.Request("https://api.gumroad.com/v2/licenses/verify",
                                     data=data, method="POST",
                                     headers={"Content-Type":"application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            j = json.loads(resp.read().decode("utf-8"))
        return bool(j.get("success"))
    except Exception:
        return False

def show_license_gate():
    st.title("📈 Sales Progression Tracker (UK)")
    st.caption("Unlock with your Gumroad key. Admins can use a private override code.")
    with st.form("license"):
        access = st.text_input("Access Key", type="password")
        ok = st.form_submit_button("Unlock", use_container_width=True)
    if ok:
        if ADMIN_BYPASS and access.strip() == ADMIN_BYPASS.strip():
            st.session_state.usage["bypass"] = True
            st.session_state.licensed = True
            st.session_state.uid = "admin:" + hashlib.sha256(access.encode()).hexdigest()[:16]
            st.success("Admin override accepted ✅")
            st.rerun()
        if not GUMROAD_PRODUCT_PERMALINK:
            st.error("Missing GUMROAD_PRODUCT_PERMALINK in Secrets.")
            st.stop()
        if verify_gumroad_license(access.strip(), GUMROAD_PRODUCT_PERMALINK):
            st.session_state.licensed = True
            st.session_state.uid = "gum:" + hashlib.sha256(access.encode()).hexdigest()[:16]
            st.success("License verified ✅")
            st.rerun()
        else:
            st.error("Invalid key. Please try again or contact support.")

if not st.session_state.licensed:
    show_license_gate()
    if not st.session_state.licensed: st.stop()

UID = st.session_state.uid or "anon"

# ================== Persistence (SQLite) ==================
DB_PATH = "progress.db"

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid TEXT NOT NULL,
        name TEXT NOT NULL,
        created_ts INTEGER NOT NULL,
        last_progress REAL DEFAULT 0.0,
        last_summary TEXT DEFAULT ''
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS case_inputs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER NOT NULL,
        ts INTEGER NOT NULL,
        source TEXT NOT NULL,   -- 'emails'|'docs'|'description'
        text TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS case_snapshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER NOT NULL,
        ts INTEGER NOT NULL,
        milestones_json TEXT NOT NULL,
        progress REAL NOT NULL,
        summary TEXT
    )""")
    conn.commit()
    return conn

def db_create_case(uid: str, name: str) -> int:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO cases(uid,name,created_ts) VALUES(?,?,?)",
                (uid, name, int(time.time())))
    conn.commit()
    return cur.lastrowid

def db_rename_case(uid: str, case_id: int, new_name: str):
    conn = db_conn()
    conn.execute("UPDATE cases SET name=? WHERE uid=? AND id=?", (new_name, uid, case_id))
    conn.commit()

def db_delete_case(uid: str, case_id: int):
    conn = db_conn()
    conn.execute("DELETE FROM case_inputs WHERE case_id=?", (case_id,))
    conn.execute("DELETE FROM case_snapshots WHERE case_id=?", (case_id,))
    conn.execute("DELETE FROM cases WHERE uid=? AND id=?", (uid, case_id))
    conn.commit()

def db_list_cases(uid: str) -> List[Dict[str,Any]]:
    conn = db_conn()
    rows = conn.execute("SELECT id,name,created_ts,last_progress FROM cases WHERE uid=? ORDER BY id DESC", (uid,)).fetchall()
    return [{"id":r[0], "name":r[1], "ts":r[2], "progress":r[3]} for r in rows]

def db_add_input(case_id: int, source: str, text: str):
    conn = db_conn()
    conn.execute("INSERT INTO case_inputs(case_id,ts,source,text) VALUES (?,?,?,?)",
                 (case_id, int(time.time()), source, text))
    conn.commit()

def db_all_input_text(case_id: int) -> str:
    conn = db_conn()
    rows = conn.execute("SELECT text FROM case_inputs WHERE case_id=? ORDER BY id ASC", (case_id,)).fetchall()
    return "\n\n".join(r[0] for r in rows)

def db_save_snapshot(case_id: int, milestones: List[Dict[str,Any]], progress: float, summary: str):
    conn = db_conn()
    conn.execute("INSERT INTO case_snapshots(case_id,ts,milestones_json,progress,summary) VALUES (?,?,?,?,?)",
                 (case_id, int(time.time()), json.dumps(milestones), float(progress), summary))
    conn.execute("UPDATE cases SET last_progress=?, last_summary=? WHERE id=?",
                 (float(progress), summary, case_id))
    conn.commit()

def db_last_snapshot(case_id: int) -> Optional[Dict[str,Any]]:
    conn = db_conn()
    r = conn.execute("SELECT milestones_json,progress,summary,ts FROM case_snapshots WHERE case_id=? ORDER BY id DESC LIMIT 1", (case_id,)).fetchone()
    if not r: return None
    return {"milestones": json.loads(r[0] or "[]"), "progress": r[1], "summary": r[2], "ts": r[3]}

# ================== Canonical stages & mapping ==================
CANONICAL = [
    "Offer Received","Offer Accepted","Memorandum of Sale",
    "ID/AML Checks","Mortgage Application","Mortgage Offer",
    "Survey / Valuation","Searches Ordered","Searches Returned",
    "Draft Contracts Issued","Enquiries Raised","Enquiries Answered",
    "Exchange of Contracts","Completion","Fall-through","Other"
]
STAGE_ORDER = {name:i for i,name in enumerate(CANONICAL)}

_STAGE_MAP = [
    (r"\boffer (received|made)\b",             "Offer Received"),
    (r"\boffer accepted\b|\bagreed\b",         "Offer Accepted"),
    (r"\bmemorandum of sale|\bmos\b",          "Memorandum of Sale"),
    (r"\baml\b|\bid\b|\bkyc\b",                "ID/AML Checks"),
    (r"\bmortgage application\b|\baip\b",      "Mortgage Application"),
    (r"\bmortgage offer\b|\boffer issued\b",   "Mortgage Offer"),
    (r"\bvaluation\b|\bsurvey(or)?\b|\bsurvey\b", "Survey / Valuation"),
    (r"\bsearches (ordered|applied)\b",        "Searches Ordered"),
    (r"\bsearches (returned|back|received)\b", "Searches Returned"),
    (r"\bdraft contract(s)? issued\b|\bcontract pack\b", "Draft Contracts Issued"),
    (r"\benquiries (raised|sent)\b",           "Enquiries Raised"),
    (r"\breplies to enquiries|\benquiries answered\b", "Enquiries Answered"),
    (r"\bexchange\b|\bexchanged\b",            "Exchange of Contracts"),
    (r"\bcompletion\b|\bcomplete on\b",        "Completion"),
    (r"\bfall[- ]?through\b|\bchain collapsed\b", "Fall-through"),
]

def normalize_stage(raw: str) -> str:
    if not raw: return "Other"
    text = str(raw).lower().strip()
    for pat, dst in _STAGE_MAP:
        if re.search(pat, text): return dst
    for c in CANONICAL:
        if c.lower() in text: return c
    return "Other"

# ================== Model prompts ==================
SYSTEM_LAW = (
    "You are a UK residential conveyancing assistant. Follow UK practice from instruction to completion. "
    "Do NOT invent dates or outcomes. Keep responses factual and concise."
)

def prompt_extract(text: str) -> str:
    t = (text or "").strip()[:100_000]
    return (
        "From the following text (emails, notes, or description), identify progression milestones in a UK residential sale. "
        "For each milestone return: stage, status (done|pending|blocked), date_iso (YYYY-MM-DD or null), "
        "actor (agent|solicitor|buyer|seller|lender|other), details, blockers, next_action.\n\n"
        f"TEXT START >>>\n{t}\n<<< TEXT END\n\n"
        "Return ONLY JSON like:\n"
        "{ \"milestones\": [ {\"stage\":\"\",\"status\":\"pending\",\"date_iso\":null,\"actor\":\"other\",\"details\":\"\",\"blockers\":null,\"next_action\":null} ] }"
    )

def prompt_summary(milestones: List[Dict[str,Any]]) -> str:
    return (
        "Write a client-facing status summary of this case. "
        "Use short paragraphs and a final section with bullet points under headings 'Action to take' and 'Waiting for'. "
        "Keep it UK property tone, professional, concise.\n\n"
        f"DATA:\n{json.dumps(milestones, ensure_ascii=False)}"
    )

# ================== LLM helpers ==================
def chat_json(model: str, system: str, user: str, max_tokens: int = 1300, temperature: float = 0.2) -> Dict[str, Any]:
    for _ in range(2):
        r = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=max_tokens,
        )
        text = (r.choices[0].message.content or "").strip()
        try:
            return json.loads(text)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Model did not return valid JSON.")

# ================== Utilities ==================
def _decode(data: bytes) -> str:
    for enc in ("utf-8","utf-16","latin-1"):
        try: return data.decode(enc)
        except Exception: continue
    return data.decode("utf-8", errors="ignore")

def read_file_to_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".txt"):
        return _decode(data)
    if name.endswith(".pdf"):
        if PdfReader is None: return ""
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    if name.endswith(".docx"):
        if docx2txt is None: return ""
        try:
            return docx2txt.process(io.BytesIO(data)) or ""
        except Exception:
            return ""
    return ""

def _to_iso(date_str: Optional[str]) -> Optional[str]:
    if not date_str: return None
    s = date_str.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s): return s
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        d, mo, y = m.groups()
        y = "20"+y if len(y)==2 else y
        try: return dt.date(int(y), int(mo), int(d)).isoformat()
        except Exception: return None
    return None

def normalize_rows(rows: List[Dict[str, Any]], merge_duplicates: bool, require_date_for_done: bool) -> List[Dict[str, Any]]:
    norm, seen = [], set()
    for r in rows or []:
        stage = normalize_stage(r.get("stage",""))
        status = str(r.get("status","pending")).lower()
        status = status if status in {"done","pending","blocked"} else "pending"
        date_iso = _to_iso(r.get("date_iso"))
        actor = str(r.get("actor") or "other").lower()
        actor = actor if actor in {"agent","solicitor","buyer","seller","lender","other"} else "other"
        details = (r.get("details") or "").strip()
        blockers = (r.get("blockers") or None)
        next_action = (r.get("next_action") or None)
        if require_date_for_done and status=="done" and not date_iso:
            status = "pending"
        key = (stage, status, date_iso or "", actor, details[:90])
        if merge_duplicates and key in seen: continue
        seen.add(key)
        norm.append({"stage":stage,"status":status,"date":date_iso,"actor":actor,
                     "details":details,"blockers":blockers,"next_action":next_action})
    return norm

def compute_progress(rows: List[Dict[str,Any]]) -> float:
    done_idx = [STAGE_ORDER.get(r["stage"], -1) for r in rows if r["status"]=="done" and r["stage"] in STAGE_ORDER]
    if not done_idx: return 3.0
    last = max(done_idx)
    max_index = STAGE_ORDER["Completion"]
    pct = max(0.0, min(100.0, (last / max_index) * 100.0))
    return round(pct, 1)

def enforce_caps() -> Optional[str]:
    usage = st.session_state.usage
    now = time.time()
    today = dt.date.today().isoformat()
    if usage["date"] != today:
        usage.update({"date":today, "count":0, "last_ts":0.0})
    if not usage.get("bypass"):
        if usage["count"] >= USAGE_DAILY_LIMIT:
            reset_at = dt.datetime.combine(dt.date.today()+dt.timedelta(days=1), dt.time.min)
            mins_left = int((reset_at - dt.datetime.now()).total_seconds()//60)
            return f"Daily limit reached. Resets in ~{mins_left} minutes."
        since = now - float(usage["last_ts"])
        if since < USAGE_COOLDOWN_SECONDS:
            wait = int(USAGE_COOLDOWN_SECONDS - since + 1)
            return f"Please wait {wait}s before generating again."
    return None

# ================== Routing helpers (Home vs Case) ==================
def set_route(page: str, case_id: Optional[int] = None):
    if case_id is None:
        st.experimental_set_query_params(page=page)
    else:
        st.experimental_set_query_params(page=page, case_id=str(case_id))
    st.experimental_rerun()

def get_route() -> Tuple[str, Optional[int]]:
    q = st.experimental_get_query_params()
    page = q.get("page", ["home"])[0]
    case_id = q.get("case_id", [None])[0]
    return page, int(case_id) if case_id else None

# ================== Sidebar (global) ==================
with st.sidebar:
    st.header("🏠 Navigation")
    if st.button("Home", use_container_width=True):
        set_route("home")

    st.markdown("---")
    st.header("📁 Your cases")

    cases = db_list_cases(UID)
    if not cases:
        st.caption("No cases yet. Create one on Home.")
    else:
        for c in cases:
            when = dt.datetime.fromtimestamp(c["ts"]).strftime("%d %b %Y %H:%M")
            st.markdown(
                f"<div class='sidebar-case'>"
                f"<a class='case-link' href='?page=case&case_id={c['id']}'>{c['name']}</a>"
                f"<span class='sidebar-pill'>{int(c['progress'])}%</span>"
                f"</div><div class='small'>{when}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.header("⚙️ Settings")
    models = list(dict.fromkeys([DEFAULT_MODEL, "gpt-4o-mini", "gpt-4.1", "gpt-5-mini"]))
    model = st.selectbox("Model", options=models, index=0)
    is_admin = bool(st.session_state.usage.get("bypass"))
    remain = "∞" if is_admin else max(0, USAGE_DAILY_LIMIT - st.session_state.usage["count"])
    c1, c2 = st.columns(2)
    c1.metric("Generations left", remain)
    c2.metric("Cooldown (s)", USAGE_COOLDOWN_SECONDS)
    if is_admin: st.success("Admin bypass active")

# ================== HOME PAGE ==================
def page_home():
    st.title("📈 Sales Progression Tracker (UK)")
    st.caption("Create a new case or open one from the sidebar.")
    st.markdown("### Create a new case")
    name = st.text_input("Case name / address", placeholder="e.g., 20 Maunder Close, RM16 6BB")
    if st.button("Create case", type="primary", disabled=not bool(name.strip())):
        cid = db_create_case(UID, name.strip())
        set_route("case", cid)

    st.markdown("---")
    st.markdown("### Quick start")
    st.write("Paste emails, upload documents (.txt, .pdf, .docx), or describe the process. "
             "The app builds a timeline, progress bar, and a client-ready status report with ‘Action to take’ and ‘Waiting for’.")

# ================== CASE PAGE ==================
def _input_widgets():
    tabs = st.tabs(["Emails / Notes", "Upload Documents", "Describe Process"])
    with tabs[0]:
        emails_text = st.text_area("Emails or notes (any order)", height=240, key="emails_text_case")
    with tabs[1]:
        uploads = st.file_uploader("Attach files (.txt, .pdf, .docx)", type=["txt","pdf","docx"], accept_multiple_files=True, key="uploads_case")
    with tabs[2]:
        desc_text = st.text_area("Describe what's happened so far (include dates where possible)", height=200, key="desc_text_case")
    return emails_text, uploads, desc_text

def _build_blob_from_inputs(emails_text, uploads, desc_text) -> Tuple[str, str]:
    source, blob = "emails", ""
    if emails_text and emails_text.strip():
        blob = emails_text.strip()
    elif uploads:
        source = "docs"
        parts = []
        for up in uploads[:8]:
            txt = read_file_to_text(up)
            if txt: parts.append(txt)
        blob = "\n\n".join(parts).strip()
    elif desc_text and desc_text.strip():
        source = "description"
        blob = desc_text.strip()
    return source, blob

def page_case(case_id: int):
    # Header + rename/delete
    cases = db_list_cases(UID)
    case = next((c for c in cases if c["id"] == case_id), None)
    if not case:
        st.error("Case not found or not yours.")
        st.stop()

    st.title(f"📄 {case['name']}")
    created = dt.datetime.fromtimestamp(case["ts"]).strftime("%d %b %Y %H:%M")
    st.caption(f"Created {created}")

    colA, colB, colC = st.columns([2,1,1])
    new_name = colA.text_input("Rename case", value=case["name"])
    if colB.button("Save name"):
        if new_name.strip():
            db_rename_case(UID, case_id, new_name.strip())
            st.success("Name updated.")
            st.experimental_rerun()
    if colC.button("Delete case", type="secondary"):
        db_delete_case(UID, case_id)
        st.warning("Case deleted.")
        set_route("home")

    st.markdown("---")
    st.subheader("Add new information to this case")
    emails_text, uploads, desc_text = _input_widgets()

    # Clear labels replacing earlier confusing toggles
    col1, col2, col3 = st.columns([1,1,2])
    merge_duplicates = col1.checkbox("Merge duplicate updates", value=True)
    require_date_for_done = col2.checkbox('Only mark a step "Done" if a date is present', value=False)
    sla_days = col3.number_input("Highlight pending older than (days)", min_value=0, max_value=120, value=DEFAULT_SLA_PENDING_DAYS, step=1)

    if st.button("🔎 Analyse & Update Case", type="primary", use_container_width=True):
        source, blob = _build_blob_from_inputs(emails_text, uploads, desc_text)
        if not blob:
            st.error("Provide emails, upload docs, or enter a description.")
            st.stop()

        # Save raw input to case
        db_add_input(case_id, source, blob)

        # Enforce caps / cooldown
        msg = enforce_caps()
        if msg: st.warning(msg); st.stop()

        # Build full context from all inputs for this case and extract
        all_text = db_all_input_text(case_id)
        with st.spinner("Extracting milestones…"):
            try:
                j = chat_json(model, SYSTEM_LAW, prompt_extract(all_text))
                raw_rows = (j or {}).get("milestones") or []
            except Exception as e:
                st.error(f"Model error: {e}"); st.stop()

        rows = normalize_rows(raw_rows, merge_duplicates, require_date_for_done)
        progress = compute_progress(rows)

        # Summary
        summary_text = ""
        bullets = []
        try:
            s_json = chat_json(model, SYSTEM_LAW, prompt_summary(rows), max_tokens=600)
            if isinstance(s_json, dict):
                summary_text = str(s_json.get("summary") or s_json.get("text") or "").strip()
                b = s_json.get("bullets")
                if isinstance(b, list): bullets = [str(x) for x in b if str(x).strip()]
        except Exception:
            pass

        # Save snapshot
        db_save_snapshot(case_id, rows, progress, summary_text)

        # Update usage counters
        st.session_state.usage["count"] += 1
        st.session_state.usage["last_ts"] = time.time()

        st.success("Case updated.")
        st.experimental_rerun()

    # ======= Current snapshot =======
    snap = db_last_snapshot(case_id)
    if not snap:
        st.info("No analysis yet. Add information above and click “Analyse & Update Case”.")
        return

    rows = snap["milestones"]
    df = pd.DataFrame(rows)
    pct = float(snap["progress"])
    summary_text = snap.get("summary") or ""
    done_count = int((df["status"]=="done").sum()) if not df.empty else 0
    pending_count = int((df["status"]=="pending").sum()) if not df.empty else 0
    blocked_count = int((df["status"]=="blocked").sum()) if not df.empty else 0

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><b>Progress</b><br><span style='font-size:1.25rem'>{pct}%</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><b>Done</b><br><span style='font-size:1.25rem'>{done_count}</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><b>Pending</b><br><span style='font-size:1.25rem'>{pending_count}</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><b>Blocked</b><br><span style='font-size:1.25rem'>{blocked_count}</span></div>", unsafe_allow_html=True)

    st.progress(int(pct))
    st.markdown(f"<div class='progress-label'><span>Start</span><span>{pct:.1f}%</span><span>Completion</span></div>", unsafe_allow_html=True)

    left, right = st.columns([2,1], gap="large")

    # Timeline
    with right:
        st.markdown("### Timeline")
        if df.empty:
            st.info("No milestones detected yet.")
        else:
            def _key(row):
                d = row.get("date")
                try: t = dt.datetime.fromisoformat(d).timestamp() if d else float("inf")
                except: t = float("inf")
                rank = {"done":0,"pending":1,"blocked":2}.get(row.get("status","pending"),3)
                return (t,rank,row.get("stage",""))
            df["__k__"] = df.apply(_key, axis=1)
            df = df.sort_values("__k__").drop(columns="__k__")

            def badge(s): return {"done":"✅","pending":"🕒","blocked":"⛔"}.get(s,"•")
            for _, r in df.iterrows():
                date_txt = f" — {r['date']}" if r.get("date") else ""
                actor = (r.get("actor") or "").capitalize()
                det = (r.get("details") or "").strip()
                st.markdown(f"**{badge(r['status'])} {r['stage']}**{date_txt}  \n"
                            f"<span class='small'>{actor}</span>  \n*{det}*", unsafe_allow_html=True)

            # SLA highlights
            today = dt.date.today()
            aged = []
            if sla_days and sla_days > 0:
                for _, r in df[df["status"]=="pending"].iterrows():
                    d = r.get("date")
                    if d:
                        try:
                            days = (today - dt.date.fromisoformat(d)).days
                            if days >= sla_days: aged.append((r["stage"], days))
                        except: pass
            if aged:
                st.markdown("<hr/>", unsafe_allow_html=True)
                st.warning("Pending items older than SLA:")
                for s, days in aged:
                    st.write(f"- {s}: {days} days")

    # Status report + table + exports
    with left:
        st.markdown("### Status Report")
        if summary_text: st.write(summary_text)

        # Build a “Next steps” section from rows (Action to take / Waiting for)
        actions, waiting = [], []
        for r in rows:
            if (r.get("next_action") or "").strip():
                actions.append(f"{r['stage']}: {r['next_action']}")
            if (r.get("blockers") or "").strip():
                waiting.append(f"{r['stage']}: {r['blockers']}")
        if actions or waiting:
            st.markdown("**Action to take / Waiting for**")
            for a in actions: st.markdown(f"- {a}")
            for w in waiting: st.markdown(f"- {w}")

        st.markdown("### Milestones (table)")
        st.dataframe(df, use_container_width=True, height=340)

        dump = {"case_id": case_id, "progress": pct, "milestones": rows}
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ JSON", data=json.dumps(dump, indent=2).encode("utf-8"),
                           file_name=f"case_{case_id}_progress.json", mime="application/json", use_container_width=True)
        c2.download_button("⬇️ CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"case_{case_id}_milestones.csv", mime="text/csv", use_container_width=True)

# ================== Router ==================
page, case_id = get_route()
if page == "case" and case_id:
    page_case(case_id)
else:
    page_home()

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🔐 Gumroad-locked (admin override available). Cases persist per user key. © Relura")
