# app.py — Sales Progression Tracker (UK, Pro)
# Emails OR Doc uploads OR Free-text description
# Corporate UI: KPIs, timeline, progress bar, blockers / next-actions
# Robust parsing + stage normalization (no schema crashes)
# Gumroad gate + Admin override + daily caps
# SQLite persistence per user (history across sessions)

import os, re, io, json, time, hashlib, sqlite3, datetime as dt
import urllib.parse, urllib.request
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st
from openai import OpenAI

# Optional readers (only load if used)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# -------------------- Config & Secrets --------------------
st.set_page_config(page_title="Sales Progression Tracker", page_icon="📈", layout="wide")

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
DEFAULT_MODEL  = get_secret("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
GUMROAD_PRODUCT_PERMALINK = get_secret("GUMROAD_PRODUCT_PERMALINK", "")
ADMIN_BYPASS   = get_secret("ADMIN_BYPASS", "")
USAGE_DAILY_LIMIT      = int(get_secret("USAGE_DAILY_LIMIT", "150"))
USAGE_COOLDOWN_SECONDS = int(get_secret("USAGE_COOLDOWN_SECONDS", "5"))
SLA_PENDING_DAYS       = int(get_secret("SLA_PENDING_DAYS", "10"))

if not OPENAI_API_KEY:
    st.error("Server misconfigured: missing OPENAI_API_KEY in Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Styles (high-contrast in light & dark) --------------------
st.markdown("""
<style>
/* Theme-aware variables */
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

/* Layout + cards */
.block-container { padding-top: 1.1rem; }
.card, .kpi {
  border:1px solid var(--border);
  background:var(--card);
  border-radius:16px; padding:16px; box-shadow:var(--shadow);
}
.card, .card * { color: var(--fg) !important; text-shadow:none !important; opacity:1 !important; }
.kpi, .kpi * { color: var(--fg) !important; text-shadow:none !important; opacity:1 !important; }

.small { color:var(--muted) !important; font-size:.9rem }
h1, h2, h3 { letter-spacing:.2px }
div.stButton>button, div.stDownloadButton>button { border-radius:10px; padding:.6rem 1rem; }
hr { border:none; border-top:1px solid var(--border); margin: 18px 0; }
.progress-label { display:flex; justify-content:space-between; font-size:.9rem; color:var(--muted); margin-top:6px;}
</style>
""", unsafe_allow_html=True)

# -------------------- Session --------------------
if "licensed" not in st.session_state:
    st.session_state.licensed = False
if "usage" not in st.session_state:
    st.session_state.usage = {"date": dt.date.today().isoformat(), "count": 0, "last_ts": 0.0, "bypass": False}
if "uid" not in st.session_state:
    st.session_state.uid = None  # set after unlock

# -------------------- Gumroad gate --------------------
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
    st.caption("Unlock with your Gumroad key. Admins can use an override code.")
    with st.form("license"):
        access = st.text_input("Access Key", type="password")
        ok = st.form_submit_button("Unlock", use_container_width=True)
    if ok:
        # Admin bypass
        if ADMIN_BYPASS and access.strip() == ADMIN_BYPASS.strip():
            st.session_state.usage["bypass"] = True
            st.session_state.licensed = True
            st.session_state.uid = "admin:" + hashlib.sha256(access.encode()).hexdigest()[:16]
            st.success("Admin override accepted ✅")
            st.rerun()
        # Gumroad verify
        if not GUMROAD_PRODUCT_PERMALINK:
            st.error("Missing GUMROAD_PRODUCT_PERMALINK in Secrets.")
            st.stop()
        valid = verify_gumroad_license(access.strip(), GUMROAD_PRODUCT_PERMALINK)
        if valid:
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

# -------------------- Persistence (SQLite) --------------------
DB_PATH = "progress.db"

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid TEXT NOT NULL,
        name TEXT NOT NULL,
        created_ts INTEGER NOT NULL,
        input_text TEXT,
        source TEXT,
        milestones_json TEXT,
        progress REAL,
        summary TEXT
    )""")
    conn.commit()
    return conn

def db_save_case(uid: str, name: str, in_text: str, source: str,
                 milestones: List[Dict[str,Any]], progress: float, summary: str) -> int:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO cases(uid,name,created_ts,input_text,source,milestones_json,progress,summary) VALUES (?,?,?,?,?,?,?,?)",
                (uid, name, int(time.time()), in_text, source, json.dumps(milestones), float(progress), summary))
    conn.commit()
    return cur.lastrowid

def db_list_cases(uid: str, limit: int = 25) -> List[Dict[str,Any]]:
    conn = db_conn()
    rows = conn.execute("SELECT id,name,created_ts,progress FROM cases WHERE uid=? ORDER BY id DESC LIMIT ?", (uid, limit)).fetchall()
    return [{"id":r[0],"name":r[1],"ts":r[2],"progress":r[3]} for r in rows]

# -------------------- Canonical stages & mapping --------------------
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

# -------------------- Model prompts --------------------
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

# -------------------- LLM helpers --------------------
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

# -------------------- Utilities --------------------
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

def normalize_rows(rows: List[Dict[str, Any]], dedupe: bool, strict_dates: bool) -> List[Dict[str, Any]]:
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
        if strict_dates and status=="done" and not date_iso:
            status = "pending"
        key = (stage, status, date_iso or "", actor, details[:90])
        if dedupe and key in seen: continue
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

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    models = list(dict.fromkeys([DEFAULT_MODEL, "gpt-4o-mini", "gpt-4.1", "gpt-5-mini"]))
    model = st.selectbox("Model", options=models, index=0)
    st.markdown("---")
    is_admin = bool(st.session_state.usage.get("bypass"))
    remain = "∞" if is_admin else max(0, USAGE_DAILY_LIMIT - st.session_state.usage["count"])
    c1, c2 = st.columns(2)
    c1.metric("Generations left", remain)
    c2.metric("Cooldown (s)", USAGE_COOLDOWN_SECONDS)
    if is_admin: st.success("Admin bypass active")
    st.markdown("---")
    st.subheader("📁 Your saved cases")
    for it in db_list_cases(st.session_state.uid or "anon", 25):
        when = dt.datetime.fromtimestamp(it["ts"]).strftime("%d %b %Y %H:%M")
        st.write(f"• **{it['name']}** — {it['progress']}%  \n<span class='small'>{when}</span>", unsafe_allow_html=True)

# -------------------- Input Modes --------------------
st.title("📈 Sales Progression Tracker (UK)")
st.caption("Paste emails, upload docs, or describe the process. Get progress, timeline, and next steps.")

tabs = st.tabs(["Emails / Notes", "Upload Documents", "Describe Process"])
with tabs[0]:
    emails_text = st.text_area("Emails or notes (any order)", height=280, key="emails_text")
with tabs[1]:
    uploads = st.file_uploader("Attach files (.txt, .pdf, .docx)", type=["txt","pdf","docx"], accept_multiple_files=True)
with tabs[2]:
    desc_text = st.text_area("Describe what's happened so far, with dates where possible", height=220, key="desc_text")

st.markdown("---")
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    case_name = st.text_input("Case name / address (for saving)", placeholder="e.g., 20 Maunder Close, RM16 6BB")
with colB:
    dedupe = st.checkbox("De-duplicate", value=True)
with colC:
    strict_dates = st.checkbox("Require date for 'done'", value=False)
with colD:
    show_age = st.checkbox("Highlight aged pending", value=True)

go = st.button("🔎 Analyse & Generate Report", type="primary", use_container_width=True)

# -------------------- Run --------------------
if go:
    # Gather input
    source = "emails"; blob = ""
    if emails_text and emails_text.strip():
        blob = emails_text.strip()
    elif uploads:
        source = "docs"
        parts = []
        for up in uploads[:5]:
            txt = read_file_to_text(up)
            if txt: parts.append(txt)
        blob = "\n\n".join(parts).strip()
    elif desc_text and desc_text.strip():
        source = "description"; blob = desc_text.strip()
    else:
        st.error("Provide emails, upload documents, or enter a description first.")
        st.stop()

    msg = enforce_caps()
    if msg: st.warning(msg); st.stop()

    with st.spinner("Extracting milestones…"):
        try:
            j = chat_json(model, SYSTEM_LAW, prompt_extract(blob))
            raw_rows = (j or {}).get("milestones") or []
        except Exception as e:
            st.error(f"Model error: {e}"); st.stop()

    rows = normalize_rows(raw_rows, dedupe=dedupe, strict_dates=strict_dates)
    df = pd.DataFrame(rows)

    pct = compute_progress(rows)

    # Executive summary (guard if model returns non-JSON)
    summary_text, bullets = "", []
    try:
        s_json = chat_json(model, SYSTEM_LAW, prompt_summary(rows), max_tokens=600)
        if isinstance(s_json, dict):
            summary_text = str(s_json.get("summary") or s_json.get("text") or "").strip()
            b = s_json.get("bullets")
            if isinstance(b, list): bullets = [str(x) for x in b if str(x).strip()]
    except Exception:
        pass

    # Update usage
    st.session_state.usage["count"] += 1
    st.session_state.usage["last_ts"] = time.time()

    case_id = None
    if case_name.strip():
        case_id = db_save_case(st.session_state.uid or "anon", case_name.strip(), blob[:4000], source, rows, pct, summary_text)

    # ======= RESULTS =======
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><b>Progress</b><br><span style='font-size:1.25rem'>{pct}%</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><b>Done</b><br><span style='font-size:1.25rem'>{int((df['status']=='done').sum())}</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><b>Pending</b><br><span style='font-size:1.25rem'>{int((df['status']=='pending').sum())}</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><b>Blocked</b><br><span style='font-size:1.25rem'>{int((df['status']=='blocked').sum())}</span></div>", unsafe_allow_html=True)

    st.progress(int(pct))
    st.markdown(f"<div class='progress-label'><span>Start</span><span>{pct:.1f}%</span><span>Completion</span></div>", unsafe_allow_html=True)

    left, right = st.columns([2,1], gap="large")

    with right:
        st.markdown("### Timeline")
        def badge(s): return {"done":"✅","pending":"🕒","blocked":"⛔"}.get(s,"•")
        if df.empty:
            st.info("No milestones detected.")
        else:
            # order for readability
            def _key(row):
                d = row.get("date")
                try: t = dt.datetime.fromisoformat(d).timestamp() if d else float("inf")
                except: t = float("inf")
                rank = {"done":0,"pending":1,"blocked":2}.get(row.get("status","pending"),3)
                return (t,rank,row.get("stage",""))
            df["__k__"] = df.apply(_key, axis=1)
            df = df.sort_values("__k__").drop(columns="__k__")

            for _, r in df.iterrows():
                date_txt = f" — {r['date']}" if r.get("date") else ""
                actor = (r.get("actor") or "").capitalize()
                det = (r.get("details") or "").strip()
                st.markdown(f"**{badge(r['status'])} {r['stage']}**{date_txt}  \n"
                            f"<span class='small'>{actor}</span>  \n*{det}*", unsafe_allow_html=True)

            if show_age:
                today = dt.date.today()
                aged = []
                for _, r in df[df["status"]=="pending"].iterrows():
                    d = r.get("date")
                    if d:
                        try:
                            days = (today - dt.date.fromisoformat(d)).days
                            if days >= SLA_PENDING_DAYS: aged.append((r["stage"], days))
                        except: pass
                if aged:
                    st.markdown("<hr/>", unsafe_allow_html=True)
                    st.warning("Pending items older than SLA:")
                    for s, days in aged:
                        st.write(f"- {s}: {days} days")

    with left:
        st.markdown("### Status Report")
        if summary_text:
            st.write(summary_text)
        if bullets:
            st.markdown("**Action to take / Waiting for**")
            for b in bullets: st.markdown(f"- {b}")

        st.markdown("### Milestones (table)")
        st.dataframe(df, use_container_width=True, height=340)

        # Exports
        dump = {"source": source, "progress": pct, "milestones": rows}
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ JSON", data=json.dumps(dump, indent=2).encode("utf-8"),
                           file_name="progression.json", mime="application/json", use_container_width=True)
        c2.download_button("⬇️ CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="progression.csv", mime="text/csv", use_container_width=True)

        if case_id:
            st.success(f"Saved as case #{case_id} — “{case_name}”. View in sidebar.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🔐 Gumroad-locked (admin override available). Cases saved per user key. © Relura")
