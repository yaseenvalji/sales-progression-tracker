# app.py — Sales Progression Tracker (UK)
# - Gumroad license gate + admin override
# - Uses your master OPENAI_API_KEY (kept in Streamlit Secrets)
# - Daily usage cap + cooldown + midnight reset
# - Paste email threads → extract conveyancing milestones → timeline + CSV/JSON

import os
import json
import time
import datetime as dt
import urllib.parse
import urllib.request
from typing import List, Optional, Literal, Dict, Any

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ================== Secrets & Config ==================
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read from Streamlit Secrets first, then env, then default."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

# Required
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
DEFAULT_MODEL  = get_secret("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

# License gate
GUMROAD_PRODUCT_PERMALINK = get_secret("GUMROAD_PRODUCT_PERMALINK", "")
ADMIN_BYPASS              = get_secret("ADMIN_BYPASS", "")  # leave blank to disable

# Usage controls
USAGE_DAILY_LIMIT      = int(get_secret("USAGE_DAILY_LIMIT", "100"))
USAGE_COOLDOWN_SECONDS = int(get_secret("USAGE_COOLDOWN_SECONDS", "5"))

# ================== Page & Styles ==================
st.set_page_config(page_title="Sales Progression Tracker", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.25rem; }
.result-card { border:1px solid #e8e8e8; border-radius:12px; padding:14px; background:#fff; }
.small { color:#666; font-size:0.9rem; }
div.stButton > button, div.stDownloadButton > button { border-radius: 10px; padding: 0.55rem 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Sales Progression Tracker (UK)")
if not st.session_state.get("licensed", False):
    st.caption("Unlock with your Gumroad key. Admins can use a private override code.")

# ================== Guards ==================
if not OPENAI_API_KEY:
    st.error("Server misconfigured: missing OPENAI_API_KEY (add it in Streamlit Secrets).")
    st.stop()

# ================== Session State ==================
if "licensed" not in st.session_state:
    st.session_state.licensed = False
if "usage" not in st.session_state:
    st.session_state.usage = {
        "date": dt.date.today().isoformat(),
        "count": 0,
        "last_ts": 0.0,
        "bypass": False,  # True after admin override
    }

# ================== License Helpers ==================
def verify_gumroad_license(license_key: str, product_permalink: str) -> bool:
    """Return True if the Gumroad license is valid for the product. No external deps."""
    try:
        data = urllib.parse.urlencode({
            "product_permalink": product_permalink,
            "license_key": license_key,
            "increment_uses_count": "false",
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.gumroad.com/v2/licenses/verify",
            data=data, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            j = json.loads(resp.read().decode("utf-8"))
        return bool(j.get("success"))
    except Exception:
        return False

def show_license_gate():
    st.info("🔒 Enter your Access Key to unlock.", icon="🔑")
    with st.form("license_form"):
        access_key = st.text_input("Access Key", placeholder="Your Gumroad key", type="password")
        ok = st.form_submit_button("Unlock")

    if ok:
        # Admin override first
        if ADMIN_BYPASS and access_key.strip() == ADMIN_BYPASS.strip():
            st.session_state.usage["bypass"] = True
            st.session_state.licensed = True
            st.success("Admin override accepted ✅")
            st.rerun()

        # Otherwise validate with Gumroad
        if not GUMROAD_PRODUCT_PERMALINK:
            st.error("Server misconfigured: missing GUMROAD_PRODUCT_PERMALINK in Secrets.")
            st.stop()

        valid = verify_gumroad_license(access_key.strip(), GUMROAD_PRODUCT_PERMALINK)
        if valid:
            st.session_state.licensed = True
            st.success("License verified ✅")
            st.rerun()
        else:
            st.error("Invalid access key. Please check your key or contact support.")

# Show gate if not yet licensed
if not st.session_state.licensed:
    show_license_gate()
    if not st.session_state.licensed:
        st.stop()

# ================== OpenAI Client ==================
client = OpenAI(api_key=OPENAI_API_KEY)

# ================== Sidebar (Models & Usage) ==================
with st.sidebar:
    st.header("⚙️ Settings")

    # De-duplicated models (DEFAULT first)
    models = list(dict.fromkeys([DEFAULT_MODEL, "gpt-4o-mini", "gpt-4.1", "gpt-5-mini"]))
    model = st.selectbox("Model", options=models, index=0)

    st.markdown("---")
    # Midnight reset for local timezone
    today = dt.date.today().isoformat()
    if st.session_state.usage["date"] != today:
        st.session_state.usage["date"] = today
        st.session_state.usage["count"] = 0
        st.session_state.usage["last_ts"] = 0.0

    is_admin = bool(st.session_state.usage.get("bypass"))
    remaining = "∞" if is_admin else max(0, USAGE_DAILY_LIMIT - st.session_state.usage["count"])
    st.metric(label="Generations left (today)", value=remaining)
    if is_admin:
        st.success("Admin bypass active")

# ================== Schema (Pydantic) ==================
StageName = Literal[
    "Offer Received","Offer Accepted","Memorandum of Sale",
    "ID/AML Checks","Mortgage Application","Mortgage Offer",
    "Survey / Valuation","Searches Ordered","Searches Returned",
    "Draft Contracts Issued","Enquiries Raised","Enquiries Answered",
    "Exchange of Contracts","Completion","Fall-through","Other"
]

class Milestone(BaseModel):
    stage: StageName
    status: Literal["done","pending","blocked"] = Field(description="Current status")
    date_iso: Optional[str] = Field(default=None, description="YYYY-MM-DD if known")
    actor: Optional[str] = Field(default=None, description="agent|solicitor|buyer|seller|lender|other")
    details: Optional[str] = None
    blockers: Optional[str] = None
    next_action: Optional[str] = None
    confidence: int = Field(ge=0, le=100, default=70)
    source_quote: Optional[str] = None

class Extraction(BaseModel):
    property_ref: Optional[str] = None
    buyer: Optional[str] = None
    seller: Optional[str] = None
    milestones: List[Milestone]

# ================== Prompt Builder (quote-safe) ==================
SYSTEM = "You are a UK residential conveyancing assistant. Extract structured milestones. Never invent dates or outcomes."

def build_prompt(thread_text: str) -> str:
    thread = (thread_text or "").strip()[:100_000]
    # No nested triple-quotes; safe string assembly.
    return (
        "Parse the following email/notes thread about a property sale progression.\n"
        "Return ONLY JSON matching the schema shown after the thread.\n\n"
        "Rules:\n"
        "- Use UK conveyancing stages when possible.\n"
        "- Only set status=\"done\" if the email clearly confirms completion.\n"
        "- Use ISO dates (YYYY-MM-DD) when a date is explicit; otherwise leave null.\n"
        "- Keep details factual; include one short source_quote if helpful.\n\n"
        "THREAD START >>>\n"
        f"{thread}\n"
        "<<< THREAD END\n\n"
        "JSON schema (shape, not types):\n"
        "{\n"
        '  "property_ref": "... or null",\n'
        '  "buyer": "... or null",\n'
        '  "seller": "... or null",\n'
        '  "milestones": [\n'
        "     {\n"
        '       "stage": "...",\n'
        '       "status": "done|pending|blocked",\n'
        '       "date_iso": "YYYY-MM-DD or null",\n'
        '       "actor": "agent|solicitor|buyer|seller|lender|other",\n'
        '       "details": "...",\n'
        '       "blockers": "... or null",\n'
        '       "next_action": "... or null",\n'
        '       "confidence": 0-100,\n'
        '       "source_quote": "... or null"\n'
        "     }\n"
        "  ]\n"
        "}\n"
    )

# ================== UI ==================
with st.form("extract_form", clear_on_submit=False):
    st.subheader("Paste Email/Notes Thread")
    sample = st.toggle("Load sample thread")
    if sample:
        SAMPLE = (
            "From: Buyer’s Solicitor\n"
            "Subject: Searches ordered\n"
            "We confirm local authority and water searches were ordered on 12/08/2025. "
            "Results expected within 10 working days.\n\n"
            "From: Lender\n"
            "Subject: Valuation Report\n"
            "The valuation was carried out on 10/08/2025. Report due 13/08/2025.\n\n"
            "From: Agent\n"
            "Subject: Offer accepted\n"
            "Vendor accepted buyer’s offer of £425,000 on 08/08/2025. "
            "Memorandum of Sale issued to both solicitors.\n\n"
            "From: Seller’s Solicitor\n"
            "Subject: Draft contract pack\n"
            "Draft contracts issued 15/08/2025. Awaiting enquiries from buyer’s side."
        )
        st.session_state["thread_text"] = SAMPLE

    thread_text = st.text_area(
        "Email / notes thread",
        value=st.session_state.get("thread_text", ""),
        height=340,
        placeholder="Paste solicitor/agent emails here (any order). Dates help: e.g., 'issued 15/08/2025'.",
    )

    submitted = st.form_submit_button("🔎 Extract Milestones", type="primary", use_container_width=True)

# ================== Caps + Cooldown ==================
def enforce_usage_caps() -> Optional[str]:
    """Returns an error message if a cap is violated, else None."""
    usage = st.session_state.usage
    now = time.time()

    if not usage.get("bypass"):  # skip caps for admin override
        if usage["count"] >= USAGE_DAILY_LIMIT:
            reset_at = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), dt.time.min)
            mins_left = int((reset_at - dt.datetime.now()).total_seconds() // 60)
            return f"Daily limit reached. Resets in ~{mins_left} minutes."

        since = now - float(usage["last_ts"])
        if since < USAGE_COOLDOWN_SECONDS:
            wait = int(USAGE_COOLDOWN_SECONDS - since + 1)
            return f"Please wait {wait}s before generating again."

    # Update counters later only on success
    return None

# ================== Extraction ==================
def extract(thread_text: str) -> Extraction:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(thread_text)},
        ],
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    return Extraction.model_validate(data)

def to_df(ex: Extraction) -> pd.DataFrame:
    rows = []
    for m in ex.milestones:
        rows.append({
            "stage": m.stage,
            "status": m.status,
            "date": m.date_iso,
            "actor": m.actor,
            "details": m.details,
            "blockers": m.blockers,
            "next_action": m.next_action,
            "confidence": m.confidence,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort by date (known first), then status, then stage
    def sort_key(row):
        d = row.get("date")
        try:
            t = dt.datetime.fromisoformat(d).timestamp() if d else float("inf")
        except Exception:
            t = float("inf")
        status_rank = {"done": 0, "pending": 1, "blocked": 2}.get(str(row.get("status")), 3)
        return (t, status_rank, str(row.get("stage")))

    df["__k__"] = df.apply(sort_key, axis=1)
    df = df.sort_values("__k__").drop(columns="__k__")
    return df

# ================== Submit Logic ==================
if submitted:
    if not thread_text.strip():
        st.error("Paste an email thread first.")
        st.stop()

    msg = enforce_usage_caps()
    if msg:
        st.warning(msg)
        st.stop()

    with st.spinner("Extracting milestones…"):
        try:
            ex = extract(thread_text)
            df = to_df(ex)
        except ValidationError as ve:
            st.error("Model output didn’t match the schema. Try again.")
            st.code(str(ve))
            st.stop()
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    # Update usage counters (success only)
    st.session_state.usage["count"] += 1
    st.session_state.usage["last_ts"] = time.time()

    # Render
    st.subheader("Results")
    if df.empty:
        st.info("No milestones found. Try including dates or clearer statements in the thread.")
    else:
        colL, colR = st.columns([2, 1], gap="large")

        with colR:
            st.subheader("Timeline")
            def badge(s): return {"done": "✅", "pending": "🕒", "blocked": "⛔"}.get(str(s), "•")
            for _, r in df.iterrows():
                date_txt = f" — {r['date']}" if r.get("date") else ""
                details  = (str(r.get("details") or "")).strip()
                st.markdown(f"**{badge(r['status'])} {r['stage']}**{date_txt}  \n*{details}*")
            st.markdown("---")
            done = int((df["status"] == "done").sum())
            pending = int((df["status"] == "pending").sum())
            blocked = int((df["status"] == "blocked").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Done", done); c2.metric("Pending", pending); c3.metric("Blocked", blocked)

        with colL:
            st.subheader("Milestones (table)")
            st.dataframe(df, use_container_width=True, height=280)

            # Downloads
            dump = ex.model_dump()
            c1, c2, c3 = st.columns(3)
            c1.download_button("⬇️ JSON", data=json.dumps(dump, indent=2).encode("utf-8"),
                               file_name="progression.json", mime="application/json",
                               use_container_width=True)
            c2.download_button("⬇️ CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="progression.csv", mime="text/csv",
                               use_container_width=True)
            def clear_state():
                st.session_state["thread_text"] = ""
            c3.button("🔁 Clear", on_click=clear_state, use_container_width=True)

st.markdown("---")
st.caption("🔐 Locked by Gumroad key (with admin override). Contact: support@yourdomain.com")

