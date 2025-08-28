import os
import json
import datetime as dt
from typing import List, Optional, Literal, Dict, Any
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ---------- Config ----------
st.set_page_config(page_title="Sales Progression Tracker", page_icon="📈", layout="wide")

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

API_KEY = get_secret("OPENAI_API_KEY")
DEFAULT_MODEL = get_secret("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

if not API_KEY:
    st.warning("Add OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable before extracting.")
client = OpenAI(api_key=API_KEY)

# ---------- Schema ----------
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
    actor: Optional[str] = Field(default=None, description="Who’s responsible next (agent/solicitor/buyer/seller/lender)")
    details: Optional[str] = Field(default=None, description="Short factual summary")
    blockers: Optional[str] = None
    next_action: Optional[str] = None
    confidence: int = Field(ge=0, le=100, default=70)
    source_quote: Optional[str] = None

class Extraction(BaseModel):
    property_ref: Optional[str] = None
    buyer: Optional[str] = None
    seller: Optional[str] = None
    milestones: List[Milestone]

# ---------- Prompt ----------
SYSTEM = "You are a UK residential conveyancing assistant. Extract structured milestones from messages. Never invent dates or outcomes."

def build_prompt(thread_text: str) -> str:
    return f"""
Parse the following email/notes thread about a property sale progression.
Return **only** JSON matching the schema shown after the thread.

Rules:
- Use UK conveyancing stages when possible.
- Only set status="done" if the email clearly confirms completion.
- Use ISO dates (YYYY-MM-DD) when a date is explicit. Otherwise leave null.
- Keep details factual; include one short source_quote where helpful.

THREAD:
"""{thread_text.strip()[:100000]}"""

JSON schema (shape, not types):
{{
  "property_ref": "... or null",
  "buyer": "... or null",
  "seller": "... or null",
  "milestones": [
     {{
       "stage": "...",
       "status": "done|pending|blocked",
       "date_iso": "YYYY-MM-DD or null",
       "actor": "agent|solicitor|buyer|seller|lender|other",
       "details": "...",
       "blockers": "... or null",
       "next_action": "... or null",
       "confidence": 0-100,
       "source_quote": "... or null"
     }}
  ]
}}
""".strip()

# ---------- UI ----------
st.title("📈 Sales Progression Tracker (UK)")
st.caption("Paste solicitor/agent emails below. The app extracts milestones and builds a timeline.")

colL, colR = st.columns([2,1], gap="large")

with colL:
    sample = st.toggle("Load sample thread")
    if sample:
        SAMPLE = """
From: Buyer’s Solicitor
Subject: Searches ordered
We confirm local authority and water searches were ordered on 12/08/2025. Results expected within 10 working days.

From: Lender
Subject: Valuation Report
The valuation was carried out on 10/08/2025. Report due 13/08/2025.

From: Agent
Subject: Offer accepted
Vendor accepted buyer’s offer of £425,000 on 08/08/2025. Memorandum of Sale issued to both solicitors.

From: Seller’s Solicitor
Subject: Draft contract pack
Draft contracts issued 15/08/2025. Awaiting enquiries from buyer’s side.
"""
        st.session_state["thread_text"] = SAMPLE
    thread_text = st.text_area("Email / notes thread", value=st.session_state.get("thread_text",""),
                               height=360, placeholder="Paste emails in chronological or mixed order…")

    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", [DEFAULT_MODEL, "gpt-4o-mini","gpt-4.1","gpt-5-mini"], index=0)
    with col2:
        run = st.button("Extract Milestones", type="primary", use_container_width=True)

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

    def sort_key(row):
        d = row["date"]
        try:
            t = dt.datetime.fromisoformat(d).timestamp() if d else 1e15
        except Exception:
            t = 1e15
        status_rank = {"done":0, "pending":1, "blocked":2}.get(row["status"], 3)
        return (t, status_rank, row["stage"])

    if not df.empty:
        df["__k__"] = df.apply(sort_key, axis=1)
        df = df.sort_values("__k__").drop(columns="__k__")
    return df

with colR:
    st.subheader("Timeline")
    placeholder = st.empty()

if run:
    if not thread_text.strip():
        st.error("Paste an email thread first.")
    elif not API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in .streamlit/secrets.toml or your environment.")
        st.stop()
    else:
        with st.spinner("Extracting milestones…"):
            try:
                ex = extract(thread_text)
                df = to_df(ex)
                st.session_state["extraction"] = ex.model_dump()
                st.session_state["df"] = df
            except ValidationError as ve:
                st.error("Model output didn’t match the schema. Try again.")
                st.text(str(ve))
            except Exception as e:
                st.error(f"Extraction failed: {e}")

# ---------- Render results ----------
ex_dump: Dict[str, Any] = st.session_state.get("extraction")
df: Optional[pd.DataFrame] = st.session_state.get("df")

if ex_dump and df is not None:
    with colR:
        def badge(s):
            return {"done":"✅","pending":"🕒","blocked":"⛔"}.get(s,"•")
        for _, r in df.iterrows():
            st.markdown(f"**{badge(r['status'])} {r['stage']}**  "
                        f"{'— '+str(r['date']) if r['date'] else ''}  \n"
                        f"*{(str(r['details'] or '')).strip()}*")
        st.markdown("---")
        done = int((df["status"]=="done").sum()) if "status" in df else 0
        pending = int((df["status"]=="pending").sum()) if "status" in df else 0
        blocked = int((df["status"]=="blocked").sum()) if "status" in df else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Done", done)
        c2.metric("Pending", pending)
        c3.metric("Blocked", blocked)

    with colL:
        st.subheader("Milestones (table)")
        st.dataframe(df, use_container_width=True, height=280)

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ JSON", data=json.dumps(ex_dump, indent=2).encode("utf-8"),
                           file_name="progression.json", mime="application/json", use_container_width=True)
        c2.download_button("⬇️ CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="progression.csv", mime="text/csv", use_container_width=True)
        c3.button("🔁 Clear", on_click=lambda: st.session_state.update({"extraction":None,"df":None,"thread_text":""}),
                  use_container_width=True)

st.markdown("""---
*Tip: include dates in emails (e.g., 'issued on 15/08/2025'). The extractor won’t invent dates or outcomes.*
""")
