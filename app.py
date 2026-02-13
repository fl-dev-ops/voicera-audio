#!/usr/bin/env python3
"""
Streamlit app to browse and download Voicera call recordings and transcripts.

Usage:
    streamlit run app.py
"""

import os
import re
import logging
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

# Suppress noisy thread warnings from Streamlit cache
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)

# Load environment variables
load_dotenv()

MONGODB_URL = os.getenv(
    "MONGODB_URL", "mongodb://admin:admin123@localhost:27017/voicera?authSource=admin"
)
MINIO_PUBLIC_URL = os.getenv(
    "MINIO_PUBLIC_URL", "https://s3.voicera.foreverlearning.in"
)

# Page config
st.set_page_config(page_title="Voicera Audio Browser", page_icon="🎙️", layout="wide")


# ── Database helpers ─────────────────────────────────────────────────────────


@st.cache_resource
def get_mongo_client():
    """Create a cached MongoDB client."""
    return MongoClient(MONGODB_URL)


def minio_to_public_url(minio_url: str) -> str:
    """Convert minio://bucket/object to a public HTTPS URL."""
    if not minio_url or not minio_url.startswith("minio://"):
        return ""
    path = minio_url.replace("minio://", "")
    return f"{MINIO_PUBLIC_URL}/{path}"


# ── Transcript parsing ───────────────────────────────────────────────────────


def parse_transcript(transcript_content: str) -> list[dict]:
    """
    Parse raw transcript text into structured messages.

    Handles formats:
        [timestamp] user: message
        [timestamp] assistant: message
        user: message
        agent: message
    """
    if not transcript_content:
        return []

    messages: list[dict] = []
    lines = transcript_content.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match: [timestamp] role: message
        match = re.match(
            r"^\[([^\]]+)\]\s*(user|assistant|agent|human|bot):\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:
            timestamp, role, message = match.groups()
            normalized_role = "user" if role.lower() in ["user", "human"] else "agent"
            messages.append(
                {
                    "role": normalized_role,
                    "content": message.strip(),
                    "timestamp": timestamp.strip(),
                }
            )
        elif line.lower().startswith(("user:", "human:")):
            content = re.sub(
                r"^(user|human):\s*", "", line, flags=re.IGNORECASE
            ).strip()
            messages.append({"role": "user", "content": content})
        elif line.lower().startswith(("agent:", "assistant:", "bot:")):
            content = re.sub(
                r"^(agent|assistant|bot):\s*", "", line, flags=re.IGNORECASE
            ).strip()
            messages.append({"role": "agent", "content": content})
        else:
            if messages:
                last_role = messages[-1]["role"]
                next_role = "user" if last_role == "agent" else "agent"
            else:
                next_role = "agent"
            messages.append({"role": next_role, "content": line})

    return messages


def format_duration(seconds: float | None) -> str:
    """Format seconds into human-readable duration."""
    if seconds is None or pd.isna(seconds) or seconds <= 0:
        return "—"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    secs = total % 60
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:02d}s"


def format_datetime(dt_str: str | None) -> str:
    """Format ISO datetime string for display."""
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %I:%M %p")
    except (ValueError, AttributeError):
        return str(dt_str)


# ── Data loading ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_meetings():
    """Load all meetings from MongoDB CallLogs collection."""
    try:
        client = get_mongo_client()
        db = client.get_default_database()
        collection = db["CallLogs"]

        meetings = list(collection.find({}).sort("created_at", -1))

        if not meetings:
            return pd.DataFrame(), None

        # Normalize documents
        rows = []
        for doc in meetings:
            rows.append(
                {
                    "meeting_id": doc.get("meeting_id", ""),
                    "agent_type": doc.get("agent_type", ""),
                    "org_id": doc.get("org_id", ""),
                    "from_number": doc.get("from_number", ""),
                    "to_number": doc.get("to_number", ""),
                    "created_at": doc.get("created_at", ""),
                    "start_time_utc": doc.get("start_time_utc", ""),
                    "end_time_utc": doc.get("end_time_utc", ""),
                    "duration": doc.get("duration"),
                    "call_busy": doc.get("call_busy", False),
                    "inbound": doc.get("inbound"),
                    "recording_url": doc.get("recording_url", ""),
                    "transcript_url": doc.get("transcript_url", ""),
                    "transcript_content": doc.get("transcript_content", ""),
                }
            )

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return None, str(e)


# ── UI ───────────────────────────────────────────────────────────────────────

# Title
st.title("🎙️ Voicera Audio Browser")
st.markdown("Browse and download call recordings and transcripts.")

# Refresh button in sidebar
st.sidebar.header("🔄 Data")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Load data
df, error = load_meetings()

if error:
    st.error(f"Database connection error: {error}")
    st.info("Make sure the MONGODB_URL environment variable is set correctly.")
    st.stop()

if df is None or df.empty:
    st.warning("No call recordings found in the database.")
    st.stop()

# ── Sidebar filters ─────────────────────────────────────────────────────────

st.sidebar.header("🔍 Filters")

# Agent type filter
agent_types = ["All"] + sorted([a for a in df["agent_type"].unique().tolist() if a])
selected_agent = st.sidebar.selectbox("Agent Type", agent_types)

filtered_df = df.copy()
if selected_agent != "All":
    filtered_df = filtered_df[filtered_df["agent_type"] == selected_agent]

# Direction filter
directions = ["All", "Inbound", "Outbound"]
selected_direction = st.sidebar.selectbox("Call Direction", directions)
if selected_direction == "Inbound":
    filtered_df = filtered_df[filtered_df["inbound"] == True]  # noqa: E712
elif selected_direction == "Outbound":
    filtered_df = filtered_df[filtered_df["inbound"] == False]  # noqa: E712

# Phone number search
phone_search = st.sidebar.text_input("Search Phone Number", "")
if phone_search:
    mask = filtered_df["from_number"].str.contains(
        phone_search, case=False, na=False
    ) | filtered_df["to_number"].str.contains(phone_search, case=False, na=False)
    filtered_df = filtered_df[mask]

# Show only calls with recordings
only_recordings = st.sidebar.checkbox("Only with recordings", value=False)
if only_recordings:
    filtered_df = filtered_df[
        filtered_df["recording_url"].str.startswith("minio://", na=False)
    ]

# Stats
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Records:** {len(df)}")
st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)}")

# ── Stats cards ──────────────────────────────────────────────────────────────

st.markdown("---")

total_calls = len(filtered_df)
total_duration_seconds = filtered_df["duration"].fillna(0).sum()
has_recording = filtered_df["recording_url"].str.startswith("minio://", na=False).sum()
has_transcript = (filtered_df["transcript_content"].fillna("").str.len() > 0).sum()

total_hours = int(total_duration_seconds // 3600)
total_minutes = int((total_duration_seconds % 3600) // 60)
if total_hours > 0:
    duration_display = f"{total_hours}h {total_minutes}m"
else:
    duration_display = f"{total_minutes}m"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Calls", f"{total_calls:,}")
with col2:
    st.metric("Total Talk Time", duration_display)
with col3:
    st.metric("With Recording", f"{has_recording:,}")
with col4:
    st.metric("With Transcript", f"{has_transcript:,}")

st.markdown("---")

# ── Meetings table ───────────────────────────────────────────────────────────

if filtered_df.empty:
    st.warning("No recordings match the selected filters.")
else:
    # Build display dataframe
    display_df = filtered_df.copy()
    display_df["direction"] = display_df["inbound"].map(
        {True: "📞 Inbound", False: "📱 Outbound", None: "—"}
    )
    display_df["duration_fmt"] = display_df["duration"].apply(format_duration)
    display_df["date"] = display_df["created_at"].apply(format_datetime)
    display_df["audio_url"] = filtered_df["recording_url"].apply(minio_to_public_url)
    display_df["transcript"] = filtered_df["transcript_content"].fillna(
        "No transcript available"
    )

    # Table columns
    table_cols = [
        "date",
        "agent_type",
        "direction",
        "from_number",
        "to_number",
        "duration_fmt",
        "audio_url",
        "transcript",
    ]
    table_names = {
        "date": "Date",
        "agent_type": "Agent",
        "direction": "Direction",
        "from_number": "From",
        "to_number": "To",
        "duration_fmt": "Duration",
        "audio_url": "Audio URL",
        "transcript": "Transcript",
    }

    column_config = {
        "Audio URL": st.column_config.LinkColumn(
            "Audio URL",
            display_text="🔗 Download",
        ),
        "Transcript": st.column_config.TextColumn(
            "Transcript",
            width="large",
            help="Full conversation transcript",
        ),
    }

    st.dataframe(
        display_df[table_cols].rename(columns=table_names),
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )

    # ── Bulk download ────────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("📥 Bulk Download")

    col1, col2 = st.columns(2)

    with col1:
        export_cols = [
            "meeting_id",
            "agent_type",
            "from_number",
            "to_number",
            "created_at",
            "duration",
            "inbound",
            "transcript_content",
        ]
        export_df = filtered_df[export_cols].copy()
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name="voicera_calls.csv",
            mime="text/csv",
        )

    with col2:
        # Export public audio URLs
        audio_urls = display_df["audio_url"][display_df["audio_url"] != ""].tolist()
        if audio_urls:
            urls_text = "\n".join(audio_urls)
            st.download_button(
                label="📄 Download Audio URL List",
                data=urls_text,
                file_name="audio_urls.txt",
                mime="text/plain",
            )
        else:
            st.info("No recordings to export.")

# Footer
st.markdown("---")
st.caption(
    "Data is cached for 1 hour. Click 'Refresh Data' in the sidebar to fetch latest records."
)
