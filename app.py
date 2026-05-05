import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GuideAI V2", page_icon="🐕")
st.title("GuideAI")
st.write("Prototype assistant for early service-dog behavior support.")

# -----------------------------
# Data-informed scoring rules
# -----------------------------
# These are based on your notes from the Erica analysis:
# - positive demeanor, plays well with dogs, nail trim behavior, reaction to stairs
#   were positively correlated with graduation
# - excitable greetings, impulsivity, fear/anxiety, and vocalizations were negatively correlated
# - poor eye contact, poor responsivity, and inappropriate play with other dogs
#   were associated with persistence/clustering of undesirable behaviors

BASE_BEHAVIOR_WEIGHTS = {
    "fear and anxiety": 4.0,
    "growling": 4.0,
    "barking": 2.5,
    "excitable greetings": 3.0,
    "impulsivity": 3.5,
    "poor eye contact": 2.5,
    "poor responsivity": 2.5,
    "inappropriate play with dogs": 2.5,
    "avoidance": 4.0,
    "mouthing": 2.0,
    "jumping on people": 2.0,
    "distraction": 2.0,
    "positive demeanor": -3.0,
    "plays well with dogs": -2.5,
    "nail trim tolerance": -2.0,
    "reaction to stairs": -2.0,
}

FREQUENCY_MULTIPLIER = {
    "once": 1.0,
    "intermittent": 1.25,
    "repeated": 1.6,
}

CLUSTER_BONUS_GROUP = {
    "poor eye contact",
    "poor responsivity",
    "inappropriate play with dogs",
}

VOCALIZATION_GROUP = {
    "barking",
    "growling",
}

PROTECTIVE_GROUP = {
    "positive demeanor",
    "plays well with dogs",
    "nail trim tolerance",
    "reaction to stairs",
}

CONTEXT_KEYWORDS = {
    "barking": ["crowded", "public", "stranger", "separation", "noise"],
    "growling": ["unfamiliar", "person", "handling", "dog", "pressure"],
    "fear and anxiety": ["new", "environment", "sound", "noise", "crowd"],
    "poor eye contact": ["training", "class", "session", "focus"],
    "poor responsivity": ["training", "home", "cue", "session"],
    "avoidance": ["novel", "person", "object", "environment"],
    "positive demeanor": ["general", "daily", "routine"],
}

# -----------------------------
# Load CSV
# -----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("guideai_data.csv")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data loaded. Check guideai_data.csv.")
    st.stop()

required_cols = [
    "behavior",
    "context",
    "frequency",
    "likely_issue",
    "risk",
    "why_it_matters",
    "immediate_action",
    "long_term_support",
    "escalate_when",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required CSV columns: {missing}")
    st.stop()

# -----------------------------
# Helper functions
# -----------------------------
def context_bonus(behavior: str, user_context: str) -> float:
    if not user_context:
        return 0.0
    user_words = set(user_context.lower().split())
    expected = set(CONTEXT_KEYWORDS.get(behavior, []))
    overlap = len(user_words & expected)
    return min(overlap * 0.5, 1.5)

def compute_risk_score(behavior: str, frequency: str, context: str) -> float:
    base = BASE_BEHAVIOR_WEIGHTS.get(behavior, 2.0)
    mult = FREQUENCY_MULTIPLIER.get(frequency, 1.0)
    score = base * mult

    # persistence / clustering signal
    if behavior in CLUSTER_BONUS_GROUP and frequency == "repeated":
        score += 1.0

    # vocalization concerns
    if behavior in VOCALIZATION_GROUP and frequency == "repeated":
        score += 0.75

    # protective behaviors reduce concern
    if behavior in PROTECTIVE_GROUP and frequency == "repeated":
        score -= 0.75

    # context relevance
    score += context_bonus(behavior, context)

    return score

def score_to_label(score: float) -> str:
    if score >= 5.5:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"

def confidence_text(behavior: str, frequency: str) -> str:
    if frequency == "repeated":
        return "higher confidence because the behavior is reported as repeated"
    if behavior in PROTECTIVE_GROUP:
        return "moderate confidence because this behavior is generally protective in the historical findings"
    return "moderate confidence based on behavior category alone"

def pick_best_row(matches: pd.DataFrame, user_context: str, user_frequency: str) -> pd.Series:
    def row_score(row):
        score = 0
        if isinstance(user_context, str) and isinstance(row.get("context"), str):
            row_words = set(str(row["context"]).lower().split())
            user_words = set(user_context.lower().split())
            score += len(row_words & user_words)
        if user_frequency == row.get("frequency"):
            score += 3
        elif user_frequency == "repeated" and row.get("frequency") in ["intermittent", "repeated"]:
            score += 2
        elif user_frequency == "intermittent" and row.get("frequency") in ["once", "intermittent"]:
            score += 1
        return score

    matches = matches.copy()
    matches["match_score"] = matches.apply(row_score, axis=1)
    return matches.sort_values("match_score", ascending=False).iloc[0]

# -----------------------------
# UI
# -----------------------------
behavior = st.selectbox("Observed behavior", sorted(df["behavior"].dropna().unique()))
context = st.text_input("Context", placeholder="Example: crowded public space, unfamiliar person, training class")
frequency = st.selectbox("How often has this happened?", ["once", "intermittent", "repeated"])

st.subheader("Step 1: Data-informed result")

matches = df[df["behavior"] == behavior].copy()
if matches.empty:
    st.error("No rows found for that behavior in guideai_data.csv")
    st.stop()

best = pick_best_row(matches, context, frequency)

risk_score = compute_risk_score(behavior, frequency, context)
risk_label = score_to_label(risk_score)

st.write(f"**Likely issue:** {best.get('likely_issue', 'N/A')}")
st.write(f"**Risk level:** {risk_label}")
st.caption(
    "This risk level reflects historical associations in puppy-report analyses, not causal or clinical determination."
)

st.write(f"**Why it matters:** {best.get('why_it_matters', 'N/A')}")
st.write(f"**Immediate next step:** {best.get('immediate_action', 'N/A')}")
st.write(f"**Longer-term support:** {best.get('long_term_support', 'N/A')}")
st.write(f"**Escalate when:** {best.get('escalate_when', 'N/A')}")
st.write(f"**Confidence note:** {confidence_text(behavior, frequency)}")
st.write(f"**Internal risk score:** {risk_score:.2f}")

with st.expander("Why this result was selected"):
    st.write(f"- Selected behavior: `{behavior}`")
    st.write(f"- Selected frequency: `{frequency}`")
    st.write(f"- User context: `{context or 'none provided'}`")
    st.write("- Risk score combines base behavior weight, persistence, behavior-group effects, and context relevance.")
    st.write("- Matching row is chosen from the CSV using behavior + context overlap + frequency fit.")

with st.expander("Debug: show matched rows"):
    st.dataframe(matches, use_container_width=True)

with st.expander("Debug: show full data"):
    st.dataframe(df, use_container_width=True)
