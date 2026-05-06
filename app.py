import os
import pandas as pd
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="GuideAI V2", page_icon="🐕")

st.title("GuideAI")
st.write("Prototype assistant for early service-dog behavior support.")
st.caption(
    "Prototype only — not official training advice. Designed to help raisers translate behavior patterns into structured next steps."
)

LOG_FILE = "behavior_log.csv"
FEEDBACK_FILE = "user_feedback.csv"

# -----------------------------
# Data-informed scoring rules
# -----------------------------

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
    "barking": ["crowded", "public", "stranger", "separation", "noise", "class", "training"],
    "growling": ["unfamiliar", "person", "handling", "dog", "pressure"],
    "fear and anxiety": ["new", "environment", "sound", "noise", "crowd"],
    "poor eye contact": ["training", "class", "session", "focus"],
    "poor responsivity": ["training", "home", "cue", "session"],
    "avoidance": ["novel", "new", "person", "object", "environment", "public", "crowded"],
    "positive demeanor": ["general", "daily", "routine", "home"],
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

    if behavior in CLUSTER_BONUS_GROUP and frequency == "repeated":
        score += 1.0

    if behavior in VOCALIZATION_GROUP and frequency == "repeated":
        score += 0.75

    if behavior in PROTECTIVE_GROUP and frequency == "repeated":
        score -= 0.75

    score += context_bonus(behavior, context)

    return score


def score_to_label(score: float) -> str:
    if score >= 5.5:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"


def confidence_text(behavior: str, frequency: str, context: str) -> str:
    if frequency == "repeated" and context.strip():
        return "higher confidence because the behavior is repeated and context was provided"
    if frequency == "repeated":
        return "moderate-to-higher confidence because the behavior is reported as repeated"
    if behavior in PROTECTIVE_GROUP:
        return "moderate confidence because this behavior is generally protective in historical findings"
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
# UI inputs
# -----------------------------

behavior = st.selectbox(
    "Observed behavior",
    sorted(df["behavior"].dropna().unique()),
)

context = st.text_input(
    "Context",
    placeholder="Example: crowded public space, unfamiliar person, training class",
)

frequency = st.selectbox(
    "How often has this happened?",
    ["once", "intermittent", "repeated"],
)

tester_id = st.text_input(
    "Tester name/initials or dog name",
    placeholder="Example: Izzy / Puppy A / Dog initials"
)

if not tester_id:
    st.info("Enter your name or dog name to enable tracking and trends.")

# -----------------------------
# Results
# -----------------------------

st.subheader("Data-informed result")

matches = df[df["behavior"] == behavior].copy()

if matches.empty:
    st.error("No rows found for that behavior in guideai_data.csv.")
    st.stop()

best = pick_best_row(matches, context, frequency)

risk_score = compute_risk_score(behavior, frequency, context)
risk_label = score_to_label(risk_score)

st.write(f"**Likely issue:** {best.get('likely_issue', 'N/A')}")
st.write(f"**Concern level:** {risk_label}")

st.caption(
    "This concern level reflects structured behavior rules informed by prior puppy-report analysis and puppy-raiser experience. "
    "It is not an official training, causal, or clinical determination."
)

st.write(f"**Why it matters:** {best.get('why_it_matters', 'N/A')}")
st.write(f"**Immediate next step:** {best.get('immediate_action', 'N/A')}")
st.write(f"**Longer-term support:** {best.get('long_term_support', 'N/A')}")
#st.write(f"**Escalate when:** {best.get('escalate_when', 'N/A')}")
st.write("**When to get extra help:**")
st.write(best.get("escalate_when", "N/A"))
#st.write(f"**Confidence note:** {confidence_text(behavior, frequency, context)}")
#st.write(f"**Internal concern score:** {risk_score:.2f}")
st.caption("Based on behavior type, how often it occurs, and the context provided.")

# -----------------------------
# Step-by-step guidance (NEW)
# -----------------------------

# -----------------------------
# Step-by-step guidance (ALWAYS SHOW)
# -----------------------------

st.write("**Step-by-step guidance:**")

if behavior == "barking":
    st.write("1. Increase distance from the trigger (move further away)")
    st.write("2. Wait for a moment of calm (no barking)")
    st.write("3. Immediately reward calm behavior")
    st.write("4. Keep sessions short and controlled")

elif behavior == "fear and anxiety":
    st.write("1. Reduce exposure to the stressful environment")
    st.write("2. Allow the dog to observe from a safe distance")
    st.write("3. Reward calm, non-reactive behavior")
    st.write("4. Gradually increase exposure over time")

else:
    st.write("1. Simplify the situation")
    st.write("2. Reinforce the desired behavior")
    st.write("3. Gradually reintroduce difficulty")

st.write("**Helpful resource to add later:**")
if behavior == "barking":
    st.write("A short video showing how to increase distance, reward calm check-ins, and re-enter the environment gradually would be useful here.")
elif behavior == "fear and anxiety":
    st.write("A short video showing gradual exposure/desensitization would be useful here.")
else:
    st.write("A short demo video showing the recommended training steps would be useful here.")

# -----------------------------
# AI Coach
# -----------------------------

st.subheader("AI Coach")

st.caption(
    "Optional: generate more detailed guidance based on the structured result above. "
    "AI-generated guidance may be incomplete or incorrect and should not replace trainer guidance."
)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.info("AI Coach is not enabled yet because GEMINI_API_KEY is not set.")
else:
    if st.button("Generate more detailed guidance"):
        prompt = f"""
You are helping a service-dog puppy raiser interpret a behavior concern.

Important constraints:
- Do not diagnose the dog.
- Do not claim certainty.
- Do not replace a trainer.
- Stay grounded in the structured result below.
- Give concrete, practical, step-by-step guidance.
- Use supportive language.
- If the situation seems serious, recommend contacting a trainer.

User inputs:
Behavior: {behavior}
Context: {context}
Frequency: {frequency}

Structured result:
Likely issue: {best.get('likely_issue', 'N/A')}
Concern level: {risk_label}
Why it matters: {best.get('why_it_matters', 'N/A')}
Immediate next step: {best.get('immediate_action', 'N/A')}
Longer-term support: {best.get('long_term_support', 'N/A')}
When to get help: {best.get('escalate_when', 'N/A')}

Write the answer in this exact format:

### What may be going on
2-3 sentences.

### What to try next time
Give 4 numbered steps.

### What not to do
Give 2 bullets.

### What to watch
Give 3 bullets.

### Helpful resource to add later
Describe what kind of short demo video or training resource would help for this scenario. Do not link to any specific organization.
"""

        try:
            with st.spinner("Generating detailed guidance..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)

            st.markdown(response.text)

        except Exception as e:
            st.error(f"AI Coach failed: {e}")
            
# -----------------------------
# Log behavior
# -----------------------------

st.subheader("Track this behavior over time")

if st.button("Log this observation") and tester_id:
    log_entry = {
         "timestamp": pd.Timestamp.now(),
        "tester_id": tester_id,
        "behavior": behavior,
        "context": context,
        "frequency": frequency,
        "likely_issue": best.get("likely_issue", "N/A"),
        "concern_level": risk_label,
        "concern_score": risk_score,
    }

    try:
        log_df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        log_df = pd.DataFrame()

    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)

    st.success("Observation saved. You are now tracking behavior over time.")

    st.caption("You can log multiple observations to see trends over time.")

if st.checkbox("Show my behavior trends over time"):

    # 🚨 require user identity FIRST
    if not tester_id or not tester_id.strip():
        st.warning("Enter your name or dog name to view your behavior trends.")
        st.stop()

    try:
        log_df = pd.read_csv(LOG_FILE)

        # ✅ filter to THIS user only
        log_df = log_df[log_df["tester_id"] == tester_id]

        if log_df.empty:
            st.info("No logged observations yet for this tester/dog.")
        else:
            # ensure proper time ordering
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
            log_df = log_df.sort_values("timestamp")

            # 📈 trend chart
            st.line_chart(
                log_df.set_index("timestamp")["concern_score"]
            )

            st.caption(f"{len(log_df)} observations logged")

            # 🧠 simple trend insight (nice PM touch)
            if len(log_df) >= 3:
                recent = log_df["concern_score"].tail(3).tolist()

                if recent[-1] > recent[0]:
                    st.warning("Recent observations suggest concern may be trending upward.")
                elif recent[-1] < recent[0]:
                    st.success("Recent observations suggest concern may be trending downward.")
                else:
                    st.info("Recent concern level appears stable.")

    except FileNotFoundError:
        st.info("No logged observations yet.")






# -----------------------------
# Feedback
# -----------------------------

st.subheader("Feedback for user testing")

useful = st.selectbox(
    "Was this output useful?",
    ["select one", "yes", "somewhat", "no"],
)

trust = st.selectbox(
    "Would you trust this as a raiser support tool?",
    ["select one", "yes", "maybe", "no"],
)

feedback = st.text_area(
    "What felt missing, unclear, or too generic?",
    placeholder="Example: I want more specific steps for barking during puppy class...",
)

if st.button("Save feedback") and tester_id:
    if useful == "select one" or trust == "select one":
        st.warning("Please answer the usefulness and trust questions before saving feedback.")
    else:
        feedback_entry = {
            "timestamp": pd.Timestamp.now(),
            "tester_id": tester_id,
            "behavior": behavior,
            "context": context,
            "frequency": frequency,
            "likely_issue": best.get("likely_issue", "N/A"),
            "concern_level": risk_label,
            "concern_score": risk_score,
            "feedback_useful": useful,
            "feedback_trust": trust,
            "feedback_text": feedback,
        }

        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
        except FileNotFoundError:
            feedback_df = pd.DataFrame()

        feedback_df = pd.concat([feedback_df, pd.DataFrame([feedback_entry])], ignore_index=True)
        feedback_df.to_csv(FEEDBACK_FILE, index=False)

        st.success("Feedback saved. Use this to guide the next iteration.")

        st.caption("Thank you — this feedback helps improve the system.")


st.subheader("Admin only")

admin_password = st.text_input("Enter admin password", type="password")

if admin_password == "izzy123":
    st.success("Access granted")

    st.write("### Saved feedback")
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(feedback_df, use_container_width=True)
    except FileNotFoundError:
        st.info("No feedback saved yet.")

    st.write("### Behavior logs")
    try:
        log_df = pd.read_csv(LOG_FILE)
        st.dataframe(log_df, use_container_width=True)
    except FileNotFoundError:
        st.info("No behavior logs saved yet.")

elif admin_password:
    st.error("Incorrect password")
