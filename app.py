import os
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="GuideAI MVP", page_icon="🐕")
st.title("GuideAI MVP")
st.write("Prototype assistant for early service-dog behavior support.")
st.write("App started successfully.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("You need to set your OPENAI_API_KEY before using the AI explanation feature.")

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guideai_data.csv")
    return pd.read_csv(file_path)

try:
    df = load_data()
    st.write("Loaded CSV successfully.")
    st.write("CSV columns:", df.columns.tolist())
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

required_columns = [
    "behavior", "context", "frequency", "likely_issue", "risk",
    "baseline_rate", "why_it_matters", "immediate_action",
    "long_term_support", "escalate_when"
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

behavior = st.selectbox("Observed behavior", sorted(df["behavior"].dropna().unique().tolist()))
context = st.text_input("Context", placeholder="Example: crowded public space, visitors at home, puppy class")
frequency = st.selectbox("How often has this happened?", ["once", "intermittent", "repeated"])

st.subheader("Step 1: Rule-based result")

matches = df[df["behavior"] == behavior].copy()

def score_row(row, user_context, user_frequency):
    score = 0

    if isinstance(user_context, str) and isinstance(row["context"], str):
        row_words = set(str(row["context"]).lower().split())
        user_words = set(user_context.lower().split())
        score += len(row_words & user_words)

    if user_frequency == row["frequency"]:
        score += 3
    elif user_frequency == "repeated" and row["frequency"] in ["intermittent", "repeated"]:
        score += 2
    elif user_frequency == "intermittent" and row["frequency"] in ["once", "intermittent"]:
        score += 1

    return score

best = None

if not matches.empty:
    matches["score"] = matches.apply(lambda r: score_row(r, context, frequency), axis=1)
    best = matches.sort_values("score", ascending=False).iloc[0]

    st.write(f"**Likely issue:** {best['likely_issue']}")
    st.write(f"**Risk level:** {best['risk']}")
    st.write(f"**Compared to typical dogs:** {best['baseline_rate']}")
    st.write(f"**Why it matters:** {best['why_it_matters']}")
    st.write(f"**Immediate next step:** {best['immediate_action']}")
    st.write(f"**Longer-term support:** {best['long_term_support']}")
    st.write(f"**Escalate when:** {best['escalate_when']}")
else:
    st.error("No match found for that behavior in the current prototype dataset.")

st.subheader("Step 2: AI explanation")

use_ai = st.checkbox("Generate a clearer AI explanation")

if use_ai:
    if not OPENAI_AVAILABLE:
        st.error("The openai package is not installed in this environment.")
    elif not api_key:
        st.error("OPENAI_API_KEY is not set.")
    elif best is None:
        st.error("No result available to explain.")
    else:
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are helping explain a cautious, trainer-support prototype for service-dog raisers.

Observed behavior: {behavior}
User context: {context}
Frequency: {frequency}

Prototype system output:
Likely issue: {best['likely_issue']}
Risk level: {best['risk']}
Why it matters: {best['why_it_matters']}
Immediate next step: {best['immediate_action']}
Longer-term support: {best['long_term_support']}
Escalate when: {best['escalate_when']}

Write a short, supportive answer with these sections:
1. What may be going on
2. What to do now
3. What to watch for
4. When to ask a trainer for help

Do not sound overly certain.
Do not claim clinical validation.
Do not invent facts outside the provided information.
"""

        try:
            with st.spinner("Generating explanation..."):
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You are a careful, supportive assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI explanation failed: {e}")

with st.expander("Show prototype knowledge base"):
    st.dataframe(df, use_container_width=True)
