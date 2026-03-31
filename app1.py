import streamlit as st
import google.generativeai as genai
import numpy as np
import pickle
import pandas as pd
import re
# ---------- CONFIG ----------
import os
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# ✅ Updated Gemini model (IMPORTANT)
llm = genai.GenerativeModel("models/gemini-flash-latest")

# ---------- LOAD ML MODEL ----------
with open("fish_model_elite.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts['model']
columns = artifacts['columns']
mean = artifacts['mean']
std = artifacts['std']
species_list = artifacts['species_list']
use_log = artifacts['use_log']

# ---------- UI ----------
st.set_page_config(page_title="🐟 Fish Chatbot", layout="centered")
st.title("🐟 Fish Weight Chatbot")
st.write("Type like: Bream 30 10 20")

# ---------- CHAT MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- EXTRACT FUNCTION ----------
def extract_values(text):
    numbers = list(map(float, re.findall(r"\d+\.?\d*", text)))

    length = numbers[0] if len(numbers) > 0 else 0
    height = numbers[1] if len(numbers) > 1 else 0
    width = numbers[2] if len(numbers) > 2 else 0

    species = None
    for sp in species_list:
        if sp.lower() in text.lower():
            species = sp
            break

    return species, length, height, width

# ---------- CHAT INPUT ----------
user_input = st.chat_input("Enter fish details...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        species, length3, height, width = extract_values(user_input)

        if species is None or length3 == 0:
            # Ask user properly using Gemini
            response = llm.generate_content(
                f"User said: {user_input}. Ask them to provide species, length, height, and width clearly in one sentence."
            ).text

        else:
            # ---------- PREPARE INPUT ----------
            input_df = pd.DataFrame(columns=columns)
            input_df.loc[0] = 0

            for col in columns:
                col_lower = col.lower()

                if "length3" in col_lower:
                    input_df[col] = length3
                elif "height" in col_lower:
                    input_df[col] = height
                elif "width" in col_lower:
                    input_df[col] = width

            # One-hot encoding
            for col in columns:
                if species.lower() in col.lower():
                    input_df[col] = 1

            # ---------- SCALING ----------
            input_scaled = (input_df - mean) / std

            # ---------- PREDICTION ----------
            pred_log = model.predict(input_scaled)[0]

            if use_log:
                weight = np.exp(pred_log)
            else:
                weight = pred_log

            # ---------- RESPONSE ----------
            response = llm.generate_content(
                f"""
                The predicted fish weight is {weight:.2f} grams.
                Explain this result in a friendly and simple way.
                """
            ).text
    except Exception as e:
        response = f"⚠️ Error: {e}"

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)