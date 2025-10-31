import streamlit as st
import pandas as pd

st.set_page_config(page_title="NFL Predictor", page_icon="🏈")
st.title("🏈 NFL Predictor Pro")
st.write("Welcome to your NFL predictions app!")

# Simple sidebar
week = st.sidebar.selectbox("Select Week", [1, 2, 3, 4, 5])
season = st.sidebar.selectbox("Select Season", [2023, 2024])

st.write(f"Showing predictions for {season} Week {week}")

# Sample data
games = [
    {"Away": "Chiefs", "Home": "49ers", "Predicted Winner": "49ers"},
    {"Away": "Eagles", "Home": "Cowboys", "Predicted Winner": "Cowboys"}
]

st.dataframe(games)
