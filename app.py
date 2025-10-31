import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="NFL Predictor Pro",
    page_icon="🏈",
    layout="wide"
)

def main():
    st.title("🏈 NFL Predictor Pro")
    st.markdown("Machine Learning Powered NFL Predictions")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        st.subheader("Prediction Mode")
        prediction_mode = st.radio(
            "Select Prediction Type",
            ["Weekly Schedule", "Custom Matchup"]
        )
        
        # Week selection
        current_week = get_current_nfl_week()
        week = st.selectbox("Select Week", list(range(1, 19)), index=current_week-1)
        season = st.number_input("Season", min_value=1966, max_value=2025, value=2024)
        
        st.markdown("---")
        st.info("Upload your data files to get started")
    
    # Main content area
    if prediction_mode == "Weekly Schedule":
        display_weekly_predictions(season, week)
    else:
        display_custom_matchup()

def display_weekly_predictions(season, week):
    st.header(f"📅 {season} Week {week} Predictions")
    
    # Load models and data
    models = load_models()
    schedule = load_schedule(season, week)
    historical_data = load_historical_data()
    
    if models is None:
        st.error("❌ Please upload nfl_models.joblib file")
        return
        
    if schedule is None:
        st.warning("📋 Please upload NFL schedule file")
        return
    
    # Generate predictions for each game
    with st.spinner("Generating predictions..."):
        predictions = generate_predictions(models, schedule, historical_data)
    
    # Display predictions
    for _, game in predictions.iterrows():
        display_game_prediction(game)

def display_custom_matchup():
    st.header("🔮 Custom Matchup Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", get_nfl_teams())
        home_team = st.selectbox("Away Team", get_nfl_teams())
    
    with col2:
        spread_input = st.number_input("Vegas Spread", value=0.0, step=0.5)
        total_input = st.number_input("Vegas Total", value=45.0, step=0.5)
    
    if st.button("Generate Prediction"):
        models = load_models()
        if models is not None:
            # Generate custom prediction
            custom_prediction = generate_custom_prediction(
                models, home_team, away_team, spread_input, total_input
            )
            display_game_prediction(custom_prediction)
        else:
            st.error("Please upload nfl_models.joblib first")

def display_game_prediction(game):
    """Display prediction for a single game"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Moneyline")
        st.metric(f"{game['away_team']}", f"{game['away_ml']:+}")
        st.metric(f"{game['home_team']}", f"{game['home_ml']:+}")
        st.write(f"**Win Probability:** {game['home_win_prob']:.1%}")
    
    with col2:
        st.subheader("📊 Spread")
        st.metric("Predicted Spread", f"{game['predicted_spread']:+.1f}")
        st.metric("Model Pick", game['spread_pick'])
        st.write(f"**Cover Probability:** {game['cover_prob']:.1%}")
    
    with col3:
        st.subheader("🔢 Total")
        st.metric("Predicted Total", f"{game['predicted_total']:.1f}")
        st.metric("Model Pick", game['total_pick'])
        st.write(f"**Confidence:** {game['total_confidence']:.1%}")

def load_models():
    """Load the trained ML models"""
    try:
        if os.path.exists("nfl_models.joblib"):
            models = joblib.load("nfl_models.joblib")
            st.success("✅ Models loaded successfully!")
            return models
        else:
            return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def load_historical_data():
    """Load historical game data"""
    try:
        if os.path.exists("spreadspokescores.csv"):
            return pd.read_csv("spreadspokescores.csv")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

def load_schedule(season, week):
    """Load NFL schedule"""
    try:
        # You'll need to create/upload a schedule file
        if os.path.exists(f"nfl_schedule_{season}.csv"):
            schedule = pd.read_csv(f"nfl_schedule_{season}.csv")
            return schedule[schedule['week'] == week]
        else:
            return None
    except:
        return None

def generate_predictions(models, schedule, historical_data):
    """Generate predictions using your ML models"""
    # This is where your model prediction logic goes
    # For now, return sample data
    sample_games = [
        {
            'away_team': 'Kansas City',
            'home_team': 'San Francisco', 
            'away_ml': +180,
            'home_ml': -210,
            'home_win_prob': 0.68,
            'predicted_spread': -3.5,
            'spread_pick': 'SF -3.5',
            'cover_prob': 0.62,
            'predicted_total': 47.2,
            'total_pick': 'OVER',
            'total_confidence': 0.58
        },
        {
            'away_team': 'Philadelphia',
            'home_team': 'Dallas',
            'away_ml': +150, 
            'home_ml': -170,
            'home_win_prob': 0.63,
            'predicted_spread': -2.5,
            'spread_pick': 'DAL -2.5',
            'cover_prob': 0.55,
            'predicted_total': 51.5,
            'total_pick': 'UNDER',
            'total_confidence': 0.52
        }
    ]
    return pd.DataFrame(sample_games)

def generate_custom_prediction(models, home_team, away_team, spread, total):
    """Generate prediction for custom matchup"""
    # Your custom prediction logic here
    return {
        'away_team': away_team,
        'home_team': home_team,
        'away_ml': +160,
        'home_ml': -180,
        'home_win_prob': 0.64,
        'predicted_spread': -3.0,
        'spread_pick': f'{home_team} -3.0',
        'cover_prob': 0.58,
        'predicted_total': total + 1.2,
        'total_pick': 'OVER',
        'total_confidence': 0.54
    }

def get_nfl_teams():
    """Return list of NFL teams"""
    return [
        "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
        "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
        "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
        "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
        "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
        "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
        "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
    ]

def get_current_nfl_week():
    """Calculate current NFL week (simplified)"""
    return min(18, max(1, (datetime.now().month - 8) * 4 + 1))

if __name__ == "__main__":
    main()
