import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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
        season = st.number_input("Season", min_value=1966, max_value=2025, value=2025)
        
        st.markdown("---")
        st.info("Using your trained ML models for predictions")
    
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
        st.error("❌ nfl_models.joblib not found. Please make sure it's uploaded to GitHub.")
        return
        
    if schedule is None or schedule.empty:
        st.warning(f"📋 No schedule found for {season} Week {week}")
        return
    
    # Generate predictions for each game
    with st.spinner("Generating predictions using your ML models..."):
        predictions = generate_predictions(models, schedule, historical_data, season, week)
    
    # Display predictions
    st.subheader(f"🎯 Model Predictions - {season} Week {week}")
    for _, game in predictions.iterrows():
        display_game_prediction(game)

def display_custom_matchup():
    st.header("🔮 Custom Matchup Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", get_nfl_teams())
    with col2:
        away_team = st.selectbox("Away Team", get_nfl_teams())
    
    col3, col4 = st.columns(2)
    with col3:
        spread_input = st.number_input("Vegas Spread", value=0.0, step=0.5, 
                                      help="Positive for home underdog, negative for home favorite")
    with col4:
        total_input = st.number_input("Vegas Total", value=45.0, step=0.5)
    
    if st.button("Generate Prediction", type="primary"):
        models = load_models()
        if models is not None:
            with st.spinner("Running model prediction..."):
                custom_prediction = generate_custom_prediction(
                    models, home_team, away_team, spread_input, total_input
                )
                display_game_prediction(custom_prediction)
        else:
            st.error("Please make sure nfl_models.joblib is uploaded to GitHub")

def display_game_prediction(game):
    """Display prediction for a single game"""
    st.markdown("---")
    
    # Game header
    col_header1, col_header2, col_header3 = st.columns([2, 1, 2])
    with col_header1:
        st.markdown(f"### 🏈 {game['away_team']}")
    with col_header2:
        st.markdown("### @")
    with col_header3:
        st.markdown(f"### {game['home_team']}")
    
    # Predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Moneyline")
        # Away team moneyline
        away_ml_color = "green" if game['away_ml_prob'] > 0.5 else "gray"
        st.metric(
            f"{game['away_team']}", 
            f"{game['away_ml']:+}",
            f"Win Prob: {game['away_ml_prob']:.1%}"
        )
        # Home team moneyline  
        home_ml_color = "green" if game['home_ml_prob'] > 0.5 else "gray"
        st.metric(
            f"{game['home_team']}", 
            f"{game['home_ml']:+}",
            f"Win Prob: {game['home_ml_prob']:.1%}"
        )
        
        # Recommended bet
        if game['away_ml_prob'] > game['home_ml_prob']:
            st.success(f"✅ Model Pick: {game['away_team']} ML")
        else:
            st.success(f"✅ Model Pick: {game['home_team']} ML")
    
    with col2:
        st.subheader("📊 Spread")
        st.metric("Predicted Spread", f"{game['predicted_spread']:+.1f}")
        st.metric("Model Pick", f"**{game['spread_pick']}**")
        st.write(f"**Cover Probability:** {game['cover_prob']:.1%}")
        
        # Confidence indicator
        if game['cover_prob'] > 0.6:
            st.success("High Confidence")
        elif game['cover_prob'] > 0.4:
            st.warning("Medium Confidence")
        else:
            st.error("Low Confidence")
    
    with col3:
        st.subheader("🔢 Total Points")
        st.metric("Predicted Total", f"{game['predicted_total']:.1f}")
        st.metric("Model Pick", f"**{game['total_pick']}**")
        st.write(f"**Confidence:** {game['total_confidence']:.1%}")
        
        # Over/Under indicator
        if game['total_pick'] == 'OVER':
            st.success("Trending OVER")
        else:
            st.error("Trending UNDER")

def load_models():
    """Load the trained ML models"""
    try:
        if os.path.exists("nfl_models.joblib"):
            models = joblib.load("nfl_models.joblib")
            st.sidebar.success("✅ Models loaded successfully!")
            return models
        else:
            st.sidebar.error("❌ nfl_models.joblib not found")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None

def load_historical_data():
    """Load historical game data"""
    try:
        if os.path.exists("spreadspoke_scores.csv"):
            df = pd.read_csv("spreadspoke_scores.csv")
            st.sidebar.success("✅ Historical data loaded!")
            return df
        else:
            st.sidebar.warning("📊 Historical data not found")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading historical data: {e}")
        return None

def load_schedule(season, week):
    """Load NFL schedule from JSON file"""
    try:
        if os.path.exists("nfl_2025_schedule.json"):
            with open("nfl_2025_schedule.json", 'r') as f:
                schedule_data = json.load(f)
            
            # Extract games for the requested week
            week_key = str(week)
            if week_key in schedule_data['weeks']:
                games = schedule_data['weeks'][week_key]
                schedule_df = pd.DataFrame(games)
                schedule_df['week'] = week
                schedule_df['season'] = season
                return schedule_df
            else:
                return pd.DataFrame()
        else:
            return None
    except Exception as e:
        st.error(f"Error loading schedule: {e}")
        return None

def generate_predictions(models, schedule, historical_data, season, week):
    """Generate predictions using your ML models"""
    predictions = []
    
    for _, game in schedule.iterrows():
        home_team = game['home']
        away_team = game['away']
        date = game['date']
        
        # TODO: Replace this with your actual model prediction logic
        # This is a placeholder - you'll need to integrate your actual model calls
        
        # Sample prediction (replace with your model output)
        prediction = {
            'away_team': away_team,
            'home_team': home_team,
            'date': date,
            'away_ml': +180,  # Your model should generate this
            'home_ml': -210,  # Your model should generate this
            'away_ml_prob': 0.32,  # Your model should generate this
            'home_ml_prob': 0.68,  # Your model should generate this
            'predicted_spread': -3.5,  # Your model should generate this
            'spread_pick': f"{home_team} -3.5",  # Your model should generate this
            'cover_prob': 0.62,  # Your model should generate this
            'predicted_total': 47.2,  # Your model should generate this
            'total_pick': 'OVER',  # Your model should generate this
            'total_confidence': 0.58  # Your model should generate this
        }
        predictions.append(prediction)
    
    return pd.DataFrame(predictions)

def generate_custom_prediction(models, home_team, away_team, spread, total):
    """Generate prediction for custom matchup"""
    # TODO: Replace this with your actual model prediction logic
    # This should call your model with the custom parameters
    
    return {
        'away_team': away_team,
        'home_team': home_team,
        'date': 'Custom Matchup',
        'away_ml': +160,
        'home_ml': -180,
        'away_ml_prob': 0.36,
        'home_ml_prob': 0.64,
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
