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
    st.markdown("Machine Learning Powered NFL Predictions vs Vegas Odds")
    
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
        
        # Odds selection
        st.subheader("Odds Source")
        use_vegas_odds = st.checkbox("Compare with Vegas Odds", value=True)
        
        st.markdown("---")
        st.info("Using your trained ML models for predictions")
    
    # Main content area
    if prediction_mode == "Weekly Schedule":
        display_weekly_predictions(season, week, use_vegas_odds)
    else:
        display_custom_matchup()

def display_weekly_predictions(season, week, use_vegas_odds):
    st.header(f"📅 {season} Week {week} Predictions")
    
    # Load models and data
    models = load_models()
    schedule = load_schedule(season, week)
    historical_data = load_historical_data()
    vegas_odds = load_vegas_odds(week) if use_vegas_odds else None
    
    if models is None:
        st.error("❌ nfl_models.joblib not found. Please make sure it's uploaded to GitHub.")
        return
        
    if schedule is None or schedule.empty:
        st.warning(f"📋 No schedule found for {season} Week {week}")
        return
    
    # Generate predictions for each game
    with st.spinner("Generating predictions using your ML models..."):
        predictions = generate_predictions(models, schedule, historical_data, season, week, vegas_odds)
    
    # Display predictions
    st.subheader(f"🎯 Model Predictions vs Vegas Odds - {season} Week {week}")
    
    if use_vegas_odds and vegas_odds is None:
        st.warning("Vegas odds not found for this week. Using model predictions only.")
    
    for _, game in predictions.iterrows():
        display_game_prediction(game, use_vegas_odds)

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
                display_game_prediction(custom_prediction, False)
        else:
            st.error("Please make sure nfl_models.joblib is uploaded to GitHub")

def display_game_prediction(game, show_vegas_comparison):
    """Display prediction for a single game with Vegas comparison"""
    st.markdown("---")
    
    # Game header
    col_header1, col_header2, col_header3 = st.columns([2, 1, 2])
    with col_header1:
        st.markdown(f"### 🏈 {game['away_team']}")
    with col_header2:
        st.markdown("### @")
    with col_header3:
        st.markdown(f"### {game['home_team']}")
    
    if show_vegas_comparison and game.get('vegas_available', False):
        st.subheader("🎰 Vegas vs Model Comparison")
        
        # Moneyline comparison
        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            st.markdown("**💰 Moneyline**")
            st.write(f"Vegas: {game['vegas_away_ml']:+} / {game['vegas_home_ml']:+}")
            st.write(f"Model: {game['away_ml']:+} / {game['home_ml']:+}")
            
            # Value indicator
            if game['ml_value_bet']:
                st.success(f"🎯 VALUE BET: {game['ml_value_team']}")
            else:
                st.info("⚖️ No clear value")
                
        with col_ml2:
            st.markdown("**📊 Spread**")
            if game.get('vegas_spread'):
                st.write(f"Vegas: {game['vegas_spread']}")
                st.write(f"Model: {game['predicted_spread']:+.1f}")
                
                if game['spread_value_bet']:
                    st.success(f"🎯 VALUE: {game['spread_value_team']}")
                else:
                    st.info("⚖️ No clear value")
            else:
                st.info("No spread data")
                
        with col_ml3:
            st.markdown("**🔢 Total**")
            if game.get('vegas_total'):
                st.write(f"Vegas: {game['vegas_total']}")
                st.write(f"Model: {game['predicted_total']:.1f}")
                
                if game['total_value_bet']:
                    st.success(f"🎯 VALUE: {game['total_value_pick']}")
                else:
                    st.info("⚖️ No clear value")
            else:
                st.info("No total data")
    
    # Model predictions
    st.subheader("🤖 Model Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💰 Moneyline**")
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
        st.markdown("**📊 Spread**")
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
        st.markdown("**🔢 Total Points**")
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

def load_vegas_odds(week):
    """Load Vegas odds for the specified week"""
    try:
        odds_filename = f"week_{week}_odds.json"
        if os.path.exists(odds_filename):
            with open(odds_filename, 'r') as f:
                odds_data = json.load(f)
            st.sidebar.success(f"✅ Vegas odds loaded for Week {week}!")
            return odds_data
        else:
            st.sidebar.warning(f"📊 No Vegas odds found for Week {week}")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading Vegas odds: {e}")
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

def generate_predictions(models, schedule, historical_data, season, week, vegas_odds):
    """Generate predictions using your ML models and compare with Vegas odds"""
    predictions = []
    
    for _, game in schedule.iterrows():
        home_team = game['home']
        away_team = game['away']
        date = game['date']
        
        # Get Vegas odds for this game if available
        game_odds = get_game_odds(vegas_odds, home_team, away_team) if vegas_odds else None
        
        # TODO: Replace this with your actual model prediction logic
        # This should call your model and get real predictions
        
        # Sample prediction (replace with your model output)
        prediction = {
            'away_team': away_team,
            'home_team': home_team,
            'date': date,
            
            # Model predictions
            'away_ml': +180,
            'home_ml': -210,
            'away_ml_prob': 0.32,
            'home_ml_prob': 0.68,
            'predicted_spread': -3.5,
            'spread_pick': f"{home_team} -3.5",
            'cover_prob': 0.62,
            'predicted_total': 47.2,
            'total_pick': 'OVER',
            'total_confidence': 0.58,
            
            # Vegas comparison
            'vegas_available': game_odds is not None,
            'vegas_away_ml': game_odds['away_ml'] if game_odds else None,
            'vegas_home_ml': game_odds['home_ml'] if game_odds else None,
            'vegas_spread': game_odds['spread'] if game_odds else None,
            'vegas_total': game_odds['total'] if game_odds else None,
            
            # Value bets (you'll need to implement your value calculation logic)
            'ml_value_bet': True,  # Your model should calculate this
            'ml_value_team': away_team,  # Your model should calculate this
            'spread_value_bet': True,  # Your model should calculate this
            'spread_value_team': f"{home_team} -3.5",  # Your model should calculate this
            'total_value_bet': True,  # Your model should calculate this
            'total_value_pick': "OVER"  # Your model should calculate this
        }
        predictions.append(prediction)
    
    return pd.DataFrame(predictions)

def get_game_odds(vegas_odds, home_team, away_team):
    """Extract odds for a specific game from the Vegas odds data"""
    if not vegas_odds:
        return None
    
    game_odds = {
        'away_ml': None,
        'home_ml': None,
        'spread': None,
        'total': None
    }
    
    # Filter odds for this specific game
    game_data = [odds for odds in vegas_odds if odds['home_team'] == home_team and odds['away_team'] == away_team]
    
    if not game_data:
        return None
    
    # Extract moneyline odds
    for odds in game_data:
        if odds['market'] == 'h2h':
            if odds['label'] == away_team:
                game_odds['away_ml'] = odds['price']
            elif odds['label'] == home_team:
                game_odds['home_ml'] = odds['price']
        # You'll need to add logic for spreads and totals once you have that data
    
    return game_odds

def generate_custom_prediction(models, home_team, away_team, spread, total):
    """Generate prediction for custom matchup"""
    # TODO: Replace this with your actual model prediction logic
    
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
        'total_confidence': 0.54,
        'vegas_available': False
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
