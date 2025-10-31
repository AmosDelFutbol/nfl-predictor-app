import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="NFL Predictor Pro",
    page_icon="🏈",
    layout="wide"
)

# Import your existing functions (I'll include the key ones)
def load_nfl_data():
    """Load the NFL betting data"""
    try:
        df = pd.read_csv("spreadspoke_scores.csv")
        st.success("✅ Historical data loaded!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def load_nfl_schedule():
    """Load the 2025 NFL schedule from JSON file"""
    try:
        with open("nfl_2025_schedule.json", 'r') as f:
            schedule = json.load(f)
        st.success("✅ Schedule loaded!")
        return schedule
    except Exception as e:
        st.error(f"❌ Error loading schedule: {e}")
        return None

def load_vegas_odds(week):
    """Load Vegas odds for specific week"""
    try:
        odds_file = f"week_{week}_odds.json"
        with open(odds_file, 'r') as f:
            odds = json.load(f)
        st.success(f"✅ Vegas odds loaded for Week {week}!")
        return odds
    except Exception as e:
        st.warning(f"⚠️ No Vegas odds found for Week {week}")
        return None

def prepare_data(df):
    """Prepare data (from your existing code)"""
    # Your data preparation logic here
    df_recent = df[df['schedule_season'].between(2020, 2025)].copy()
    
    # Add your data cleaning and feature engineering
    df_recent['home_win'] = (df_recent['score_home'] > df_recent['score_away']).astype(int)
    df_recent['actual_spread'] = df_recent['score_home'] - df_recent['score_away']
    df_recent['total_points'] = df_recent['score_home'] + df_recent['score_away']
    
    return df_recent

def calculate_team_stats(df):
    """Calculate team statistics (simplified version)"""
    teams = pd.unique(np.concatenate([df['team_home'].unique(), df['team_away'].unique()]))
    team_stats = {}
    
    for team in teams:
        home_games = df[df['team_home'] == team]
        away_games = df[df['team_away'] == team]
        
        total_games = len(home_games) + len(away_games)
        total_wins = home_games['home_win'].sum() + (len(away_games) - away_games['home_win'].sum())
        
        team_stats[team] = {
            'total_games': total_games,
            'total_wins': total_wins,
            'win_pct': total_wins / total_games if total_games > 0 else 0.5
        }
    
    return team_stats

def get_team_recent_stats(team, df, games_back=5):
    """Get recent performance for a team"""
    team_games = pd.concat([
        df[df['team_home'] == team],
        df[df['team_away'] == team]
    ]).sort_values('schedule_date').tail(games_back)
    
    if len(team_games) == 0:
        return {'win_pct': 0.5, 'ppg': 21, 'ppg_against': 21}
    
    wins = 0
    points_for = 0
    points_against = 0
    
    for _, game in team_games.iterrows():
        if game['team_home'] == team:
            wins += game['home_win']
            points_for += game['score_home']
            points_against += game['score_away']
        else:
            wins += (1 - game['home_win'])  # Away win
            points_for += game['score_away']
            points_against += game['score_home']
    
    return {
        'win_pct': wins / len(team_games),
        'ppg': points_for / len(team_games),
        'ppg_against': points_against / len(team_games)
    }

def predict_winner(home_team, away_team, team_stats, df):
    """Predict game winner based on team stats and recent performance"""
    home_recent = get_team_recent_stats(home_team, df)
    away_recent = get_team_recent_stats(away_team, df)
    
    home_win_pct = team_stats[home_team]['win_pct']
    away_win_pct = team_stats[away_team]['win_pct']
    
    # Simple weighted prediction (60% recent form, 40% overall)
    home_strength = (home_recent['win_pct'] * 0.6) + (home_win_pct * 0.4)
    away_strength = (away_recent['win_pct'] * 0.6) + (away_win_pct * 0.4)
    
    # Home field advantage (~3 points)
    home_advantage = 0.05
    
    home_prob = home_strength / (home_strength + away_strength) + home_advantage
    away_prob = 1 - home_prob
    
    predicted_winner = home_team if home_prob > away_prob else away_team
    confidence = max(home_prob, away_prob)
    
    return predicted_winner, home_prob, away_prob, confidence

def predict_score(home_team, away_team, df):
    """Predict final score based on recent performance"""
    home_recent = get_team_recent_stats(home_team, df)
    away_recent = get_team_recent_stats(away_team, df)
    
    # Project scores: (team offense + opponent defense) / 2 + home field advantage
    home_score = (home_recent['ppg'] + away_recent['ppg_against']) / 2 + 2.5  # Home field advantage
    away_score = (away_recent['ppg'] + home_recent['ppg_against']) / 2 - 1.0  # Away disadvantage
    
    return round(home_score), round(away_score)

def get_vegas_game_odds(vegas_odds, home_team, away_team):
    """Extract Vegas odds for a specific game"""
    if not vegas_odds:
        return None
    
    game_odds = {
        'moneyline': {},
        'spread': None,
        'total': None
    }
    
    # Find moneyline odds
    for odds in vegas_odds:
        if odds['home_team'] == home_team and odds['away_team'] == away_team:
            if odds['market'] == 'h2h':
                if odds['label'] == home_team:
                    game_odds['moneyline']['home'] = odds['price']
                elif odds['label'] == away_team:
                    game_odds['moneyline']['away'] = odds['price']
    
    return game_odds

def analyze_spread_prediction(predicted_winner, predicted_home_score, predicted_away_score, vegas_spread):
    """Analyze spread prediction vs Vegas"""
    if not vegas_spread:
        return "No spread data", 0.5
    
    actual_spread = predicted_home_score - predicted_away_score
    
    # Determine who covers based on Vegas spread
    # Negative spread means home favorite, positive means away favorite
    if vegas_spread < 0:  # Home favorite
        home_needs_to_win_by = abs(vegas_spread)
        if predicted_winner == 'home' and actual_spread > home_needs_to_win_by:
            return "Home covers", 0.7
        else:
            return "Away covers", 0.3
    else:  # Away favorite
        away_needs_to_win_by = vegas_spread
        if predicted_winner == 'away' and abs(actual_spread) > away_needs_to_win_by:
            return "Away covers", 0.7
        else:
            return "Home covers", 0.3

def analyze_total_prediction(predicted_total, vegas_total):
    """Analyze over/under prediction vs Vegas"""
    if not vegas_total:
        return "No total data", 0.5
    
    if predicted_total > vegas_total:
        return "OVER", (predicted_total - vegas_total) / 10  # Confidence based on difference
    elif predicted_total < vegas_total:
        return "UNDER", (vegas_total - predicted_total) / 10
    else:
        return "PUSH", 0.5

def main():
    st.title("🏈 NFL Predictor Pro")
    st.markdown("### Machine Learning Powered Betting Predictions")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_nfl_data()
        schedule = load_nfl_schedule()
        
        if df is None:
            st.error("❌ Could not load historical data. Please make sure spreadspoke_scores.csv is in your GitHub repository.")
            return
            
        if schedule is None:
            st.warning("⚠️ Could not load schedule. Some features may not work.")
        
        df_prepared = prepare_data(df)
        team_stats = calculate_team_stats(df_prepared)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Prediction Settings")
        
        prediction_mode = st.radio(
            "Select Mode",
            ["Weekly Predictions", "Single Game Analysis"]
        )
        
        if prediction_mode == "Weekly Predictions":
            week = st.selectbox("Select Week", options=list(range(1, 19)))
            use_vegas_odds = st.checkbox("Use Vegas Odds", value=True)
            
        else:
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Home Team", options=sorted(team_stats.keys()))
            with col2:
                away_team = st.selectbox("Away Team", options=sorted(team_stats.keys()))
            
            use_vegas_odds = st.checkbox("Use Vegas Odds", value=True)
    
    # Main content
    if prediction_mode == "Weekly Predictions":
        display_weekly_predictions(week, df_prepared, team_stats, schedule, use_vegas_odds)
    else:
        display_single_game_analysis(home_team, away_team, df_prepared, team_stats, use_vegas_odds)

def display_weekly_predictions(week, df, team_stats, schedule, use_vegas_odds):
    st.header(f"📅 Week {week} Predictions")
    
    # Load Vegas odds if requested
    vegas_odds = load_vegas_odds(week) if use_vegas_odds else None
    
    # Get schedule for the week
    if schedule and str(week) in schedule['weeks']:
        week_games = schedule['weeks'][str(week)]
    else:
        st.error("❌ No schedule found for this week")
        return
    
    # Generate predictions for each game
    predictions = []
    
    for game in week_games:
        home_team = game['home']
        away_team = game['away']
        
        # Get Vegas odds for this game
        game_vegas_odds = get_vegas_game_odds(vegas_odds, home_team, away_team) if vegas_odds else None
        
        # Make predictions
        predicted_winner, home_prob, away_prob, win_confidence = predict_winner(home_team, away_team, team_stats, df)
        predicted_home_score, predicted_away_score = predict_score(home_team, away_team, df)
        predicted_total = predicted_home_score + predicted_away_score
        
        # Analyze against Vegas if available
        if game_vegas_odds:
            # For spread and total analysis, you would need to extract those from your odds data
            spread_analysis, spread_confidence = "No spread data", 0.5
            total_analysis, total_confidence = "No total data", 0.5
        else:
            spread_analysis, spread_confidence = "No Vegas data", 0.5
            total_analysis, total_confidence = "No Vegas data", 0.5
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'home_win_prob': home_prob,
            'away_win_prob': away_prob,
            'predicted_home_score': predicted_home_score,
            'predicted_away_score': predicted_away_score,
            'predicted_total': predicted_total,
            'win_confidence': win_confidence,
            'spread_analysis': spread_analysis,
            'spread_confidence': spread_confidence,
            'total_analysis': total_analysis,
            'total_confidence': total_confidence
        })
    
    # Display predictions
    for prediction in predictions:
        display_game_prediction(prediction)

def display_single_game_analysis(home_team, away_team, df, team_stats, use_vegas_odds):
    st.header(f"🔍 Game Analysis: {away_team} @ {home_team}")
    
    # Load recent Vegas odds (you might want to specify week for single games)
    vegas_odds = None  # You can modify this to load specific odds
    
    # Make predictions
    predicted_winner, home_prob, away_prob, win_confidence = predict_winner(home_team, away_team, team_stats, df)
    predicted_home_score, predicted_away_score = predict_score(home_team, away_team, df)
    predicted_total = predicted_home_score + predicted_away_score
    
    # Display team comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team}")
        home_recent = get_team_recent_stats(home_team, df)
        st.metric("Recent Win %", f"{home_recent['win_pct']:.1%}")
        st.metric("Points Per Game", f"{home_recent['ppg']:.1f}")
        st.metric("Points Allowed", f"{home_recent['ppg_against']:.1f}")
    
    with col2:
        st.subheader(f"✈️ {away_team}")
        away_recent = get_team_recent_stats(away_team, df)
        st.metric("Recent Win %", f"{away_recent['win_pct']:.1%}")
        st.metric("Points Per Game", f"{away_recent['ppg']:.1f}")
        st.metric("Points Allowed", f"{away_recent['ppg_against']:.1f}")
    
    # Display predictions
    st.subheader("🎯 Model Predictions")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Predicted Winner", predicted_winner)
        st.write(f"Confidence: {win_confidence:.1%}")
    
    with col4:
        st.metric("Projected Score", f"{predicted_home_score} - {predicted_away_score}")
        st.write(f"Total: {predicted_total} points")
    
    with col5:
        st.metric("Win Probability", f"{home_team}: {home_prob:.1%}")
        st.write(f"{away_team}: {away_prob:.1%}")
    
    # Betting recommendations
    if use_vegas_odds and vegas_odds:
        st.subheader("💰 Betting Analysis")
        # Add your Vegas odds analysis here
        st.info("Vegas odds analysis would appear here with actual odds data")
    else:
        st.warning("⚠️ Vegas odds not available for this analysis")

def display_game_prediction(prediction):
    """Display a single game prediction in a nice format"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.write(f"**{prediction['away_team']}**")
        st.write(f"Win Probability: {prediction['away_win_prob']:.1%}")
    
    with col2:
        st.write("**@**")
        st.write(f"**{prediction['predicted_home_score']} - {prediction['predicted_away_score']}**")
    
    with col3:
        st.write(f"**{prediction['home_team']}**")
        st.write(f"Win Probability: {prediction['home_win_prob']:.1%}")
    
    # Prediction details
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Predicted Winner", prediction['predicted_winner'])
    
    with col5:
        st.metric("Total Points", prediction['predicted_total'])
    
    with col6:
        confidence_color = "green" if prediction['win_confidence'] > 0.7 else "orange" if prediction['win_confidence'] > 0.6 else "red"
        st.metric("Confidence", f"{prediction['win_confidence']:.1%}")
    
    # Betting analysis
    if prediction['spread_analysis'] != "No Vegas data":
        st.info(f"**Spread:** {prediction['spread_analysis']} (Confidence: {prediction['spread_confidence']:.1%})")
    
    if prediction['total_analysis'] != "No Vegas data":
        total_color = "green" if "OVER" in prediction['total_analysis'] else "red"
        st.info(f"**Total:** {prediction['total_analysis']} (Confidence: {prediction['total_confidence']:.1%})")

if __name__ == "__main__":
    main()
