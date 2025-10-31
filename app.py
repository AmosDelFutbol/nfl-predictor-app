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

def load_vegas_odds():
    """Load available Vegas odds - handles the fact you only have week_9_odds.json"""
    available_odds_files = []
    
    # Check for week_9_odds.json (the file you have)
    if os.path.exists("week_9_odds.json"):
        available_odds_files.append(9)
    
    # You can add more weeks here as you get more odds files
    # if os.path.exists("week_1_odds.json"):
    #     available_odds_files.append(1)
    
    return available_odds_files

def load_specific_vegas_odds(week):
    """Load specific week's Vegas odds"""
    try:
        odds_file = f"week_{week}_odds.json"
        if os.path.exists(odds_file):
            with open(odds_file, 'r') as f:
                odds = json.load(f)
            return odds
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading odds for Week {week}: {e}")
        return None

def prepare_data(df):
    """Prepare data for analysis"""
    # Filter for recent seasons
    df_recent = df[df['schedule_season'].between(2020, 2025)].copy()
    
    # Clean and prepare data
    df_recent['score_home'] = pd.to_numeric(df_recent['score_home'], errors='coerce')
    df_recent['score_away'] = pd.to_numeric(df_recent['score_away'], errors='coerce')
    df_recent['spread_favorite'] = pd.to_numeric(df_recent['spread_favorite'], errors='coerce')
    df_recent['over_under_line'] = pd.to_numeric(df_recent['over_under_line'], errors='coerce')
    
    # Remove games with missing scores
    df_recent = df_recent.dropna(subset=['score_home', 'score_away'])
    
    # Create derived columns
    df_recent['home_win'] = (df_recent['score_home'] > df_recent['score_away']).astype(int)
    df_recent['actual_spread'] = df_recent['score_home'] - df_recent['score_away']
    df_recent['total_points'] = df_recent['score_home'] + df_recent['score_away']
    
    return df_recent

def calculate_team_stats(df):
    """Calculate comprehensive team statistics"""
    teams = pd.unique(np.concatenate([df['team_home'].unique(), df['team_away'].unique()]))
    team_stats = {}
    
    for team in teams:
        home_games = df[df['team_home'] == team]
        away_games = df[df['team_away'] == team]
        
        total_games = len(home_games) + len(away_games)
        total_wins = home_games['home_win'].sum() + (len(away_games) - away_games['home_win'].sum())
        
        # Calculate points for and against
        points_for = home_games['score_home'].sum() + away_games['score_away'].sum()
        points_against = home_games['score_away'].sum() + away_games['score_home'].sum()
        
        team_stats[team] = {
            'total_games': total_games,
            'total_wins': total_wins,
            'total_losses': total_games - total_wins,
            'win_pct': total_wins / total_games if total_games > 0 else 0.5,
            'points_for': points_for,
            'points_against': points_against,
            'ppg': points_for / total_games if total_games > 0 else 21,
            'ppg_against': points_against / total_games if total_games > 0 else 21
        }
    
    return team_stats

def get_team_recent_stats(team, df, games_back=8):
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
        'ppg_against': points_against / len(team_games),
        'games_played': len(team_games)
    }

def predict_winner(home_team, away_team, team_stats, df):
    """Predict game winner based on team stats and recent performance"""
    home_recent = get_team_recent_stats(home_team, df)
    away_recent = get_team_recent_stats(away_team, df)
    
    home_overall = team_stats[home_team]
    away_overall = team_stats[away_team]
    
    # Weighted prediction (recent form 70%, overall 30%)
    home_strength = (home_recent['win_pct'] * 0.7) + (home_overall['win_pct'] * 0.3)
    away_strength = (away_recent['win_pct'] * 0.7) + (away_overall['win_pct'] * 0.3)
    
    # Home field advantage
    home_advantage = 0.06  # ~2.5 points
    
    home_prob = (home_strength / (home_strength + away_strength)) + home_advantage
    away_prob = 1 - home_prob
    
    # Ensure probabilities are between 0 and 1
    home_prob = max(0.05, min(0.95, home_prob))
    away_prob = 1 - home_prob
    
    predicted_winner = home_team if home_prob > away_prob else away_team
    confidence = max(home_prob, away_prob)
    
    return predicted_winner, home_prob, away_prob, confidence

def predict_score(home_team, away_team, df):
    """Predict final score based on recent performance"""
    home_recent = get_team_recent_stats(home_team, df)
    away_recent = get_team_recent_stats(away_team, df)
    
    # More sophisticated scoring projection
    home_offense = home_recent['ppg']
    home_defense = home_recent['ppg_against']
    away_offense = away_recent['ppg']
    away_defense = away_recent['ppg_against']
    
    # Project scores with adjustments
    home_score = (home_offense + away_defense) / 2 + 2.8  # Home field advantage
    away_score = (away_offense + home_defense) / 2 - 1.2  # Away disadvantage
    
    # Add some randomness and round
    home_score = max(0, round(home_score + np.random.normal(0, 1)))
    away_score = max(0, round(away_score + np.random.normal(0, 1)))
    
    return home_score, away_score

def extract_vegas_data(vegas_odds, home_team, away_team):
    """Extract spread and total from Vegas odds data"""
    if not vegas_odds:
        return None, None, None
    
    # Find all odds for this game
    game_odds = [odds for odds in vegas_odds if odds['home_team'] == home_team and odds['away_team'] == away_team]
    
    if not game_odds:
        return None, None, None
    
    # Extract moneyline
    home_ml = None
    away_ml = None
    
    for odds in game_odds:
        if odds['market'] == 'h2h':
            if odds['label'] == home_team:
                home_ml = odds['price']
            elif odds['label'] == away_team:
                away_ml = odds['price']
    
    # For spread and totals, you would need to add that data to your odds files
    # For now, we'll use placeholders
    spread = None
    total = None
    
    return home_ml, away_ml, spread, total

def analyze_bet_recommendations(predicted_winner, home_prob, away_prob, predicted_home_score, predicted_away_score, home_ml, away_ml, vegas_spread, vegas_total):
    """Generate betting recommendations based on predictions vs Vegas"""
    recommendations = {
        'moneyline': {'bet': None, 'confidence': 0, 'edge': 0},
        'spread': {'bet': None, 'confidence': 0, 'edge': 0},
        'total': {'bet': None, 'confidence': 0, 'edge': 0}
    }
    
    # Moneyline analysis
    if home_ml and away_ml:
        # Convert odds to implied probabilities
        if home_ml < 0:
            vegas_home_prob = abs(home_ml) / (abs(home_ml) + 100)
        else:
            vegas_home_prob = 100 / (home_ml + 100)
        
        if away_ml < 0:
            vegas_away_prob = abs(away_ml) / (abs(away_ml) + 100)
        else:
            vegas_away_prob = 100 / (away_ml + 100)
        
        # Calculate edge
        home_edge = home_prob - vegas_home_prob
        away_edge = away_prob - vegas_away_prob
        
        if home_edge > 0.05:  # 5% edge threshold
            recommendations['moneyline'] = {
                'bet': f'{predicted_winner} ML',
                'confidence': min(home_edge * 10, 0.9),
                'edge': home_edge
            }
        elif away_edge > 0.05:
            recommendations['moneyline'] = {
                'bet': f'{predicted_winner} ML',
                'confidence': min(away_edge * 10, 0.9),
                'edge': away_edge
            }
    
    # Spread analysis (placeholder - need spread data in your odds files)
    if vegas_spread:
        actual_spread = predicted_home_score - predicted_away_score
        # Add spread analysis logic here when you have the data
    
    # Total analysis
    predicted_total = predicted_home_score + predicted_away_score
    if vegas_total:
        if predicted_total > vegas_total + 1:
            recommendations['total'] = {
                'bet': 'OVER',
                'confidence': min((predicted_total - vegas_total) / 10, 0.8),
                'edge': predicted_total - vegas_total
            }
        elif predicted_total < vegas_total - 1:
            recommendations['total'] = {
                'bet': 'UNDER',
                'confidence': min((vegas_total - predicted_total) / 10, 0.8),
                'edge': vegas_total - predicted_total
            }
    
    return recommendations

def main():
    st.title("🏈 NFL Predictor Pro")
    st.markdown("### Machine Learning Powered Betting Predictions")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_nfl_data()
        schedule = load_nfl_schedule()
        available_odds_weeks = load_vegas_odds()
        
        if df is None:
            st.error("❌ Could not load historical data.")
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
            ["Weekly Predictions", "Single Game Analysis", "Team Statistics"]
        )
        
        if prediction_mode == "Weekly Predictions":
            week = st.selectbox("Select Week", options=list(range(1, 19)))
            
            # Show available odds weeks
            if available_odds_weeks:
                st.info(f"📊 Vegas odds available for: Week {available_odds_weeks[0]}")
            
            use_vegas_odds = st.checkbox("Use Vegas Odds", value=True)
            
        elif prediction_mode == "Single Game Analysis":
            teams = sorted(team_stats.keys())
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Home Team", options=teams)
            with col2:
                away_team = st.selectbox("Away Team", options=teams)
            
            use_vegas_odds = st.checkbox("Use Vegas Odds", value=False)
            
        else:  # Team Statistics
            selected_team = st.selectbox("Select Team", options=sorted(team_stats.keys()))
    
    # Main content
    if prediction_mode == "Weekly Predictions":
        display_weekly_predictions(week, df_prepared, team_stats, schedule, use_vegas_odds, available_odds_weeks)
    elif prediction_mode == "Single Game Analysis":
        display_single_game_analysis(home_team, away_team, df_prepared, team_stats, use_vegas_odds)
    else:
        display_team_statistics(selected_team, team_stats)

def display_weekly_predictions(week, df, team_stats, schedule, use_vegas_odds, available_odds_weeks):
    st.header(f"📅 Week {week} Predictions")
    
    # Load Vegas odds if requested and available
    vegas_odds = None
    if use_vegas_odds and week in available_odds_weeks:
        vegas_odds = load_specific_vegas_odds(week)
        if vegas_odds:
            st.success(f"✅ Using Vegas odds for Week {week}")
        else:
            st.warning(f"⚠️ Could not load Vegas odds for Week {week}")
    elif use_vegas_odds:
        st.warning(f"⚠️ No Vegas odds available for Week {week}")
    
    # Get schedule for the week
    if schedule and str(week) in schedule['weeks']:
        week_games = schedule['weeks'][str(week)]
        st.write(f"**{len(week_games)} games scheduled for Week {week}**")
    else:
        st.error("❌ No schedule found for this week")
        return
    
    # Generate predictions for each game
    predictions = []
    
    for game in week_games:
        home_team = game['home']
        away_team = game['away']
        game_date = game['date']
        
        # Get Vegas odds for this game
        home_ml, away_ml, vegas_spread, vegas_total = extract_vegas_data(vegas_odds, home_team, away_team) if vegas_odds else (None, None, None, None)
        
        # Make predictions
        predicted_winner, home_prob, away_prob, win_confidence = predict_winner(home_team, away_team, team_stats, df)
        predicted_home_score, predicted_away_score = predict_score(home_team, away_team, df)
        predicted_total = predicted_home_score + predicted_away_score
        
        # Generate betting recommendations
        bet_recommendations = analyze_bet_recommendations(
            predicted_winner, home_prob, away_prob, predicted_home_score, predicted_away_score,
            home_ml, away_ml, vegas_spread, vegas_total
        )
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date,
            'predicted_winner': predicted_winner,
            'home_win_prob': home_prob,
            'away_win_prob': away_prob,
            'predicted_home_score': predicted_home_score,
            'predicted_away_score': predicted_away_score,
            'predicted_total': predicted_total,
            'win_confidence': win_confidence,
            'vegas_home_ml': home_ml,
            'vegas_away_ml': away_ml,
            'bet_recommendations': bet_recommendations
        })
    
    # Display predictions
    for prediction in predictions:
        display_game_prediction(prediction)

def display_single_game_analysis(home_team, away_team, df, team_stats, use_vegas_odds):
    st.header(f"🔍 Game Analysis: {away_team} @ {home_team}")
    
    # Make predictions
    predicted_winner, home_prob, away_prob, win_confidence = predict_winner(home_team, away_team, team_stats, df)
    predicted_home_score, predicted_away_score = predict_score(home_team, away_team, df)
    predicted_total = predicted_home_score + predicted_away_score
    
    # Display team comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team}")
        home_recent = get_team_recent_stats(home_team, df)
        home_overall = team_stats[home_team]
        
        st.metric("Recent Win %", f"{home_recent['win_pct']:.1%}")
        st.metric("Points Per Game", f"{home_recent['ppg']:.1f}")
        st.metric("Points Allowed", f"{home_recent['ppg_against']:.1f}")
        st.write(f"Overall Record: {home_overall['total_wins']}-{home_overall['total_losses']}")
    
    with col2:
        st.subheader(f"✈️ {away_team}")
        away_recent = get_team_recent_stats(away_team, df)
        away_overall = team_stats[away_team]
        
        st.metric("Recent Win %", f"{away_recent['win_pct']:.1%}")
        st.metric("Points Per Game", f"{away_recent['ppg']:.1f}")
        st.metric("Points Allowed", f"{away_recent['ppg_against']:.1f}")
        st.write(f"Overall Record: {away_overall['total_wins']}-{away_overall['total_losses']}")
    
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
    
    # Key insights
    st.subheader("🔑 Key Insights")
    spread = predicted_home_score - predicted_away_score
    
    if abs(spread) > 10:
        st.success(f"**Blowout Alert**: {predicted_winner} projected to win by {abs(spread)} points")
    elif abs(spread) > 6:
        st.info(f"**Comfortable Win**: {predicted_winner} projected to win by {abs(spread)} points")
    else:
        st.warning(f"**Close Game**: Projected margin of {abs(spread)} points")
    
    if predicted_total > 50:
        st.success("**High Scoring**: Expect an offensive shootout")
    elif predicted_total < 40:
        st.info("**Defensive Battle**: Lower scoring game expected")

def display_team_statistics(team, team_stats):
    st.header(f"📊 {team} Statistics")
    
    stats = team_stats[team]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Record", f"{stats['total_wins']}-{stats['total_losses']}")
        st.metric("Win Percentage", f"{stats['win_pct']:.1%}")
    
    with col2:
        st.metric("Points Per Game", f"{stats['ppg']:.1f}")
        st.metric("Points Against", f"{stats['ppg_against']:.1f}")
    
    with col3:
        point_diff = stats['points_for'] - stats['points_against']
        st.metric("Point Differential", f"{point_diff:+}")
        st.metric("Average Margin", f"{(stats['ppg'] - stats['ppg_against']):+.1f}")

def display_game_prediction(prediction):
    """Display a single game prediction in a nice format"""
    st.markdown("---")
    
    # Game header
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader(f"✈️ {prediction['away_team']}")
        st.write(f"Win Probability: {prediction['away_win_prob']:.1%}")
        if prediction['vegas_away_ml']:
            st.write(f"Vegas ML: {prediction['vegas_away_ml']:+}")
    
    with col2:
        st.subheader("@")
        st.subheader(f"**{prediction['predicted_home_score']} - {prediction['predicted_away_score']}**")
        st.write(f"Total: {prediction['predicted_total']} points")
    
    with col3:
        st.subheader(f"🏠 {prediction['home_team']}")
        st.write(f"Win Probability: {prediction['home_win_prob']:.1%}")
        if prediction['vegas_home_ml']:
            st.write(f"Vegas ML: {prediction['vegas_home_ml']:+}")
    
    # Prediction details and betting recommendations
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Predicted Winner", prediction['predicted_winner'])
        st.write(f"Confidence: {prediction['win_confidence']:.1%}")
    
    with col5:
        # Moneyline recommendation
        ml_bet = prediction['bet_recommendations']['moneyline']
        if ml_bet['bet']:
            st.success(f"**Moneyline**: {ml_bet['bet']}")
            st.write(f"Edge: {ml_bet['edge']:.1%}")
        else:
            st.info("No clear moneyline value")
    
    with col6:
        # Total recommendation
        total_bet = prediction['bet_recommendations']['total']
        if total_bet['bet']:
            color = "green" if total_bet['bet'] == 'OVER' else "red"
            st.metric("Total Pick", total_bet['bet'])
            st.write(f"Confidence: {total_bet['confidence']:.1%}")
        else:
            st.info("Total: No clear play")
    
    # Game date
    st.caption(f"📅 Scheduled: {prediction['game_date']}")

if __name__ == "__main__":
    main()
