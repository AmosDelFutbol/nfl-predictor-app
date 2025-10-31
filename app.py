import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
import joblib
import os
import json
warnings.filterwarnings('ignore')

# Streamlit configuration
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

def load_vegas_odds(week):
    """Load Vegas odds for specific week"""
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

def validate_data(df):
    """Validate the loaded data has required columns"""
    required_columns = ['team_home', 'team_away', 'score_home', 'score_away', 
                       'team_favorite_id', 'spread_favorite', 'over_under_line',
                       'schedule_season', 'schedule_date']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return False
    
    st.success("✅ All required columns present")
    return True

def create_team_mapping():
    """Create mapping between team abbreviations and full names"""
    team_mapping = {
        'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins', 'NE': 'New England Patriots', 'NYJ': 'New York Jets',
        'BAL': 'Baltimore Ravens', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'PIT': 'Pittsburgh Steelers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'TEN': 'Tennessee Titans',
        'DEN': 'Denver Broncos', 'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
        'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants', 'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders',
        'CHI': 'Chicago Bears', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
        'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints', 'TB': 'Tampa Bay Buccaneers',
        'ARI': 'Arizona Cardinals', 'LAR': 'Los Angeles Rams', 'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks'
    }
    return team_mapping

def standardize_team_data(df, team_mapping):
    """Standardize team names and map abbreviations to full names"""
    df['team_favorite'] = df['team_favorite_id'].map(team_mapping)
    
    washington_mapping = {
        'Washington Redskins': 'Washington Commanders',
        'Washington Football Team': 'Washington Commanders',
        'Washington Commanders': 'Washington Commanders'
    }
    
    df['team_home'] = df['team_home'].replace(washington_mapping)
    df['team_away'] = df['team_away'].replace(washington_mapping)
    df['team_favorite'] = df['team_favorite'].replace(washington_mapping)
    
    return df

def are_teams_in_same_division(team1, team2):
    """Check if two teams are in the same division"""
    divisions = {
        'AFC East': ['Buffalo Bills', 'Miami Dolphins', 'New England Patriots', 'New York Jets'],
        'AFC North': ['Baltimore Ravens', 'Cincinnati Bengals', 'Cleveland Browns', 'Pittsburgh Steelers'],
        'AFC South': ['Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Tennessee Titans'],
        'AFC West': ['Denver Broncos', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers'],
        'NFC East': ['Dallas Cowboys', 'New York Giants', 'Philadelphia Eagles', 'Washington Commanders'],
        'NFC North': ['Chicago Bears', 'Detroit Lions', 'Green Bay Packers', 'Minnesota Vikings'],
        'NFC South': ['Atlanta Falcons', 'Carolina Panthers', 'New Orleans Saints', 'Tampa Bay Buccaneers'],
        'NFC West': ['Arizona Cardinals', 'Los Angeles Rams', 'San Francisco 49ers', 'Seattle Seahawks']
    }
    
    for division, teams in divisions.items():
        if team1 in teams and team2 in teams:
            return True
    return False

def is_primetime(date_str):
    """Check if game is in primetime"""
    try:
        game_date = pd.to_datetime(date_str)
        if game_date.weekday() in [3, 6, 0]:  # Thursday, Sunday, Monday
            return True
        return False
    except:
        return False

def add_advanced_metrics(df):
    """Add advanced betting and performance metrics"""
    df['score_differential'] = abs(df['score_home'] - df['score_away'])
    df['total_points'] = df['score_home'] + df['score_away']
    
    df['favorite_won'] = np.where(
        ((df['team_favorite'] == df['team_home']) & (df['home_win'] == 1)) |
        ((df['team_favorite'] == df['team_away']) & (df['away_win'] == 1)), 1, 0
    )
    
    df['close_game'] = (df['score_differential'] <= 7).astype(int)
    df['blowout_game'] = (df['score_differential'] >= 17).astype(int)
    df['over_under_differential'] = df['total_points'] - df['over_under_line']
    
    return df

def add_advanced_features(df, team_stats):
    """Add more sophisticated features for better predictions"""
    st.info("🔄 Adding advanced features...")
    
    def calculate_team_sos(team, df):
        """Calculate strength of schedule for a team"""
        team_games = pd.concat([
            df[df['team_home'] == team],
            df[df['team_away'] == team]
        ])
        
        if len(team_games) == 0:
            return 0.5
        
        opponent_win_pct = []
        for _, game in team_games.iterrows():
            if game['team_home'] == team:
                opponent = game['team_away']
                opponent_wins = team_stats.get(opponent, {}).get('total_wins', 0)
                opponent_games = team_stats.get(opponent, {}).get('total_games', 1)
                opponent_win_pct.append(opponent_wins / opponent_games)
            else:
                opponent = game['team_home']
                opponent_wins = team_stats.get(opponent, {}).get('total_wins', 0)
                opponent_games = team_stats.get(opponent, {}).get('total_games', 1)
                opponent_win_pct.append(opponent_wins / opponent_games)
        
        return np.mean(opponent_win_pct) if opponent_win_pct else 0.5
    
    sos_cache = {}
    for team in team_stats.keys():
        sos_cache[team] = calculate_team_sos(team, df)
    
    df['home_sos'] = df['team_home'].map(sos_cache)
    df['away_sos'] = df['team_away'].map(sos_cache)
    
    df['home_efficiency'] = df['score_home'] / (df['over_under_line'] / 2)
    df['away_efficiency'] = df['score_away'] / (df['over_under_line'] / 2)
    
    df['divisional_game'] = df.apply(lambda x: are_teams_in_same_division(x['team_home'], x['team_away']), axis=1)
    df['primetime_game'] = df['schedule_date'].apply(is_primetime)
    
    return df

def prepare_data(df):
    """Prepare data for 2020-2025 analysis"""
    team_mapping = create_team_mapping()
    df = standardize_team_data(df, team_mapping)
    
    df_recent = df[df['schedule_season'].between(2020, 2025)].copy()
    
    df_recent['score_home'] = pd.to_numeric(df_recent['score_home'], errors='coerce')
    df_recent['score_away'] = pd.to_numeric(df_recent['score_away'], errors='coerce')
    df_recent['spread_favorite'] = pd.to_numeric(df_recent['spread_favorite'], errors='coerce')
    df_recent['over_under_line'] = pd.to_numeric(df_recent['over_under_line'], errors='coerce')
    
    df_recent = df_recent.dropna(subset=['score_home', 'score_away', 'team_favorite', 'spread_favorite', 'over_under_line'])
    
    st.success(f"✅ Cleaned data: {len(df_recent):,} games remaining after cleaning")
    
    df_recent['home_win'] = (df_recent['score_home'] > df_recent['score_away']).astype(int)
    df_recent['away_win'] = (df_recent['score_home'] < df_recent['score_away']).astype(int)
    df_recent['tie'] = (df_recent['score_home'] == df_recent['score_away']).astype(int)
    
    df_recent['actual_spread'] = df_recent['score_home'] - df_recent['score_away']
    df_recent['total_points'] = df_recent['score_home'] + df_recent['score_away']
    
    df_recent['favorite_covered'] = 0
    
    for idx, game in df_recent.iterrows():
        if game['team_favorite'] == game['team_home']:
            if game['actual_spread'] > -game['spread_favorite']:
                df_recent.at[idx, 'favorite_covered'] = 1
        else:
            if game['actual_spread'] < game['spread_favorite']:
                df_recent.at[idx, 'favorite_covered'] = 1
    
    df_recent['underdog_covered'] = 1 - df_recent['favorite_covered']
    
    df_recent['over_hit'] = (df_recent['total_points'] > df_recent['over_under_line']).astype(int)
    df_recent['under_hit'] = (df_recent['total_points'] < df_recent['over_under_line']).astype(int)
    df_recent['push'] = (df_recent['total_points'] == df_recent['over_under_line']).astype(int)
    
    df_recent['winner'] = np.where(df_recent['home_win'] == 1, df_recent['team_home'], 
                                  np.where(df_recent['away_win'] == 1, df_recent['team_away'], 'Tie'))
    
    return df_recent

def calculate_team_stats(df):
    """Calculate comprehensive team statistics including ATS and Over/Under"""
    teams = pd.unique(np.concatenate([df['team_home'].unique(), df['team_away'].unique()]))
    team_stats = {}
    
    for team in teams:
        home_games = df[df['team_home'] == team]
        home_wins = home_games['home_win'].sum()
        home_total = len(home_games)
        
        away_games = df[df['team_away'] == team]
        away_wins = away_games['away_win'].sum()
        away_total = len(away_games)
        
        total_games = home_total + away_total
        total_wins = home_wins + away_wins
        
        fav_wins = 0
        fav_losses = 0
        dog_wins = 0
        dog_losses = 0
        
        fav_ats_wins = 0
        dog_ats_wins = 0
        total_ats_wins = 0
        
        all_team_games = pd.concat([home_games, away_games])
        
        for _, game in all_team_games.iterrows():
            was_favorite = (game['team_favorite'] == team)
            
            if game['team_home'] == team:
                team_won = (game['home_win'] == 1)
            else:
                team_won = (game['away_win'] == 1)
            
            if was_favorite:
                if team_won:
                    fav_wins += 1
                else:
                    fav_losses += 1
            else:
                if team_won:
                    dog_wins += 1
                else:
                    dog_losses += 1
            
            if was_favorite:
                if game['favorite_covered'] == 1:
                    fav_ats_wins += 1
                    total_ats_wins += 1
            else:
                if game['underdog_covered'] == 1:
                    dog_ats_wins += 1
                    total_ats_wins += 1
        
        team_games = pd.concat([home_games, away_games])
        overs = len(team_games[team_games['over_hit'] == 1])
        unders = len(team_games[team_games['under_hit'] == 1])
        pushes = len(team_games[team_games['push'] == 1])
        
        team_stats[team] = {
            'home_games': home_total,
            'home_wins': home_wins,
            'home_losses': home_total - home_wins,
            'home_win_pct': round(home_wins / home_total * 100, 1) if home_total > 0 else 0,
            
            'away_games': away_total,
            'away_wins': away_wins,
            'away_losses': away_total - away_wins,
            'away_win_pct': round(away_wins / away_total * 100, 1) if away_total > 0 else 0,
            
            'total_games': total_games,
            'total_wins': total_wins,
            'total_losses': total_games - total_wins,
            'total_win_pct': round(total_wins / total_games * 100, 1) if total_games > 0 else 0,
            
            'fav_games': fav_wins + fav_losses,
            'fav_wins': fav_wins,
            'fav_losses': fav_losses,
            'fav_win_pct': round(fav_wins / (fav_wins + fav_losses) * 100, 1) if (fav_wins + fav_losses) > 0 else 0,
            
            'dog_games': dog_wins + dog_losses,
            'dog_wins': dog_wins,
            'dog_losses': dog_losses,
            'dog_win_pct': round(dog_wins / (dog_wins + dog_losses) * 100, 1) if (dog_wins + dog_losses) > 0 else 0,
            
            'fav_ats_wins': fav_ats_wins,
            'fav_ats_losses': (fav_wins + fav_losses) - fav_ats_wins,
            'fav_ats_pct': round(fav_ats_wins / (fav_wins + fav_losses) * 100, 1) if (fav_wins + fav_losses) > 0 else 0,
            
            'dog_ats_wins': dog_ats_wins,
            'dog_ats_losses': (dog_wins + dog_losses) - dog_ats_wins,
            'dog_ats_pct': round(dog_ats_wins / (dog_wins + dog_losses) * 100, 1) if (dog_wins + dog_losses) > 0 else 0,
            
            'total_ats_wins': total_ats_wins,
            'total_ats_losses': total_games - total_ats_wins,
            'total_ats_pct': round(total_ats_wins / total_games * 100, 1) if total_games > 0 else 0,
            
            'over_games': overs,
            'under_games': unders,
            'push_games': pushes,
            'over_pct': round(overs / (overs + unders) * 100, 1) if (overs + unders) > 0 else 0,
            'under_pct': round(unders / (overs + unders) * 100, 1) if (overs + unders) > 0 else 0,
        }
    
    return team_stats

def get_week_schedule_from_json(schedule, week):
    """Get the schedule for a specific week from JSON"""
    week_str = str(week)
    if week_str not in schedule['weeks']:
        st.error(f"❌ Week {week} not found in schedule")
        return None
    
    week_games = schedule['weeks'][week_str]
    
    # Convert to DataFrame format
    games_data = []
    for game in week_games:
        games_data.append({
            'team_home': game['home'],
            'team_away': game['away'],
            'schedule_date': game['date']
        })
    
    return pd.DataFrame(games_data)

def create_game_features_for_prediction(home_team, away_team, team_stats, df):
    """Create features for a specific game prediction"""
    def get_current_stats(team, games_back=5):
        team_games = pd.concat([
            df[df['team_home'] == team],
            df[df['team_away'] == team]
        ]).sort_values('schedule_date').tail(games_back)
        
        if len(team_games) == 0:
            return {
                'recent_win_pct': 0.5,
                'recent_ppg': 20,
                'recent_ppg_against': 20,
                'recent_ats_pct': 0.5
            }
        
        wins = 0
        points_for = 0
        points_against = 0
        ats_wins = 0
        total_games = len(team_games)
        
        for _, game in team_games.iterrows():
            if game['team_home'] == team:
                if game['home_win'] == 1:
                    wins += 1
                points_for += game['score_home']
                points_against += game['score_away']
                if game['team_favorite'] == team and game['favorite_covered'] == 1:
                    ats_wins += 1
                elif game['team_favorite'] != team and game['underdog_covered'] == 1:
                    ats_wins += 1
            else:
                if game['away_win'] == 1:
                    wins += 1
                points_for += game['score_away']
                points_against += game['score_home']
                if game['team_favorite'] == team and game['favorite_covered'] == 1:
                    ats_wins += 1
                elif game['team_favorite'] != team and game['underdog_covered'] == 1:
                    ats_wins += 1
        
        return {
            'recent_win_pct': wins / total_games,
            'recent_ppg': points_for / total_games,
            'recent_ppg_against': points_against / total_games,
            'recent_ats_pct': ats_wins / total_games
        }
    
    home_recent = get_current_stats(home_team)
    away_recent = get_current_stats(away_team)
    
    home_overall = team_stats.get(home_team, {})
    away_overall = team_stats.get(away_team, {})
    
    feature_vector = [
        home_recent['recent_win_pct'],
        home_recent['recent_ppg'],
        home_recent['recent_ppg_against'],
        home_recent['recent_ats_pct'],
        home_overall.get('home_win_pct', 0.5) / 100,
        home_overall.get('total_win_pct', 0.5) / 100,
        
        away_recent['recent_win_pct'],
        away_recent['recent_ppg'],
        away_recent['recent_ppg_against'],
        away_recent['recent_ats_pct'],
        away_overall.get('away_win_pct', 0.5) / 100,
        away_overall.get('total_win_pct', 0.5) / 100,
        
        home_overall.get('home_win_pct', 0.5) / 100 - away_overall.get('away_win_pct', 0.5) / 100,
        home_recent['recent_ppg'] - away_recent['recent_ppg_against'],
        away_recent['recent_ppg'] - home_recent['recent_ppg_against'],
    ]
    
    return np.array([feature_vector])

def project_game_score(home_team, away_team, team_stats, df):
    """Project final score for a game"""
    def get_team_averages(team):
        team_games = pd.concat([
            df[df['team_home'] == team],
            df[df['team_away'] == team]
        ]).tail(5)
        
        if len(team_games) == 0:
            return {'ppg': 21, 'ppg_against': 21}
        
        points_for = 0
        points_against = 0
        
        for _, game in team_games.iterrows():
            if game['team_home'] == team:
                points_for += game['score_home']
                points_against += game['score_away']
            else:
                points_for += game['score_away']
                points_against += game['score_home']
        
        return {
            'ppg': points_for / len(team_games),
            'ppg_against': points_against / len(team_games)
        }
    
    home_avg = get_team_averages(home_team)
    away_avg = get_team_averages(away_team)
    
    home_score_proj = (home_avg['ppg'] + away_avg['ppg_against']) / 2
    away_score_proj = (away_avg['ppg'] + home_avg['ppg_against']) / 2
    
    home_field_advantage = 2.5
    home_score_proj += home_field_advantage
    away_score_proj -= home_field_advantage / 2
    
    return round(home_score_proj), round(away_score_proj)

def calculate_simple_projection(home_team, away_team, team_stats, df):
    """Calculate projection using simple averages (fallback method)"""
    def get_team_averages(team):
        team_games = pd.concat([
            df[df['team_home'] == team],
            df[df['team_away'] == team]
        ]).tail(5)
        
        if len(team_games) == 0:
            return {'ppg': 21, 'ppg_against': 21, 'win_pct': 0.5}
        
        points_for = 0
        points_against = 0
        wins = 0
        
        for _, game in team_games.iterrows():
            if game['team_home'] == team:
                points_for += game['score_home']
                points_against += game['score_away']
                wins += game['home_win']
            else:
                points_for += game['score_away']
                points_against += game['score_home']
                wins += game['away_win']
        
        return {
            'ppg': points_for / len(team_games),
            'ppg_against': points_against / len(team_games),
            'win_pct': wins / len(team_games)
        }
    
    home_avg = get_team_averages(home_team)
    away_avg = get_team_averages(away_team)
    
    home_score = (home_avg['ppg'] + away_avg['ppg_against']) / 2 + 2.5
    away_score = (away_avg['ppg'] + home_avg['ppg_against']) / 2 - 1.0
    
    home_win_prob = home_avg['win_pct'] * 0.6 + (1 - away_avg['win_pct']) * 0.4
    
    return {
        'home_score': round(home_score),
        'away_score': round(away_score),
        'home_win_prob': home_win_prob,
        'total': round(home_score + away_score)
    }

def generate_weekly_projections(models, df, team_stats, schedule, season, week, use_vegas_odds=False):
    """Generate projections for all games in a specific week"""
    st.header(f"📅 Week {week} Projections - {season} Season")
    
    # Get the schedule for the week from JSON
    week_schedule = get_week_schedule_from_json(schedule, week)
    
    if week_schedule is None or len(week_schedule) == 0:
        st.error(f"❌ No games available for {season} Week {week}")
        return
    
    # Load Vegas odds if requested
    vegas_odds = load_vegas_odds(week) if use_vegas_odds else None
    
    # Check if these are real games or future projections
    real_games = df[(df['schedule_season'] == season) & (df['schedule_week'] == week)]
    is_future_week = len(real_games) == 0
    
    if is_future_week:
        st.info(f"🎯 Future Week Projections - Using 2025 NFL Schedule")
    else:
        st.info(f"🎯 Found {len(real_games)} actual games for Week {week}")
    
    st.write(f"**Schedule: {len(week_schedule)} games this week**")
    
    projections = []
    
    for _, game in week_schedule.iterrows():
        home_team = game['team_home']
        away_team = game['team_away']
        game_date = game['schedule_date']
        
        with st.expander(f"🏈 {away_team} @ {home_team}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Date:** {game_date}")
                
            # Get features for prediction
            X_new = create_game_features_for_prediction(home_team, away_team, team_stats, df)
            
            if models:
                # ML-based projections
                projected_spread = models['spread'].predict(X_new)[0]
                projected_total = models['total'].predict(X_new)[0]
                win_probability = models['win_prob'].predict_proba(X_new)[0][1]
                
                # Determine favorite
                if projected_spread < 0:
                    favorite = home_team
                    underdog = away_team
                    spread = abs(projected_spread)
                else:
                    favorite = away_team
                    underdog = home_team
                    spread = projected_spread
            else:
                # Fallback to simple projection
                projection = calculate_simple_projection(home_team, away_team, team_stats, df)
                win_probability = projection['home_win_prob']
                projected_total = projection['total']
                
                # Simple spread calculation
                spread = abs(projection['home_score'] - projection['away_score'])
                if projection['home_score'] > projection['away_score']:
                    favorite = home_team
                    underdog = away_team
                else:
                    favorite = away_team
                    underdog = home_team
            
            # Score projection
            home_score, away_score = project_game_score(home_team, away_team, team_stats, df)
            
            # Display predictions
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Projected Score", f"{home_score} - {away_score}")
                st.metric("Total Points", f"{home_score + away_score}")
                
            with col4:
                st.metric("Spread", f"{favorite} -{spread:.1f}")
                st.metric("Win Probability", f"{home_team}: {win_probability:.1%}")
                
            with col5:
                # Over/Under analysis
                total_points = home_score + away_score
                if total_points > 51:
                    over_under = "STRONG OVER"
                elif total_points > 47:
                    over_under = "LEAN OVER"
                elif total_points < 39:
                    over_under = "STRONG UNDER"
                elif total_points < 43:
                    over_under = "LEAN UNDER"
                else:
                    over_under = "NEUTRAL"
                    
                st.metric("Over/Under", over_under)
                
                # Confidence
                margin = abs(home_score - away_score)
                if margin > 14:
                    confidence = "HIGH"
                elif margin > 7:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                st.metric("Confidence", confidence)
            
            projections.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'favorite': favorite,
                'spread': spread,
                'total': projected_total,
                'over_under': over_under,
                'home_win_prob': win_probability,
                'confidence': confidence
            })
    
    return projections

def main():
    st.title("🏈 NFL Predictor Pro")
    st.markdown("### Advanced NFL Betting Analysis & Projections")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_nfl_data()
        schedule = load_nfl_schedule()
        
        if df is None:
            st.error("❌ Could not load historical data.")
            return
            
        if schedule is None:
            st.warning("⚠️ Could not load schedule. Some features may not work.")
        
        if not validate_data(df):
            return
        
        df_prepared = prepare_data(df)
        team_stats = calculate_team_stats(df_prepared)
        df_prepared = add_advanced_features(df_prepared, team_stats)
        df_prepared = add_advanced_metrics(df_prepared)
    
    # Sidebar navigation
    st.sidebar.header("🎯 Navigation")
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Weekly Projections", "Team Comparison", "Single Game Analysis", "Betting Insights"]
    )
    
    if analysis_mode == "Weekly Projections":
        st.sidebar.header("📅 Week Selection")
        week = st.sidebar.selectbox("Select Week", options=list(range(1, 19)))
        use_vegas_odds = st.sidebar.checkbox("Use Vegas Odds", value=False)
        
        # Load models
        models = None
        try:
            if os.path.exists("nfl_models.joblib"):
                models = joblib.load("nfl_models.joblib")
                st.sidebar.success("✅ ML Models Loaded")
        except:
            st.sidebar.info("ℹ️ Using simple projection models")
        
        generate_weekly_projections(models, df_prepared, team_stats, schedule, 2025, week, use_vegas_odds)
        
    elif analysis_mode == "Team Comparison":
        st.header("🏈 Team Comparison")
        teams = sorted(team_stats.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home Team", teams)
        with col2:
            away_team = st.selectbox("Away Team", teams)
        
        if st.button("Compare Teams"):
            # Display team comparison (you'll need to adapt your display_team_comparison function for Streamlit)
            st.info("Team comparison feature would be implemented here")
            
    elif analysis_mode == "Single Game Analysis":
        st.header("🔍 Single Game Analysis")
        # Implement single game analysis
        
    elif analysis_mode == "Betting Insights":
        st.header("💰 Betting Insights")
        # Implement betting insights

if __name__ == "__main__":
    main()
