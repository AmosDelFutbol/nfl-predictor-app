# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Set page config
st.set_page_config(
    page_title="NFL Week 10 Predictions",
    page_icon="🏈",
    layout="wide"
)

class NFLPredictor:
    def __init__(self):
        self.win_model = None
        self.score_model = None
        self.spread_model = None
        self.total_model = None
        self.schedule = None
        self.odds_data = None
        self.team_stats = {}
        self.scalers = {}
        
    def calculate_advanced_stats(self, games_df):
        """Calculate advanced team statistics using historical data"""
        st.info("📊 Calculating advanced team statistics...")
        
        advanced_stats = {}
        all_teams = list(set(games_df['team_home'].unique()) | set(games_df['team_away'].unique()))
        
        for team in all_teams:
            # Get team's games
            team_games = games_df[(games_df['team_home'] == team) | (games_df['team_away'] == team)].copy()
            team_games = team_games.sort_values('schedule_date')
            
            # Basic stats
            team_games['is_home'] = (team_games['team_home'] == team).astype(int)
            team_games['team_score'] = np.where(team_games['team_home'] == team, 
                                               team_games['score_home'], 
                                               team_games['score_away'])
            team_games['opponent_score'] = np.where(team_games['team_home'] == team, 
                                                  team_games['score_away'], 
                                                  team_games['score_away'])
            team_games['win'] = (team_games['team_score'] > team_games['opponent_score']).astype(int)
            
            # Rolling averages (last 8 games)
            team_games['win_pct'] = team_games['win'].rolling(8, min_periods=1).mean()
            team_games['points_for_avg'] = team_games['team_score'].rolling(8, min_periods=1).mean()
            team_games['points_against_avg'] = team_games['opponent_score'].rolling(8, min_periods=1).mean()
            
            # Advanced metrics
            team_games['point_differential'] = team_games['team_score'] - team_games['opponent_score']
            team_games['point_differential_avg'] = team_games['point_differential'].rolling(8, min_periods=1).mean()
            
            # Strength metrics
            team_games['offensive_efficiency'] = team_games['points_for_avg'] / team_games['points_for_avg'].mean()
            team_games['defensive_efficiency'] = team_games['points_against_avg'] / team_games['points_against_avg'].mean()
            
            # Store by date
            for _, row in team_games.iterrows():
                date = row['schedule_date']
                if team not in advanced_stats:
                    advanced_stats[team] = {}
                advanced_stats[team][date] = {
                    'win_pct': row['win_pct'],
                    'points_for_avg': row['points_for_avg'],
                    'points_against_avg': row['points_against_avg'],
                    'point_differential_avg': row['point_differential_avg'],
                    'offensive_efficiency': row['offensive_efficiency'],
                    'defensive_efficiency': row['defensive_efficiency']
                }
        
        return advanced_stats
    
    def load_pbp_data(self):
        """Load and process play-by-play data for pace calculation"""
        try:
            with open('pbp_data_2025.json', 'r') as f:
                pbp_data = json.load(f)
            pbp_df = pd.DataFrame(pbp_data)
            
            # Calculate pace metrics
            pace_stats = {}
            
            for team in pbp_df['possession_team'].unique():
                if pd.isna(team):
                    continue
                    
                team_plays = pbp_df[pbp_df['possession_team'] == team]
                
                # Calculate plays per game
                games_played = team_plays['game_id'].nunique()
                total_plays = len(team_plays)
                plays_per_game = total_plays / games_played if games_played > 0 else 65
                
                # Calculate time of possession metrics
                if 'play_duration' in pbp_df.columns:
                    avg_play_duration = team_plays['play_duration'].mean()
                else:
                    avg_play_duration = 25  # seconds default
                
                pace_stats[team] = {
                    'plays_per_game': plays_per_game,
                    'avg_play_duration': avg_play_duration,
                    'pace_factor': plays_per_game * avg_play_duration / 3600  # normalized pace
                }
            
            return pace_stats
            
        except FileNotFoundError:
            st.warning("Play-by-play data not found, using default pace metrics")
            return {}
        except Exception as e:
            st.warning(f"Could not process play-by-play data: {e}")
            return {}
    
    def train_models(self):
        """Train separate models for win probability, scores, spreads, and totals"""
        st.info("🔄 Training advanced NFL prediction models...")
        
        try:
            # Load historical data
            with open('spreadspoke_scores.json', 'r') as f:
                games = pd.DataFrame(json.load(f))
            
            # Data cleaning
            games = games[games['score_home'].notna() & games['score_away'].notna()]
            games['schedule_date'] = pd.to_datetime(games['schedule_date'])
            games = games[games['schedule_date'].dt.year >= 2020]
            
            # Calculate advanced stats
            advanced_stats = self.calculate_advanced_stats(games)
            
            # Load pace data
            pace_stats = self.load_pbp_data()
            
            # Prepare features for all models
            win_features, win_targets = [], []
            score_features, home_scores, away_scores = [], [], []
            spread_features, spread_targets = [], []
            total_features, total_targets = [], []
            
            for _, game in games.iterrows():
                home_team = game['team_home']
                away_team = game['team_away']
                game_date = game['schedule_date']
                
                # Get stats before this game
                home_stats = self.get_team_stats_before_date(advanced_stats, home_team, game_date)
                away_stats = self.get_team_stats_before_date(advanced_stats, away_team, game_date)
                
                if not home_stats or not away_stats:
                    continue
                
                # Get pace stats
                home_pace = pace_stats.get(home_team, {'pace_factor': 0.5, 'plays_per_game': 65})
                away_pace = pace_stats.get(away_team, {'pace_factor': 0.5, 'plays_per_game': 65})
                
                # Create feature vector
                feature_vector = [
                    # Home team stats
                    home_stats['win_pct'], home_stats['points_for_avg'], home_stats['points_against_avg'],
                    home_stats['point_differential_avg'], home_stats['offensive_efficiency'], home_stats['defensive_efficiency'],
                    home_pace['pace_factor'], home_pace['plays_per_game'],
                    
                    # Away team stats  
                    away_stats['win_pct'], away_stats['points_for_avg'], away_stats['points_against_avg'],
                    away_stats['point_differential_avg'], away_stats['offensive_efficiency'], away_stats['defensive_efficiency'],
                    away_pace['pace_factor'], away_pace['plays_per_game'],
                    
                    # Interaction features
                    home_stats['points_for_avg'] - away_stats['points_against_avg'],  # Home offense vs away defense
                    away_stats['points_for_avg'] - home_stats['points_against_avg'],  # Away offense vs home defense
                    1  # Home field advantage
                ]
                
                # Win model data
                win_features.append(feature_vector)
                win_targets.append(1 if game['score_home'] > game['score_away'] else 0)
                
                # Score model data
                score_features.append(feature_vector)
                home_scores.append(game['score_home'])
                away_scores.append(game['score_away'])
                
                # Spread model data
                spread_features.append(feature_vector)
                actual_spread = game['score_home'] - game['score_away']
                spread_targets.append(actual_spread)
                
                # Total model data
                total_features.append(feature_vector)
                total_points = game['score_home'] + game['score_away']
                total_targets.append(total_points)
            
            # Train models
            X_win = np.array(win_features)
            X_score = np.array(score_features)
            X_spread = np.array(spread_features)
            X_total = np.array(total_features)
            
            # Handle NaN values
            X_win = np.nan_to_num(X_win)
            X_score = np.nan_to_num(X_score)
            X_spread = np.nan_to_num(X_spread)
            X_total = np.nan_to_num(X_total)
            
            # Split data
            test_size = 0.2
            X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(X_win, win_targets, test_size=test_size, random_state=42)
            X_score_train, X_score_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
                X_score, home_scores, away_scores, test_size=test_size, random_state=42)
            X_spread_train, X_spread_test, y_spread_train, y_spread_test = train_test_split(X_spread, spread_targets, test_size=test_size, random_state=42)
            X_total_train, X_total_test, y_total_train, y_total_test = train_test_split(X_total, total_targets, test_size=test_size, random_state=42)
            
            # Scale features
            self.scalers['win'] = StandardScaler()
            self.scalers['score'] = StandardScaler()
            self.scalers['spread'] = StandardScaler()
            self.scalers['total'] = StandardScaler()
            
            X_win_train_scaled = self.scalers['win'].fit_transform(X_win_train)
            X_win_test_scaled = self.scalers['win'].transform(X_win_test)
            
            X_score_train_scaled = self.scalers['score'].fit_transform(X_score_train)
            X_score_test_scaled = self.scalers['score'].transform(X_score_test)
            
            X_spread_train_scaled = self.scalers['spread'].fit_transform(X_spread_train)
            X_spread_test_scaled = self.scalers['spread'].transform(X_spread_test)
            
            X_total_train_scaled = self.scalers['total'].fit_transform(X_total_train)
            X_total_test_scaled = self.scalers['total'].transform(X_total_test)
            
            # Train win probability model
            self.win_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
            self.win_model.fit(X_win_train_scaled, y_win_train)
            win_accuracy = self.win_model.score(X_win_test_scaled, y_win_test)
            
            # Train score prediction model
            self.score_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
            self.score_model.fit(X_score_train_scaled, np.column_stack([y_home_train, y_away_train]))
            score_mae = mean_absolute_error(
                np.column_stack([y_home_test, y_away_test]),
                self.score_model.predict(X_score_test_scaled)
            )
            
            # Train spread prediction model
            self.spread_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
            self.spread_model.fit(X_spread_train_scaled, y_spread_train)
            spread_mae = mean_absolute_error(y_spread_test, self.spread_model.predict(X_spread_test_scaled))
            
            # Train total points model
            self.total_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
            self.total_model.fit(X_total_train_scaled, y_total_train)
            total_mae = mean_absolute_error(y_total_test, self.total_model.predict(X_total_test_scaled))
            
            st.success(f"✅ Models trained successfully!")
            st.info(f"Win Accuracy: {win_accuracy:.1%} | Score MAE: {score_mae:.1f} | Spread MAE: {spread_mae:.1f} | Total MAE: {total_mae:.1f}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            return False
    
    def get_team_stats_before_date(self, advanced_stats, team, date):
        """Get team stats before a specific date"""
        if team not in advanced_stats:
            return None
        
        previous_dates = [d for d in advanced_stats[team].keys() if d < date]
        if not previous_dates:
            return None
        
        latest_date = max(previous_dates)
        return advanced_stats[team][latest_date]
    
    # ... (rest of the class methods like load_model, get_team_abbreviation, load_data, get_game_odds remain similar)
    
    def predict_game_advanced(self, home_team, away_team):
        """Make advanced predictions using all trained models"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None, None, None, None
        
        # Create feature vector for current matchup
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # Get pace stats (you'll need to load these from your pbp data)
        home_pace = {'pace_factor': 0.5, 'plays_per_game': 65}  # Defaults
        away_pace = {'pace_factor': 0.5, 'plays_per_game': 65}
        
        feature_vector = [
            # Home team stats
            home_stats['win_pct'], home_stats['points_for_avg'], home_stats['points_against_avg'],
            home_stats.get('point_differential_avg', 0), home_stats.get('offensive_efficiency', 1.0), 
            home_stats.get('defensive_efficiency', 1.0), home_pace['pace_factor'], home_pace['plays_per_game'],
            
            # Away team stats  
            away_stats['win_pct'], away_stats['points_for_avg'], away_stats['points_against_avg'],
            away_stats.get('point_differential_avg', 0), away_stats.get('offensive_efficiency', 1.0),
            away_stats.get('defensive_efficiency', 1.0), away_pace['pace_factor'], away_pace['plays_per_game'],
            
            # Interaction features
            home_stats['points_for_avg'] - away_stats['points_against_avg'],
            away_stats['points_for_avg'] - home_stats['points_against_avg'],
            1  # Home field advantage
        ]
        
        feature_vector = np.array([feature_vector])
        feature_vector = np.nan_to_num(feature_vector)
        
        # Scale features
        feature_vector_win = self.scalers['win'].transform(feature_vector)
        feature_vector_score = self.scalers['score'].transform(feature_vector)
        feature_vector_spread = self.scalers['spread'].transform(feature_vector)
        feature_vector_total = self.scalers['total'].transform(feature_vector)
        
        # Make predictions
        win_prob = self.win_model.predict_proba(feature_vector_win)[0][1]
        scores = self.score_model.predict(feature_vector_score)[0]
        home_score, away_score = scores[0], scores[1]
        spread = self.spread_model.predict(feature_vector_spread)[0]
        total = self.total_model.predict(feature_vector_total)[0]
        
        return win_prob, home_score, away_score, spread, total

# ... (main function would be updated to use the new prediction method)
