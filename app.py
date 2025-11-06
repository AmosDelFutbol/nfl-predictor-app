# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="NFL Week 10 Predictions",
    page_icon="🏈",
    layout="wide"
)

class NFLPredictor:
    def __init__(self):
        self.model = None
        self.schedule = None
        self.odds = None
        self.team_stats = {}
        
    def train_model(self):
        """Train the model if it doesn't exist"""
        st.info("🔄 Training NFL prediction model...")
        
        try:
            # Load historical data
            with open('spreadspoke_scores.json', 'r') as f:
                games = pd.DataFrame(json.load(f))
            
            # Basic data cleaning
            games = games[games['score_home'].notna() & games['score_away'].notna()]
            games['schedule_date'] = pd.to_datetime(games['schedule_date'])
            games = games[games['schedule_date'].dt.year >= 2020]
            games['home_win'] = (games['score_home'] > games['score_away']).astype(int)
            
            # Calculate team stats
            all_teams = list(set(games['team_home'].unique()) | set(games['team_away'].unique()))
            team_stats = {}
            
            for team in all_teams:
                team_games = games[(games['team_home'] == team) | (games['team_away'] == team)].copy()
                team_games = team_games.sort_values('schedule_date')
                team_games['is_home'] = (team_games['team_home'] == team).astype(int)
                team_games['team_score'] = np.where(team_games['team_home'] == team, 
                                                   team_games['score_home'], 
                                                   team_games['score_away'])
                team_games['opponent_score'] = np.where(team_games['team_home'] == team, 
                                                      team_games['score_away'], 
                                                      team_games['score_away'])
                team_games['win'] = (team_games['team_score'] > team_games['opponent_score']).astype(int)
                team_games['win_pct'] = team_games['win'].rolling(8, min_periods=1).mean()
                team_games['points_for_avg'] = team_games['team_score'].rolling(8, min_periods=1).mean()
                team_games['points_against_avg'] = team_games['opponent_score'].rolling(8, min_periods=1).mean()
                
                for _, row in team_games.iterrows():
                    date = row['schedule_date']
                    if team not in team_stats:
                        team_stats[team] = {}
                    team_stats[team][date] = {
                        'win_pct': row['win_pct'],
                        'points_for_avg': row['points_for_avg'],
                        'points_against_avg': row['points_against_avg']
                    }
            
            # Create features
            features = []
            targets = []
            
            for _, game in games.iterrows():
                home_team = game['team_home']
                away_team = game['team_away']
                game_date = game['schedule_date']
                
                home_stats = None
                away_stats = None
                
                if home_team in team_stats:
                    previous_dates = [d for d in team_stats[home_team].keys() if d < game_date]
                    if previous_dates:
                        latest_date = max(previous_dates)
                        home_stats = team_stats[home_team][latest_date]
                
                if away_team in team_stats:
                    previous_dates = [d for d in team_stats[away_team].keys() if d < game_date]
                    if previous_dates:
                        latest_date = max(previous_dates)
                        away_stats = team_stats[away_team][latest_date]
                
                if home_stats and away_stats:
                    feature_vector = [
                        home_stats['win_pct'],
                        home_stats['points_for_avg'], 
                        home_stats['points_against_avg'],
                        away_stats['win_pct'],
                        away_stats['points_for_avg'],
                        away_stats['points_against_avg'],
                        1
                    ]
                    features.append(feature_vector)
                    targets.append(game['home_win'])
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            X = np.nan_to_num(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            self.model.fit(X_train, y_train)
            
            accuracy = self.model.score(X_test, y_test)
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.1%}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            return False
    
    def load_model(self):
        """Try to load existing model, otherwise train new one"""
        try:
            with open('nfl_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            st.success("✅ Pre-trained model loaded!")
            return True
        except:
            return self.train_model()
    
    def get_team_abbreviation(self, full_name):
        """Convert full team name to abbreviation"""
        team_mapping = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        return team_mapping.get(full_name, full_name)
    
    def load_data(self):
        """Load schedule and odds data"""
        try:
            # Load schedule
            with open('week_10_schedule.json', 'r') as f:
                schedule_data = json.load(f)
                self.schedule = pd.DataFrame(schedule_data['Week 10'])
                st.info(f"📅 Loaded schedule with {len(self.schedule)} games")
            
            # Load odds
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
                self.odds = pd.DataFrame(odds_data)
                st.info(f"🎰 Loaded odds for {len(self.odds)} games")
            
            # Create default team stats based on common knowledge
            default_stats = {
                'KC': [0.75, 28.5, 19.2], 'BUF': [0.65, 26.8, 21.1], 'SF': [0.80, 30.1, 18.5],
                'PHI': [0.70, 27.3, 20.8], 'DAL': [0.68, 26.9, 21.3], 'BAL': [0.72, 27.8, 19.8],
                'MIA': [0.66, 29.2, 23.1], 'CIN': [0.62, 25.7, 22.4], 'GB': [0.58, 24.3, 23.7],
                'DET': [0.64, 26.1, 22.9], 'LAR': [0.59, 25.8, 23.5], 'SEA': [0.55, 24.2, 24.8],
                'LV': [0.45, 21.8, 25.9], 'DEN': [0.52, 23.1, 24.2], 'LAC': [0.57, 25.3, 24.1],
                'NE': [0.35, 18.9, 27.3], 'NYJ': [0.42, 20.5, 26.1], 'CHI': [0.48, 22.7, 25.3],
                'MIN': [0.53, 24.8, 23.9], 'NO': [0.51, 23.5, 24.4], 'ATL': [0.49, 22.9, 24.7],
                'CAR': [0.30, 17.8, 28.5], 'JAX': [0.56, 24.6, 23.8], 'IND': [0.54, 24.1, 24.0],
                'HOU': [0.50, 23.3, 24.5], 'TEN': [0.47, 22.4, 25.1], 'CLE': [0.61, 25.2, 22.6],
                'PIT': [0.58, 23.9, 23.4], 'NYG': [0.40, 19.8, 26.8], 'WAS': [0.43, 21.2, 26.3],
                'ARI': [0.46, 22.1, 25.6], 'TB': [0.55, 24.5, 24.2]
            }
            self.team_stats = default_stats
            st.success(f"📊 Loaded stats for {len(self.team_stats)} teams")
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return False
    
    def find_matching_odds(self, home_team, away_team, home_full, away_full):
        """Find matching odds for a game"""
        if self.odds is None or len(self.odds) == 0:
            return None
            
        # Show first few odds rows to help debug
        if not hasattr(self, 'odds_shown'):
            st.info(f"First 3 odds rows: {self.odds.head(3).to_dict('records')}")
            self.odds_shown = True
            
        for _, odds_row in self.odds.iterrows():
            # Try to match by team names in various fields
            for home_field in ['home_team', 'home', 'team_home']:
                for away_field in ['away_team', 'away', 'team_away']:
                    if home_field in odds_row and away_field in odds_row:
                        odds_home = str(odds_row[home_field]).upper()
                        odds_away = str(odds_row[away_field]).upper()
                        
                        if (home_team in odds_home or away_team in odds_away or
                            home_full.upper() in odds_home or away_full.upper() in odds_away):
                            return odds_row
        return None
    
    def predict_game(self, home_team, away_team):
        """Predict game outcome using the trained model"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
            
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        features = np.array([[
            home_stats[0], home_stats[1], home_stats[2],
            away_stats[0], away_stats[1], away_stats[2],
            1
        ]])
        
        try:
            probabilities = self.model.predict_proba(features)[0]
            home_win_prob = probabilities[1]
            return home_win_prob
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def predict_game_score(self, home_team, away_team, home_win_prob):
        """Predict the actual score of the game"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None, None
            
        home_offense = self.team_stats[home_team][1]
        away_offense = self.team_stats[away_team][1]
        
        # Base scores with home field adjustment
        home_base = home_offense + (home_win_prob - 0.5) * 7
        away_base = away_offense - (home_win_prob - 0.5) * 7
        
        home_score = max(10, round(home_base))
        away_score = max(10, round(away_base))
        
        return home_score, away_score
    
    def convert_prob_to_spread(self, home_win_prob):
        """Convert win probability to point spread"""
        if home_win_prob is None:
            return 0
            
        if home_win_prob >= 0.5:
            spread = -((home_win_prob - 0.5) * 14)
        else:
            spread = ((0.5 - home_win_prob) * 14)
        
        return round(spread * 2) / 2
    
    def predict_total_points(self, home_team, away_team):
        """Predict total points based on team stats"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return 45.0
            
        home_offense = self.team_stats[home_team][1]
        away_offense = self.team_stats[away_team][1]
        
        total = (home_offense + away_offense) * 1.1
        return round(total * 2) / 2

def main():
    st.title("🏈 NFL Week 10 Predictions & Betting Analysis")
    st.markdown("### Complete Game Analysis: Vegas vs Model Projections")
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Load model (will train if not exists)
    if not predictor.load_model():
        st.stop()
        
    # Load data
    if not predictor.load_data():
        st.error("Failed to load data files")
        st.stop()
    
    # Display predictions for each game
    st.header("🎯 Week 10 Game Predictions")
    
    for _, game in predictor.schedule.iterrows():
        home_full = game['home']
        away_full = game['away']
        game_date = game['date']
        
        home_team = predictor.get_team_abbreviation(home_full)
        away_team = predictor.get_team_abbreviation(away_full)
        
        game_odds = predictor.find_matching_odds(home_team, away_team, home_full, away_full)
        home_win_prob = predictor.predict_game(home_team, away_team)
        
        if home_win_prob is None:
            continue
        
        model_spread = predictor.convert_prob_to_spread(home_win_prob)
        model_total = predictor.predict_total_points(home_team, away_team)
        home_score, away_score = predictor.predict_game_score(home_team, away_team, home_win_prob)
        
        # Create display
        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
        
        with col1:
            st.subheader(f"{away_full} @ {home_full}")
            st.caption(f"Date: {game_date} | {away_team} @ {home_team}")
            
            # Vegas Odds
            st.markdown("**🎰 Vegas Odds**")
            if game_odds is not None:
                # Try to extract odds from various field names
                spread = game_odds.get('spread', game_odds.get('point_spread', 0))
                total = game_odds.get('over_under', game_odds.get('total', 45.0))
                home_ml = game_odds.get('home_moneyline', game_odds.get('home_ml', 0))
                away_ml = game_odds.get('away_moneyline', game_odds.get('away_ml', 0))
                
                if spread < 0:
                    st.write(f"Spread: **{home_team} {spread}**")
                else:
                    st.write(f"Spread: **{away_team} +{spread}**")
                
                st.write(f"Total: **{total}**")
                st.write(f"Moneyline: {home_team} {home_ml} | {away_team} {away_ml}")
            else:
                st.write("Vegas odds: Not found")
                spread, total = 0, 45.0
        
        with col2:
            st.markdown("**🤖 Model Projections**")
            st.metric("Win Probability", f"{home_win_prob:.1%}")
            
            winner = home_team if home_win_prob > 0.5 else away_team
            st.metric("Predicted Winner", winner)
            
            if home_win_prob > 0.5:
                st.metric("Projected Spread", f"{home_team} -{abs(model_spread)}")
            else:
                st.metric("Projected Spread", f"{away_team} +{model_spread}")
            
            st.metric("Projected Total", f"{model_total}")
            
            if home_score and away_score:
                st.metric("Projected Score", f"{home_team} {home_score}-{away_score} {away_team}")
        
        with col3:
            st.markdown("**🏆 Final Picks**")
            
            if game_odds is not None:
                # Against Spread
                spread_diff = abs(abs(model_spread) - abs(spread))
                if spread_diff > 1.0:
                    if abs(model_spread) < abs(spread):
                        st.success(f"**ATS:** {home_team if home_win_prob > 0.5 else away_team}")
                    else:
                        st.success(f"**ATS:** {away_team if home_win_prob > 0.5 else home_team}")
                else:
                    st.info("**ATS:** No strong play")
                
                # Over/Under
                total_diff = abs(model_total - total)
                if total_diff > 2.0:
                    if model_total > total:
                        st.success(f"**Total:** OVER {total}")
                    else:
                        st.success(f"**Total:** UNDER {total}")
                else:
                    st.info("**Total:** No strong play")
            else:
                st.info("Odds needed for picks")
        
        with col4:
            st.markdown("**💡 Betting Tips**")
            if game_odds is not None and home_score and away_score:
                tips = []
                if home_win_prob > 0.65:
                    tips.append("💰 Strong on favorite")
                elif home_win_prob < 0.35:
                    tips.append("💰 Strong on underdog")
                
                if abs(model_spread - spread) > 2:
                    tips.append("📈 Good spread value")
                
                if abs(model_total - total) > 3:
                    tips.append("🔥 Strong total play")
                
                for tip in tips if tips else ["⚖️ No strong edge"]:
                    st.success(tip) if tip != "⚖️ No strong edge" else st.info(tip)
        
        st.markdown("---")

if __name__ == "__main__":
    main()
