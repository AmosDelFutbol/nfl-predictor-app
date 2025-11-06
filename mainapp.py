# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

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
        self.odds_data = None
        self.team_stats = {}
        
    def load_model(self):
        """Load or train a simple model"""
        try:
            with open('nfl_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            st.success("✅ Pre-trained model loaded!")
            return True
        except:
            return self.train_simple_model()
    
    def train_simple_model(self):
        """Train a simple model quickly"""
        st.info("🔄 Training simple NFL model...")
        
        try:
            with open('spreadspoke_scores.json', 'r') as f:
                games = pd.DataFrame(json.load(f))
            
            # Simple data cleaning
            games = games[games['score_home'].notna() & games['score_away'].notna()]
            games['schedule_date'] = pd.to_datetime(games['schedule_date'])
            games = games[games['schedule_date'].dt.year >= 2020]
            games['home_win'] = (games['score_home'] > games['score_away']).astype(int)
            
            # Simple team stats
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
            st.success(f"✅ Simple model trained! Accuracy: {accuracy:.1%}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            return False
    
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
                self.odds_data = pd.DataFrame(odds_data)
                st.info(f"🎰 Loaded {len(self.odds_data)} odds entries")
            
            # Create default team stats
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
    
    def get_game_odds(self, home_team, away_team, home_full, away_full):
        """Aggregate all odds for a specific game"""
        if self.odds_data is None or len(self.odds_data) == 0:
            return None
        
        # Find all rows for this game
        game_odds_rows = self.odds_data[
            (self.odds_data['home_team'] == home_full) & 
            (self.odds_data['away_team'] == away_full)
        ]
        
        if len(game_odds_rows) == 0:
            return None
        
        # Aggregate odds into a single dictionary
        odds = {
            'home_team': home_full,
            'away_team': away_full,
            'home_moneyline': None,
            'away_moneyline': None,
            'spread': None,
            'spread_odds': None,
            'total': None,
            'over_odds': None,
            'under_odds': None
        }
        
        for _, row in game_odds_rows.iterrows():
            market = row.get('market', '')
            label = row.get('label', '')
            price = row.get('price', 0)
            point = row.get('point', None)
            
            if market == 'h2h':
                if label == home_full:
                    odds['home_moneyline'] = price
                elif label == away_full:
                    odds['away_moneyline'] = price
            
            elif market == 'spreads':
                if label == home_full:
                    odds['spread'] = point
                    odds['spread_odds'] = price
            
            elif market == 'totals':
                odds['total'] = point
                if label == 'Over':
                    odds['over_odds'] = price
                elif label == 'Under':
                    odds['under_odds'] = price
        
        return odds
    
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
        home_defense = self.team_stats[home_team][2]
        away_defense = self.team_stats[away_team][2]
        
        # Better score prediction considering both offense and defense
        home_score = (home_offense + away_defense) / 2 + (home_win_prob - 0.5) * 7
        away_score = (away_offense + home_defense) / 2 - (home_win_prob - 0.5) * 7
        
        home_score = max(10, round(home_score))
        away_score = max(10, round(away_score))
        
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
        home_defense = self.team_stats[home_team][2]
        away_defense = self.team_stats[away_team][2]
        
        # Better total prediction considering both teams
        total = (home_offense + away_offense + home_defense + away_defense) / 2
        return round(total * 2) / 2

def main():
    st.title("🏈 NFL Week 10 Predictions & Betting Analysis")
    st.markdown("### Model Projections vs Vegas Odds Comparison")
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Load model (will train if not exists)
    if not predictor.load_model():
        st.error("Failed to load model")
        st.stop()
        
    # Load data
    if not predictor.load_data():
        st.error("Failed to load data files")
        st.stop()
    
    st.success("✅ Ready to make predictions!")
    
    # Display predictions for each game
    st.header("🎯 Week 10 Game Predictions")
    
    if predictor.schedule is None or len(predictor.schedule) == 0:
        st.error("No schedule data loaded")
        return
    
    for _, game in predictor.schedule.iterrows():
        home_full = game['home']
        away_full = game['away']
        game_date = game['date']
        
        home_team = predictor.get_team_abbreviation(home_full)
        away_team = predictor.get_team_abbreviation(away_full)
        
        # Get aggregated odds for this game
        game_odds = predictor.get_game_odds(home_team, away_team, home_full, away_full)
        
        # Make model projections
        home_win_prob = predictor.predict_game(home_team, away_team)
        
        if home_win_prob is None:
            st.warning(f"Could not predict {home_team} vs {away_team}")
            continue
        
        # Model projections
        model_spread = predictor.convert_prob_to_spread(home_win_prob)
        model_total = predictor.predict_total_points(home_team, away_team)
        home_score, away_score = predictor.predict_game_score(home_team, away_team, home_win_prob)
        
        # Determine model predictions
        model_winner = home_team if home_win_prob > 0.5 else away_team
        model_loser = away_team if home_win_prob > 0.5 else home_team
        
        # Create display
        col1, col2, col3 = st.columns([2, 1.5, 1.5])
        
        with col1:
            st.subheader(f"{away_full} @ {home_full}")
            st.caption(f"Date: {game_date} | {away_team} @ {home_team}")
            
            # Vegas Odds
            st.markdown("**🎰 Vegas Odds**")
            if game_odds is not None:
                if game_odds['spread'] is not None:
                    if game_odds['spread'] < 0:
                        st.write(f"Spread: **{home_team} {game_odds['spread']}** ({game_odds['spread_odds']})")
                    else:
                        st.write(f"Spread: **{away_team} +{game_odds['spread']}** ({game_odds['spread_odds']})")
                
                if game_odds['total'] is not None:
                    st.write(f"Total: **{game_odds['total']}**")
                    if game_odds['over_odds'] and game_odds['under_odds']:
                        st.write(f"Over: {game_odds['over_odds']} | Under: {game_odds['under_odds']}")
                
                if game_odds['home_moneyline'] is not None and game_odds['away_moneyline'] is not None:
                    st.write(f"Moneyline: {home_team} {game_odds['home_moneyline']} | {away_team} {game_odds['away_moneyline']}")
            else:
                st.write("Vegas odds: Not available")
        
        with col2:
            st.markdown("**🤖 Model Projections**")
            st.metric("Win Probability", f"{home_win_prob:.1%}")
            st.metric("Predicted Winner", model_winner)
            
            if home_win_prob > 0.5:
                st.metric("Projected Spread", f"{home_team} -{abs(model_spread):.1f}")
            else:
                st.metric("Projected Spread", f"{away_team} +{abs(model_spread):.1f}")
            
            st.metric("Projected Total", f"{model_total:.1f}")
            
            if home_score and away_score:
                st.metric("Projected Score", f"{home_team} {home_score}-{away_score} {away_team}")
        
        with col3:
            st.markdown("**🏆 Final Picks**")
            
            if game_odds is not None and game_odds['spread'] is not None:
                # FIXED: Against Spread Analysis
                vegas_spread = game_odds['spread']
                
                # Determine which team is favored by Vegas
                if vegas_spread < 0:  # Home team is favorite
                    favorite = home_team
                    underdog = away_team
                    if model_spread <= vegas_spread:  # Model thinks favorite wins by MORE than Vegas expects
                        ats_pick = favorite
                        reasoning = f"Model thinks {favorite} wins by more than Vegas expects"
                    else:  # Model thinks favorite wins by LESS than Vegas expects
                        ats_pick = underdog
                        reasoning = f"Model thinks {underdog} covers the spread"
                else:  # Away team is favorite
                    favorite = away_team
                    underdog = home_team
                    if model_spread >= vegas_spread:  # Model thinks favorite wins by MORE than Vegas expects
                        ats_pick = favorite
                        reasoning = f"Model thinks {favorite} wins by more than Vegas expects"
                    else:  # Model thinks favorite wins by LESS than Vegas expects
                        ats_pick = underdog
                        reasoning = f"Model thinks {underdog} covers the spread"
                
                st.success(f"**ATS Pick:** {ats_pick}")
                st.write(reasoning)
                
                # Over/Under Analysis (this was correct)
                if game_odds['total'] is not None:
                    if model_total > game_odds['total']:
                        st.success(f"**Total Pick:** OVER {game_odds['total']}")
                        st.write(f"Model projects more points than Vegas line")
                    else:
                        st.success(f"**Total Pick:** UNDER {game_odds['total']}")
                        st.write(f"Model projects fewer points than Vegas line")
            else:
                st.info("Vegas odds needed for picks")
        
        st.markdown("---")

if __name__ == "__main__":
    main()
