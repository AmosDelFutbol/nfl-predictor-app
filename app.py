# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime

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
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open('nfl_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            st.error("Model file 'nfl_model.pkl' not found. Please run training first.")
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
                st.info(f"Loaded schedule with {len(self.schedule)} games")
            
            # Load odds - show the structure to debug
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
                self.odds = pd.DataFrame(odds_data)
                st.info(f"Loaded odds for {len(self.odds)} games")
                st.info(f"Odds columns: {list(self.odds.columns)}")
                if len(self.odds) > 0:
                    st.info(f"First odds row: {self.odds.iloc[0].to_dict()}")
            
            # Load team stats
            with open('2025_NFL_OFFENSE.json', 'r') as f:
                offense_data = json.load(f)
                offense_df = pd.DataFrame(offense_data)
                
                for _, team in offense_df.iterrows():
                    team_name = team['Team']
                    team_abbr = self.get_team_abbreviation(team_name)
                    
                    points_per_game = team.get('POINTS PER GAME', 23.0)
                    win_pct = 0.5  # Default for now
                    points_against = 23.0  # Default for now
                    
                    self.team_stats[team_abbr] = [win_pct, points_per_game, points_against]
                
                st.success(f"Loaded stats for {len(self.team_stats)} teams")
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def find_matching_odds(self, home_team, away_team, home_full, away_full):
        """Find matching odds for a game"""
        if self.odds is None or len(self.odds) == 0:
            return None
            
        # Try different matching strategies
        for _, odds_row in self.odds.iterrows():
            # Check all possible column combinations
            odds_home = odds_row.get('home_team', '')
            odds_away = odds_row.get('away_team', '')
            odds_home_abbr = odds_row.get('home_abbr', '')
            odds_away_abbr = odds_row.get('away_abbr', '')
            
            # Match by abbreviation
            if (odds_home_abbr == home_team and odds_away_abbr == away_team):
                return odds_row
            # Match by full name
            if (odds_home in home_full or home_full in odds_home) and (odds_away in away_full or away_full in odds_away):
                return odds_row
            # Match by abbreviation in full name fields
            if (home_team in odds_home or away_team in odds_away):
                return odds_row
                
        return None
    
    def predict_game_score(self, home_team, away_team, home_win_prob):
        """Predict the actual score of the game"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None, None
            
        home_offense = self.team_stats[home_team][1]
        away_offense = self.team_stats[away_team][1]
        home_defense = self.team_stats[home_team][2]
        away_defense = self.team_stats[away_team][2]
        
        # Base scores from team averages
        home_base = (home_offense + away_defense) / 2
        away_base = (away_offense + home_defense) / 2
        
        # Adjust based on win probability
        prob_adjustment = (home_win_prob - 0.5) * 7  # Adjust by up to 7 points
        
        home_score = max(0, round(home_base + prob_adjustment))
        away_score = max(0, round(away_base - prob_adjustment))
        
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
    
    # Load data
    if not predictor.load_model():
        st.stop()
        
    if not predictor.load_data():
        st.error("Failed to load data files")
        st.stop()
    
    st.success("✅ Model and data loaded successfully!")
    
    # Display predictions for each game
    st.header("🎯 Week 10 Game Predictions")
    
    for _, game in predictor.schedule.iterrows():
        home_full = game['home']
        away_full = game['away']
        game_date = game['date']
        
        # Convert to abbreviations
        home_team = predictor.get_team_abbreviation(home_full)
        away_team = predictor.get_team_abbreviation(away_full)
        
        # Find matching odds
        game_odds = predictor.find_matching_odds(home_team, away_team, home_full, away_full)
        
        # Make prediction
        home_win_prob = predictor.predict_game(home_team, away_team)
        if home_win_prob is None:
            continue
        
        # Calculate all projections
        model_spread = predictor.convert_prob_to_spread(home_win_prob)
        model_total = predictor.predict_total_points(home_team, away_team)
        home_score, away_score = predictor.predict_game_score(home_team, away_team, home_win_prob)
        
        # Determine favorites
        if home_win_prob > 0.5:
            model_winner = home_team
            model_loser = away_team
            model_fav_spread = abs(model_spread)
        else:
            model_winner = away_team
            model_loser = home_team
            model_fav_spread = model_spread
        
        # Create columns for display
        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
        
        with col1:
            st.subheader(f"{away_full} @ {home_full}")
            st.caption(f"Date: {game_date}")
            
            # Vegas Odds Section
            st.markdown("**🎰 Vegas Odds**")
            if game_odds is not None:
                vegas_spread = game_odds.get('spread', 0)
                vegas_total = game_odds.get('over_under', 45.0)
                home_ml = game_odds.get('home_moneyline', 0)
                away_ml = game_odds.get('away_moneyline', 0)
                
                if vegas_spread < 0:
                    vegas_favorite = home_team
                    vegas_underdog = away_team
                    vegas_fav_spread = abs(vegas_spread)
                    st.write(f"Spread: **{vegas_favorite} -{vegas_fav_spread}**")
                else:
                    vegas_favorite = away_team
                    vegas_underdog = home_team
                    vegas_fav_spread = vegas_spread
                    st.write(f"Spread: **{vegas_underdog} +{vegas_fav_spread}**")
                
                st.write(f"Total: **{vegas_total}**")
                st.write(f"Moneyline: {home_team} {home_ml} | {away_team} {away_ml}")
            else:
                st.write("Vegas odds: Not available")
                vegas_spread = 0
                vegas_total = 45.0
        
        with col2:
            st.markdown("**🤖 Model Projections**")
            st.metric("Win Probability", f"{home_win_prob:.1%}")
            
            if home_win_prob > 0.5:
                st.metric("Projected Spread", f"{home_team} -{abs(model_spread)}")
            else:
                st.metric("Projected Spread", f"{away_team} +{model_spread}")
            
            st.metric("Projected Total", f"{model_total}")
            
            if home_score is not None and away_score is not None:
                st.metric("Projected Score", f"{home_team} {home_score}-{away_score} {away_team}")
        
        with col3:
            st.markdown("**🏆 Final Predictions**")
            
            # Winner prediction
            st.write(f"**Winner:** {model_winner}")
            
            # Against Spread prediction
            if game_odds is not None:
                spread_diff = abs(model_fav_spread - abs(vegas_spread))
                if spread_diff > 1.0:
                    if model_fav_spread < abs(vegas_spread):
                        st.success(f"**ATS:** {model_winner} -{model_fav_spread}")
                    else:
                        st.success(f"**ATS:** {model_loser} +{abs(vegas_spread)}")
                else:
                    st.info("**ATS:** No strong play")
                
                # Over/Under prediction
                total_diff = abs(model_total - vegas_total)
                if total_diff > 2.0:
                    if model_total > vegas_total:
                        st.success(f"**Total:** OVER {vegas_total}")
                    else:
                        st.success(f"**Total:** UNDER {vegas_total}")
                else:
                    st.info("**Total:** No strong play")
            else:
                st.info("Odds needed for ATS/Total predictions")
        
        with col4:
            st.markdown("**💡 Betting Tips**")
            
            if game_odds is not None:
                tips = []
                
                # Spread analysis
                spread_diff = abs(model_fav_spread - abs(vegas_spread))
                if spread_diff > 2.0:
                    if model_fav_spread < abs(vegas_spread):
                        tips.append("📈 Strong bet on favorite")
                    else:
                        tips.append("📈 Strong bet on underdog")
                elif spread_diff > 1.0:
                    tips.append("📊 Lean on spread")
                
                # Total analysis
                total_diff = abs(model_total - vegas_total)
                if total_diff > 3.0:
                    if model_total > vegas_total:
                        tips.append("🔥 Strong OVER play")
                    else:
                        tips.append("🔥 Strong UNDER play")
                elif total_diff > 2.0:
                    tips.append("📊 Lean on total")
                
                # Moneyline value
                if home_win_prob > 0.7 and home_ml > -200:
                    tips.append("💰 ML value on home")
                elif home_win_prob < 0.3 and away_ml > -200:
                    tips.append("💰 ML value on away")
                
                if tips:
                    for tip in tips:
                        st.success(tip)
                else:
                    st.info("No strong bets")
            else:
                st.info("Odds not available")
        
        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.info("""
    **How to use this analysis:**
    - **Vegas Odds**: What the sportsbooks are offering
    - **Model Projections**: What our AI model predicts based on team performance
    - **Final Predictions**: Our recommended bets when model disagrees with Vegas
    - **Betting Tips**: Quick summary of the best opportunities
    """)

if __name__ == "__main__":
    main()
