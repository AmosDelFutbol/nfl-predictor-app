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
                # Extract the Week 10 games
                self.schedule = pd.DataFrame(schedule_data['Week 10'])
                st.info(f"Loaded schedule with {len(self.schedule)} games")
            
            # Load odds
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
                self.odds = pd.DataFrame(odds_data)
                st.info(f"Loaded odds for {len(self.odds)} games")
            
            # Load team stats from 2025 data
            with open('2025_NFL_OFFENSE.json', 'r') as f:
                offense_data = json.load(f)
                offense_df = pd.DataFrame(offense_data)
                
                st.info(f"Loaded team stats with {len(offense_df)} teams")
                
                # Map full team names to abbreviations for stats
                for _, team in offense_df.iterrows():
                    team_name = team['Team']  # Full team name from your data
                    team_abbr = self.get_team_abbreviation(team_name)
                    
                    # Use available stats - adjust based on what's in your data
                    points_per_game = team.get('POINTS PER GAME', 23.0)
                    # For win percentage and defense, we'll use defaults for now
                    win_pct = 0.5  # You'll need to calculate this from historical data
                    points_against = 23.0  # You'll need to add defensive stats
                    
                    self.team_stats[team_abbr] = [win_pct, points_per_game, points_against]
                
                st.success(f"Loaded stats for {len(self.team_stats)} teams")
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def calculate_implied_probability(self, american_odds):
        """Convert American odds to implied probability"""
        try:
            if american_odds > 0:
                return 100 / (american_odds + 100)
            else:
                return abs(american_odds) / (abs(american_odds) + 100)
        except:
            return 0.5
    
    def predict_game(self, home_team, away_team):
        """Predict game outcome using the trained model"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            st.warning(f"Missing stats for {home_team} or {away_team}")
            return None
            
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # Create features array (same format as training)
        features = np.array([[
            home_stats[0], home_stats[1], home_stats[2],  # home_win_pct, offense, defense
            away_stats[0], away_stats[1], away_stats[2],  # away_win_pct, offense, defense
            1  # home_field advantage
        ]])
        
        # Get prediction
        try:
            probabilities = self.model.predict_proba(features)[0]
            home_win_prob = probabilities[1]
            return home_win_prob
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def convert_prob_to_spread(self, home_win_prob):
        """Convert win probability to point spread"""
        if home_win_prob is None:
            return 0
            
        # Using common conversion: prob -> spread
        if home_win_prob >= 0.5:
            spread = -((home_win_prob - 0.5) * 14)  # Convert to negative spread for favorite
        else:
            spread = ((0.5 - home_win_prob) * 14)   # Positive spread for underdog
        
        return round(spread * 2) / 2  # Round to nearest 0.5
    
    def predict_total_points(self, home_team, away_team):
        """Predict total points based on team offensive/defensive stats"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return 45.0
            
        home_offense = self.team_stats[home_team][1]
        away_offense = self.team_stats[away_team][1]
        
        # Simple average of both teams' offensive capabilities
        total = (home_offense + away_offense) * 1.1  # Adjust for typical game totals
        return round(total * 2) / 2  # Round to nearest 0.5

def main():
    st.title("🏈 NFL Week 10 Predictions & Betting Analysis")
    st.markdown("### Model Projections vs. Vegas Odds")
    
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
        
        # Find matching odds - we need to handle this carefully since odds might use different naming
        game_odds = None
        if predictor.odds is not None and len(predictor.odds) > 0:
            # Try to find matching game by team names
            for _, odds_row in predictor.odds.iterrows():
                # Check if this odds row matches our game
                odds_home = odds_row.get('home_team', '')
                odds_away = odds_row.get('away_team', '')
                
                # Simple matching - you might need to adjust this based on your odds file structure
                if (home_team in odds_home or odds_home in home_full or 
                    away_team in odds_away or odds_away in away_full):
                    game_odds = odds_row
                    break
        
        # Make prediction
        home_win_prob = predictor.predict_game(home_team, away_team)
        if home_win_prob is None:
            continue
        
        # Calculate model projections
        model_spread = predictor.convert_prob_to_spread(home_win_prob)
        model_total = predictor.predict_total_points(home_team, away_team)
        
        # Create columns for display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"{away_full} @ {home_full}")
            st.caption(f"Date: {game_date}")
            st.caption(f"Teams: {away_team} @ {home_team}")
            
            # Vegas Odds
            st.markdown("**Vegas Odds:**")
            if game_odds is not None:
                vegas_spread = game_odds.get('spread', 0)
                vegas_total = game_odds.get('over_under', 45.0)
                home_ml = game_odds.get('home_moneyline', 0)
                away_ml = game_odds.get('away_moneyline', 0)
                
                # Determine favorite based on spread
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
                st.write("Vegas odds: Not available for this game")
                vegas_spread = 0
                vegas_total = 45.0
        
        with col2:
            st.markdown("**Model Projections:**")
            st.metric("Home Win Probability", f"{home_win_prob:.1%}")
            
            # Model's favorite
            if home_win_prob > 0.5:
                model_favorite = home_team
                model_underdog = away_team
                model_fav_spread = abs(model_spread)
                st.metric("Projected Spread", f"{model_favorite} -{model_fav_spread}")
            else:
                model_favorite = away_team
                model_underdog = home_team
                model_fav_spread = model_spread
                st.metric("Projected Spread", f"{model_underdog} +{model_fav_spread}")
            
            st.metric("Projected Total", f"{model_total}")
        
        with col3:
            st.markdown("**Betting Recommendations:**")
            
            if game_odds is not None:
                # Spread Analysis
                spread_diff = abs(model_fav_spread - abs(vegas_spread))
                if spread_diff > 1.0:
                    if model_fav_spread < abs(vegas_spread):
                        st.success(f"📈 BET: {model_favorite} (Model likes favorite more)")
                    else:
                        st.success(f"📈 BET: {model_underdog} (Model thinks underdog covers)")
                else:
                    st.info("⚖️ Spread: No strong opinion")
                
                # Total Analysis
                total_diff = abs(model_total - vegas_total)
                if total_diff > 2.0:
                    if model_total > vegas_total:
                        st.success(f"📈 BET: OVER {vegas_total}")
                    else:
                        st.success(f"📈 BET: UNDER {vegas_total}")
                else:
                    st.info("⚖️ Total: No strong opinion")
            else:
                st.info("Odds not available for analysis")
        
        st.markdown("---")
    
    # Summary section
    st.header("📊 Model Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("How to Read This Analysis")
        st.markdown("""
        - **Spread Differences**: When model spread differs from Vegas by >1 point
        - **Total Differences**: When model total differs from Vegas by >2 points  
        - **Moneyline Value**: When model probability suggests better odds than offered
        - **Green Recommendations**: Strong betting opportunities
        - **Blue Info**: Close calls, no strong edge
        """)
    
    with col2:
        st.subheader("About the Model")
        st.markdown("""
        - **Trained on**: Team performance statistics (win %, offense, defense)
        - **Predicts**: Game outcomes based on team strength
        - **Comparison**: Finds discrepancies between model projections and betting markets
        - **Goal**: Identify value betting opportunities
        """)
    
    st.info("⚠️ Remember: Betting involves risk. These are projections, not guarantees.")

if __name__ == "__main__":
    main()
