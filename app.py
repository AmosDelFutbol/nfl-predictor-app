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
    
    def load_data(self):
        """Load schedule and odds data"""
        try:
            # Load schedule
            with open('week_10_schedule.json', 'r') as f:
                schedule_data = json.load(f)
                self.schedule = pd.DataFrame(schedule_data)
                st.info(f"Loaded schedule with {len(self.schedule)} games")
            
            # Load odds
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
                self.odds = pd.DataFrame(odds_data)
                st.info(f"Loaded odds for {len(self.odds)} games")
            
            # Try to load team stats from 2025 data
            try:
                with open('2025_NFL_OFFENSE.json', 'r') as f:
                    offense_data = json.load(f)
                    offense_df = pd.DataFrame(offense_data)
                    
                    st.info(f"Loaded team stats with columns: {list(offense_df.columns)}")
                    
                    # Try different possible column names for team abbreviation
                    team_col = None
                    for possible_col in ['team_abbr', 'team', 'abbreviation', 'team_code', 'Team']:
                        if possible_col in offense_df.columns:
                            team_col = possible_col
                            break
                    
                    if team_col:
                        for _, team in offense_df.iterrows():
                            team_abbr = team[team_col]
                            # Use available stats or defaults
                            win_pct = team.get('win_pct', team.get('WinPct', 0.5))
                            points_for = team.get('points_per_game', team.get('PointsFor', 23.0))
                            points_against = team.get('points_allowed_per_game', team.get('PointsAgainst', 23.0))
                            
                            self.team_stats[team_abbr] = [win_pct, points_for, points_against]
                        
                        st.success(f"Loaded stats for {len(self.team_stats)} teams")
                    else:
                        st.warning("Could not find team abbreviation column in 2025 data")
                        # Create default stats for common teams
                        self.create_default_stats()
                        
            except FileNotFoundError:
                st.warning("2025_NFL_OFFENSE.json not found, using default team stats")
                self.create_default_stats()
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def create_default_stats(self):
        """Create default team stats for demonstration"""
        default_teams = {
            'KC': [0.75, 28.5, 19.2],   # [win_pct, offense, defense]
            'BUF': [0.65, 26.8, 21.1],
            'SF': [0.80, 30.1, 18.5],
            'PHI': [0.70, 27.3, 20.8],
            'DAL': [0.68, 26.9, 21.3],
            'BAL': [0.72, 27.8, 19.8],
            'MIA': [0.66, 29.2, 23.1],
            'CIN': [0.62, 25.7, 22.4],
            'GB': [0.58, 24.3, 23.7],
            'DET': [0.64, 26.1, 22.9],
            'LAR': [0.59, 25.8, 23.5],
            'SEA': [0.55, 24.2, 24.8],
            'LV': [0.45, 21.8, 25.9],
            'DEN': [0.52, 23.1, 24.2],
            'LAC': [0.57, 25.3, 24.1],
            'NE': [0.35, 18.9, 27.3],
            'NYJ': [0.42, 20.5, 26.1],
            'CHI': [0.48, 22.7, 25.3],
            'MIN': [0.53, 24.8, 23.9],
            'NO': [0.51, 23.5, 24.4],
            'ATL': [0.49, 22.9, 24.7],
            'CAR': [0.30, 17.8, 28.5],
            'JAX': [0.56, 24.6, 23.8],
            'IND': [0.54, 24.1, 24.0],
            'HOU': [0.50, 23.3, 24.5],
            'TEN': [0.47, 22.4, 25.1],
            'CLE': [0.61, 25.2, 22.6],
            'PIT': [0.58, 23.9, 23.4],
            'NYG': [0.40, 19.8, 26.8],
            'WAS': [0.43, 21.2, 26.3],
            'ARI': [0.46, 22.1, 25.6],
            'TB': [0.55, 24.5, 24.2]
        }
        self.team_stats = default_teams
        st.info("Using default team statistics for demonstration")
    
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
        home_defense = self.team_stats[home_team][2]
        away_offense = self.team_stats[away_team][1]
        away_defense = self.team_stats[away_team][2]
        
        # Simple average of team tendencies
        total = (home_offense + away_offense + home_defense + away_defense) / 2
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
        st.warning("Continuing with default data for demonstration")
    
    st.success("✅ Model loaded successfully!")
    
    # Display predictions for each game
    st.header("🎯 Game Predictions")
    
    if predictor.schedule is None or len(predictor.schedule) == 0:
        st.error("No schedule data loaded. Using sample games.")
        # Create sample games
        sample_games = [
            {'home_team': 'KC', 'away_team': 'BUF', 'time': '8:15 PM'},
            {'home_team': 'SF', 'away_team': 'SEA', 'time': '4:25 PM'},
            {'home_team': 'DAL', 'away_team': 'PHI', 'time': '4:25 PM'}
        ]
        predictor.schedule = pd.DataFrame(sample_games)
    
    for _, game in predictor.schedule.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_time = game.get('time', 'TBD')
        
        # Find matching odds
        game_odds = None
        if predictor.odds is not None and len(predictor.odds) > 0:
            matching_odds = predictor.odds[
                (predictor.odds['home_team'] == home_team) & 
                (predictor.odds['away_team'] == away_team)
            ]
            if len(matching_odds) > 0:
                game_odds = matching_odds.iloc[0]
        
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
            st.subheader(f"{away_team} @ {home_team}")
            st.caption(f"Time: {game_time}")
            
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
                st.write("Vegas odds: Not available")
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
