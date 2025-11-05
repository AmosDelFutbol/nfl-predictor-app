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
            with open('week_10_schedule.json', 'r') as f:
                self.schedule = pd.DataFrame(json.load(f))
            
            with open('week_10_odds.json', 'r') as f:
                self.odds = pd.DataFrame(json.load(f))
            
            # Load team stats from your 2025 data
            with open('2025_NFL_OFFENSE.json', 'r') as f:
                offense_data = pd.DataFrame(json.load(f))
                # Process offense data to match our feature format
                for _, team in offense_data.iterrows():
                    team_abbr = team['team_abbr']  # Adjust based on your actual column names
                    self.team_stats[team_abbr] = [
                        team.get('win_pct', 0.5),      # You may need to calculate this
                        team.get('points_per_game', 23.0),
                        team.get('points_allowed_per_game', 23.0)
                    ]
            
            return True
        except FileNotFoundError as e:
            st.error(f"Data file not found: {e}")
            return False
    
    def calculate_implied_probability(self, american_odds):
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def predict_game(self, home_team, away_team):
        """Predict game outcome using the trained model"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
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
        probabilities = self.model.predict_proba(features)[0]
        home_win_prob = probabilities[1]
        
        return home_win_prob
    
    def convert_prob_to_spread(self, home_win_prob):
        """Convert win probability to point spread"""
        # Using common conversion: prob -> spread
        # 50% = 0, 60% = -3, 70% = -6, etc.
        if home_win_prob >= 0.5:
            spread = -((home_win_prob - 0.5) * 14)  # Convert to negative spread for favorite
        else:
            spread = ((0.5 - home_win_prob) * 14)   # Positive spread for underdog
        
        return round(spread * 2) / 2  # Round to nearest 0.5
    
    def predict_total_points(self, home_team, away_team):
        """Predict total points based on team offensive/defensive stats"""
        home_offense = self.team_stats[home_team][1]
        home_defense = self.team_stats[home_team][2]
        away_offense = self.team_stats[away_team][1]
        away_defense = self.team_stats[away_team][2]
        
        # Simple average of team tendencies
        total = (home_offense + away_offense + (home_defense + away_defense)) / 2
        return round(total * 2) / 2  # Round to nearest 0.5

def main():
    st.title("🏈 NFL Week 10 Predictions & Betting Analysis")
    st.markdown("### Model Projections vs. Vegas Odds")
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Load data
    if not predictor.load_model() or not predictor.load_data():
        st.stop()
    
    st.success("✅ Model and data loaded successfully!")
    
    # Display predictions for each game
    st.header("🎯 Game Predictions")
    
    for _, game in predictor.schedule.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_time = game.get('time', 'TBD')
        
        # Find matching odds
        game_odds = predictor.odds[
            (predictor.odds['home_team'] == home_team) & 
            (predictor.odds['away_team'] == away_team)
        ]
        
        if len(game_odds) == 0:
            continue
            
        game_odds = game_odds.iloc[0]
        
        # Make prediction
        home_win_prob = predictor.predict_game(home_team, away_team)
        if home_win_prob is None:
            continue
        
        # Calculate model projections
        model_spread = predictor.convert_prob_to_spread(home_win_prob)
        model_total = predictor.predict_total_points(home_team, away_team)
        
        # Get Vegas data
        vegas_spread = game_odds.get('spread', 0)
        vegas_total = game_odds.get('over_under', 45.0)
        home_ml = game_odds.get('home_moneyline', 0)
        away_ml = game_odds.get('away_moneyline', 0)
        
        # Determine favorite based on spread
        if vegas_spread < 0:
            vegas_favorite = home_team
            vegas_underdog = away_team
            vegas_fav_spread = abs(vegas_spread)
        else:
            vegas_favorite = away_team
            vegas_underdog = home_team
            vegas_fav_spread = vegas_spread
        
        # Model's favorite
        if home_win_prob > 0.5:
            model_favorite = home_team
            model_underdog = away_team
            model_fav_spread = abs(model_spread)
        else:
            model_favorite = away_team
            model_underdog = home_team
            model_fav_spread = model_spread
        
        # Create columns for display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"{away_team} @ {home_team}")
            st.caption(f"Time: {game_time}")
            
            # Vegas Odds
            st.markdown("**Vegas Odds:**")
            st.write(f"Spread: {vegas_favorite} -{vegas_fav_spread}")
            st.write(f"Total: {vegas_total}")
            st.write(f"Moneyline: {home_team} {home_ml} | {away_team} {away_ml}")
        
        with col2:
            st.markdown("**Model Projections:**")
            st.metric("Home Win Probability", f"{home_win_prob:.1%}")
            st.metric("Projected Spread", f"{model_favorite} -{model_fav_spread}")
            st.metric("Projected Total", f"{model_total}")
        
        with col3:
            st.markdown("**Betting Recommendations:**")
            
            # Spread Analysis
            spread_diff = abs(model_fav_spread - vegas_fav_spread)
            if spread_diff > 1.0:
                if model_fav_spread < vegas_fav_spread:
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
            
            # Moneyline Value
            model_implied_ml = -100 / home_win_prob + 100 if home_win_prob > 0.5 else 100 / (1 - home_win_prob) - 100
            if home_win_prob > 0.6 and home_ml < model_implied_ml:
                st.success(f"💰 VALUE: {home_team} ML")
            elif home_win_prob < 0.4 and away_ml < model_implied_ml:
                st.success(f"💰 VALUE: {away_team} ML")
        
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
