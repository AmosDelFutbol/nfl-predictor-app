# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import requests
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="NFL Week 10 Predictions",
    page_icon="🏈",
    layout="wide"
)

class WeatherAPI:
    def __init__(self):
        self.stadiums = None
        self.team_stadiums = None
        self.load_stadiums()
    
    def load_stadiums(self):
        """Load stadium data"""
        try:
            with open('nfl_stadiums.json', 'r') as f:
                data = json.load(f)
            self.stadiums = data['stadiums']
            self.team_stadiums = data['team_stadiums']
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not load stadium data: {e}")
            self.stadiums = {}
            self.team_stadiums = {}
            return False
    
    def get_stadium_coordinates(self):
        """Coordinates for all NFL stadiums"""
        return {
            'Allegiant Stadium': (36.0908, -115.1835),
            'Arrowhead Stadium': (39.0489, -94.4839),
            'AT&T Stadium': (32.7473, -97.0945),
            'Bank of America Stadium': (35.2258, -80.8528),
            'Caesars Superdome': (29.9511, -90.0811),
            'Cleveland Browns Stadium': (41.5061, -81.6995),
            'Empower Field at Mile High': (39.7439, -105.0200),
            'FedExField': (38.9076, -76.8645),
            'Ford Field': (42.3400, -83.0456),
            'GEHA Field at Arrowhead Stadium': (39.0489, -94.4839),
            'Gillette Stadium': (42.0909, -71.2643),
            'Hard Rock Stadium': (25.9580, -80.2389),
            'Highmark Stadium': (42.7738, -78.7870),
            'Lambeau Field': (44.5013, -88.0622),
            'Levi\'s Stadium': (37.4030, -121.9700),
            'Lucas Oil Stadium': (39.7601, -86.1639),
            'Lumen Field': (47.5952, -122.3316),
            'M&T Bank Stadium': (39.2781, -76.6227),
            'MetLife Stadium': (40.8135, -74.0745),
            'Lincoln Financial Field': (39.9008, -75.1675),
            'Nissan Stadium': (36.1665, -86.7713),
            'NRG Stadium': (29.6847, -95.4108),
            'Paycor Stadium': (39.0955, -84.5160),
            'Raymond James Stadium': (27.9759, -82.5033),
            'SoFi Stadium': (33.9535, -118.3389),
            'Soldier Field': (41.8623, -87.6167),
            'State Farm Stadium': (33.5276, -112.2626),
            'TIAA Bank Field': (30.3239, -81.6373),
            'U.S. Bank Stadium': (44.9732, -93.2580)
        }
    
    def get_weather_for_stadium(self, stadium_name, game_date=None):
        """Get weather for a specific stadium using NWS API"""
        if stadium_name not in self.stadiums:
            return self.get_mock_weather(stadium_name, game_date)
        
        coordinates = self.get_stadium_coordinates()
        if stadium_name in coordinates:
            lat, lon = coordinates[stadium_name]
            weather = self.get_weather_nws(lat, lon)
            if weather['success']:
                return weather
        
        # Fallback to mock data
        return self.get_mock_weather(stadium_name, game_date)
    
    def get_weather_nws(self, lat, lon):
        """National Weather Service API (free, no key needed)"""
        try:
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            response = requests.get(points_url, headers={'User-Agent': 'NFLPredictor/1.0'}, timeout=10)
            
            if response.status_code == 200:
                points_data = response.json()
                forecast_url = points_data['properties']['forecast']
                
                forecast_response = requests.get(forecast_url, headers={'User-Agent': 'NFLPredictor/1.0'}, timeout=10)
                forecast_data = forecast_response.json()
                
                current_weather = forecast_data['properties']['periods'][0]
                
                # Extract wind speed number from string like "10 mph"
                wind_speed_str = current_weather['windSpeed'].split()[0]
                wind_speed = float(wind_speed_str) if wind_speed_str.replace('.', '').isdigit() else 10
                
                return {
                    'temperature': current_weather['temperature'],
                    'wind_speed': wind_speed,
                    'conditions': current_weather['shortForecast'],
                    'is_raining': any(word in current_weather['shortForecast'].lower() for word in ['rain', 'shower', 'storm', 'drizzle']),
                    'service': 'nws',
                    'success': True
                }
        except Exception as e:
            st.warning(f"⚠️ NWS API failed: {e}")
        
        return {'success': False}
    
    def get_mock_weather(self, stadium_name, game_date=None):
        """Fallback mock weather data"""
        stadium = self.stadiums.get(stadium_name, {})
        city = stadium.get('city', 'Unknown')
        
        month = datetime.now().month if not game_date else datetime.strptime(game_date, '%Y-%m-%d').month
        
        # Simple mock based on city and month
        if city in ['Green Bay', 'Buffalo', 'Chicago', 'Cleveland']:
            if month in [12, 1, 2]:
                return {'temperature': 25, 'wind_speed': 15, 'is_raining': False, 'service': 'mock', 'success': True}
            elif month in [11, 3]:
                return {'temperature': 45, 'wind_speed': 12, 'is_raining': True, 'service': 'mock', 'success': True}
        
        return {'temperature': 65, 'wind_speed': 8, 'is_raining': False, 'service': 'mock', 'success': True}

class WeatherPredictor:
    def __init__(self):
        self.weather_analysis = None
        self.weather_api = WeatherAPI()
        
    def load_data(self):
        """Load weather analysis data"""
        try:
            with open('nfl_weather_analysis.json', 'r') as f:
                self.weather_analysis = json.load(f)
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not load weather analysis: {e}")
            return False
    
    def get_stadium_weather_impact(self, stadium_name, temperature, wind_speed, is_raining=False):
        """Calculate weather impact for a specific stadium"""
        if stadium_name not in self.weather_api.stadiums:
            return 0, 0  # No adjustment
        
        stadium = self.weather_api.stadiums[stadium_name]
        roof_type = stadium['roof_type']
        
        # If domed stadium, weather has minimal impact
        if roof_type == 'domed':
            return 0, 0
        
        # If retractable roof and bad weather, likely closed
        if roof_type == 'retractable' and (is_raining or temperature < 40 or wind_speed > 20):
            return 0, 0
        
        # Calculate point adjustment based on our historical analysis
        point_adjustment = 0
        
        # Temperature impact
        if temperature <= 32:
            # Freezing weather reduces scoring
            freezing_avg = self.weather_analysis['key_insights']['avg_points_freezing_temp']
            moderate_avg = self.weather_analysis['key_insights']['avg_points_moderate_temp']
            point_adjustment -= (moderate_avg - freezing_avg) / 2
        elif temperature <= 45:
            # Cold weather slight reduction
            point_adjustment -= 1.0
        
        # Wind impact
        if wind_speed > 20:
            windy_avg = self.weather_analysis['key_insights']['avg_points_windy']
            calm_avg = self.weather_analysis['key_insights']['avg_points_calm_wind']
            point_adjustment -= (calm_avg - windy_avg) / 2
        elif wind_speed > 15:
            point_adjustment -= 0.5
        
        # Rain impact
        if is_raining:
            point_adjustment -= 2.0
        
        return point_adjustment, 0  # spread_adjustment
    
    def adjust_prediction_for_weather(self, home_team, away_team, projected_home_score, projected_away_score, game_date):
        """Adjust scores based on weather conditions"""
        # Get home stadium
        if home_team not in self.weather_api.team_stadiums:
            return projected_home_score, projected_away_score
        
        stadium_name = self.weather_api.team_stadiums[home_team]
        
        # Get weather data
        weather = self.weather_api.get_weather_for_stadium(stadium_name, game_date)
        
        if not weather['success']:
            return projected_home_score, projected_away_score
        
        # Get weather impact
        point_adj, spread_adj = self.get_stadium_weather_impact(
            stadium_name, 
            weather['temperature'],
            weather['wind_speed'],
            weather['is_raining']
        )
        
        # Apply adjustments
        adjusted_home_score = max(0, projected_home_score + point_adj)
        adjusted_away_score = max(0, projected_away_score + point_adj)
        
        return adjusted_home_score, adjusted_away_score, weather

class NFLPredictor:
    def __init__(self):
        self.model = None
        self.schedule = None
        self.odds_data = None
        self.team_stats = {}
        self.weather_predictor = WeatherPredictor()
        self.team_mapping = {
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
        
    def load_sos_data(self):
        """Load SOS data from JSON file"""
        try:
            with open('nfl_strength_of_schedule.json', 'r') as f:
                sos_data = json.load(f)
            return sos_data['sos_rankings']
        except Exception as e:
            st.warning(f"⚠️ SOS data not available: {e}")
            return {}

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
                        1  # Home field advantage
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
        return self.team_mapping.get(full_name, full_name)
    
    def load_data(self):
        """Load schedule and odds data with SOS integration"""
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
            
            # Load SOS data
            sos_data = self.load_sos_data()
            
            # NEW: Load weather integration
            if not self.weather_predictor.load_data():
                st.warning("⚠️ Weather data not available - using base predictions")
            else:
                st.success("✅ Weather integration loaded")
            
            # Create default team stats with SOS
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
            
            # Integrate SOS into team stats
            for team_abbr in default_stats.keys():
                # Find full team name
                full_name = None
                for full, abbr in self.team_mapping.items():
                    if abbr == team_abbr:
                        full_name = full
                        break
                
                if full_name and full_name in sos_data:
                    sos_rating = sos_data[full_name]['combined_sos']
                    # Add SOS as 4th element: [win_pct, points_for, points_against, sos]
                    default_stats[team_abbr] = default_stats[team_abbr] + [sos_rating]
                else:
                    # Default average SOS if not available
                    default_stats[team_abbr] = default_stats[team_abbr] + [0.5]
            
            self.team_stats = default_stats
            st.success(f"📊 Loaded stats for {len(self.team_stats)} teams (with SOS)")
            
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
        """Predict game outcome using the trained model with SOS"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
            
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # Use only the first 3 stats for each team to match the trained model
        # [win_pct, points_for, points_against] - no SOS for now to avoid feature mismatch
        features = np.array([[
            home_stats[0], home_stats[1], home_stats[2],
            away_stats[0], away_stats[1], away_stats[2],
            1  # Home field advantage
        ]])
        
        try:
            probabilities = self.model.predict_proba(features)[0]
            home_win_prob = probabilities[1]
            return home_win_prob
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def predict_game_score(self, home_team, away_team, home_win_prob, game_date):
        """Predict the actual score of the game with weather adjustment"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None, None, None
            
        home_offense = self.team_stats[home_team][1]
        away_offense = self.team_stats[away_team][1]
        home_defense = self.team_stats[home_team][2]
        away_defense = self.team_stats[away_team][2]
        
        # Adjust for SOS if available
        home_sos = self.team_stats[home_team][3] if len(self.team_stats[home_team]) > 3 else 0.5
        away_sos = self.team_stats[away_team][3] if len(self.team_stats[away_team]) > 3 else 0.5
        
        # Base score prediction
        home_score = (home_offense + away_defense) / 2 + (home_win_prob - 0.5) * 7 + (home_sos - 0.5) * 2
        away_score = (away_offense + home_defense) / 2 - (home_win_prob - 0.5) * 7 + (away_sos - 0.5) * 2
        
        home_score = max(10, round(home_score))
        away_score = max(10, round(away_score))
        
        # NEW: Apply weather adjustment
        weather_data = None
        if hasattr(self, 'weather_predictor') and self.weather_predictor.weather_analysis:
            home_full = None
            for full, abbr in self.team_mapping.items():
                if abbr == home_team:
                    home_full = full
                    break
            
            away_full = None
            for full, abbr in self.team_mapping.items():
                if abbr == away_team:
                    away_full = full
                    break
            
            if home_full and away_full:
                home_score, away_score, weather_data = self.weather_predictor.adjust_prediction_for_weather(
                    home_full, away_full, home_score, away_score, game_date
                )
    
        return home_score, away_score, weather_data
    
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
        
        # NEW: Get scores with weather adjustment
        home_score, away_score, weather_data = predictor.predict_game_score(home_team, away_team, home_win_prob, game_date)
        
        # Determine model predictions
        model_winner = home_team if home_win_prob > 0.5 else away_team
        model_loser = away_team if home_win_prob > 0.5 else home_team
        
        # Create display
        col1, col2, col3 = st.columns([2, 1.5, 1.5])
        
        with col1:
            st.subheader(f"{away_full} @ {home_full}")
            st.caption(f"Date: {game_date} | {away_team} @ {home_team}")
            
            # NEW: Show weather info
            if weather_data:
                stadium_name = predictor.weather_predictor.weather_api.team_stadiums.get(home_full, "Unknown")
                st.markdown(f"**🌤️  Weather at {stadium_name}:**")
                st.write(f"Temperature: {weather_data['temperature']}°F")
                st.write(f"Wind: {weather_data['wind_speed']} mph")
                st.write(f"Conditions: {weather_data['conditions']}")
                st.write(f"Source: {weather_data['service'].upper()}")
            
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
            
            # Show SOS if available
            if len(predictor.team_stats[home_team]) > 3 and len(predictor.team_stats[away_team]) > 3:
                home_sos = predictor.team_stats[home_team][3]
                away_sos = predictor.team_stats[away_team][3]
                st.caption(f"SOS: {home_team} {home_sos:.3f} | {away_team} {away_sos:.3f}")
        
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
                
                # Over/Under Analysis
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
