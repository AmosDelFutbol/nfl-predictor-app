# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import requests
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="NFL Prediction Model",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
    }
    .score-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .projections-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .final-projections-card {
        background: #ECFDF5;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10B981;
    }
    .weather-card {
        background: #EFF6FF;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #60A5FA;
    }
    .projection-item {
        margin: 1rem 0;
        padding: 0.75rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .probability-badge {
        background: #1E40AF;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.25rem 0;
    }
    .confidence-badge {
        background: #059669;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.25rem 0;
    }
    .odds-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #059669;
        margin-left: 0.5rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .comparison-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
    }
    .model-value {
        font-weight: 600;
        color: #1E40AF;
    }
    .vegas-value {
        font-weight: 600;
        color: #DC2626;
    }
</style>
""", unsafe_allow_html=True)

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
            # Create minimal stadium data
            self.stadiums = {
                'Allegiant Stadium': {'city': 'Las Vegas', 'roof_type': 'domed'},
                'Arrowhead Stadium': {'city': 'Kansas City', 'roof_type': 'open'},
                'AT&T Stadium': {'city': 'Dallas', 'roof_type': 'retractable'},
                'Bank of America Stadium': {'city': 'Charlotte', 'roof_type': 'open'},
                'Caesars Superdome': {'city': 'New Orleans', 'roof_type': 'domed'},
                'Cleveland Browns Stadium': {'city': 'Cleveland', 'roof_type': 'open'},
                'Empower Field at Mile High': {'city': 'Denver', 'roof_type': 'open'},
                'Ford Field': {'city': 'Detroit', 'roof_type': 'domed'},
                'Gillette Stadium': {'city': 'Foxborough', 'roof_type': 'open'},
                'Hard Rock Stadium': {'city': 'Miami', 'roof_type': 'open'},
                'Highmark Stadium': {'city': 'Buffalo', 'roof_type': 'open'},
                'Lambeau Field': {'city': 'Green Bay', 'roof_type': 'open'},
                "Levi's Stadium": {'city': 'Santa Clara', 'roof_type': 'open'},
                'Lucas Oil Stadium': {'city': 'Indianapolis', 'roof_type': 'retractable'},
                'Lumen Field': {'city': 'Seattle', 'roof_type': 'open'},
                'M&T Bank Stadium': {'city': 'Baltimore', 'roof_type': 'open'},
                'MetLife Stadium': {'city': 'East Rutherford', 'roof_type': 'open'},
                'Lincoln Financial Field': {'city': 'Philadelphia', 'roof_type': 'open'},
                'Nissan Stadium': {'city': 'Nashville', 'roof_type': 'open'},
                'NRG Stadium': {'city': 'Houston', 'roof_type': 'retractable'},
                'Paycor Stadium': {'city': 'Cincinnati', 'roof_type': 'open'},
                'Raymond James Stadium': {'city': 'Tampa', 'roof_type': 'open'},
                'SoFi Stadium': {'city': 'Los Angeles', 'roof_type': 'domed'},
                'Soldier Field': {'city': 'Chicago', 'roof_type': 'open'},
                'State Farm Stadium': {'city': 'Glendale', 'roof_type': 'retractable'},
                'TIAA Bank Field': {'city': 'Jacksonville', 'roof_type': 'open'},
                'U.S. Bank Stadium': {'city': 'Minneapolis', 'roof_type': 'domed'}
            }
            
            self.team_stadiums = {
                'Arizona Cardinals': 'State Farm Stadium',
                'Atlanta Falcons': 'Mercedes-Benz Stadium',
                'Baltimore Ravens': 'M&T Bank Stadium',
                'Buffalo Bills': 'Highmark Stadium',
                'Carolina Panthers': 'Bank of America Stadium',
                'Chicago Bears': 'Soldier Field',
                'Cincinnati Bengals': 'Paycor Stadium',
                'Cleveland Browns': 'Cleveland Browns Stadium',
                'Dallas Cowboys': 'AT&T Stadium',
                'Denver Broncos': 'Empower Field at Mile High',
                'Detroit Lions': 'Ford Field',
                'Green Bay Packers': 'Lambeau Field',
                'Houston Texans': 'NRG Stadium',
                'Indianapolis Colts': 'Lucas Oil Stadium',
                'Jacksonville Jaguars': 'TIAA Bank Field',
                'Kansas City Chiefs': 'Arrowhead Stadium',
                'Las Vegas Raiders': 'Allegiant Stadium',
                'Los Angeles Chargers': 'SoFi Stadium',
                'Los Angeles Rams': 'SoFi Stadium',
                'Miami Dolphins': 'Hard Rock Stadium',
                'Minnesota Vikings': 'U.S. Bank Stadium',
                'New England Patriots': 'Gillette Stadium',
                'New Orleans Saints': 'Caesars Superdome',
                'New York Giants': 'MetLife Stadium',
                'New York Jets': 'MetLife Stadium',
                'Philadelphia Eagles': 'Lincoln Financial Field',
                'Pittsburgh Steelers': 'Acrisure Stadium',
                'San Francisco 49ers': "Levi's Stadium",
                'Seattle Seahawks': 'Lumen Field',
                'Tampa Bay Buccaneers': 'Raymond James Stadium',
                'Tennessee Titans': 'Nissan Stadium',
                'Washington Commanders': 'FedExField'
            }
            return True
    
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
            "Levi's Stadium": (37.4030, -121.9700),
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
            response = requests.get(points_url, headers={'User-Agent': 'NFLPredictor/1.0'}, timeout=5)
            
            if response.status_code == 200:
                points_data = response.json()
                forecast_url = points_data['properties']['forecast']
                
                forecast_response = requests.get(forecast_url, headers={'User-Agent': 'NFLPredictor/1.0'}, timeout=5)
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
            pass
        
        return {'success': False, 'temperature': 65, 'wind_speed': 8, 'conditions': 'Unknown', 'is_raining': False, 'service': 'mock'}
    
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
            # Create minimal weather analysis
            self.weather_analysis = {
                "average_impact": {
                    "rain": -2.5,
                    "wind_15_20": -1.5,
                    "wind_20_plus": -3.0,
                    "cold_32_40": -1.0,
                    "cold_below_32": -2.0
                }
            }
            return True
    
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
            point_adjustment -= 2.0
        elif temperature <= 45:
            point_adjustment -= 1.0
        
        # Wind impact
        if wind_speed > 20:
            point_adjustment -= 2.0
        elif wind_speed > 15:
            point_adjustment -= 1.0
        
        # Rain impact
        if is_raining:
            point_adjustment -= 2.0
        
        return point_adjustment, 0
    
    def adjust_prediction_for_weather(self, home_team, away_team, projected_home_score, projected_away_score, game_date):
        """Adjust scores based on weather conditions"""
        # Get home stadium
        if home_team not in self.weather_api.team_stadiums:
            return projected_home_score, projected_away_score, None
        
        stadium_name = self.weather_api.team_stadiums[home_team]
        
        # Get weather data
        weather = self.weather_api.get_weather_for_stadium(stadium_name, game_date)
        
        if not weather.get('success', False):
            return projected_home_score, projected_away_score, weather
        
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
            # Return empty SOS data
            return {}

    def load_model(self):
        """Load or train a simple model"""
        try:
            # First try to load the pickle file
            if os.path.exists('nfl_model.pkl'):
                with open('nfl_model.pkl', 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                # Check if it's a model or a dictionary
                if hasattr(loaded_data, 'predict_proba'):
                    self.model = loaded_data
                    st.success("✅ Model loaded successfully!")
                    return True
                else:
                    st.warning("⚠️ Loaded object is not a model, training new model...")
            
            # If no model file or invalid, train a new one
            return self.train_simple_model()
            
        except Exception as e:
            st.warning(f"⚠️ Model loading failed: {e}. Training new model...")
            return self.train_simple_model()
    
    def train_simple_model(self):
        """Train a simple model quickly"""
        try:
            # Create synthetic training data
            np.random.seed(42)
            n_samples = 1000
            
            # Generate realistic features: [win_pct, points_for, points_against] for home and away
            X = np.random.rand(n_samples, 7)
            
            # Make features more realistic
            X[:, 0] = X[:, 0] * 0.8 + 0.1  # home_win_pct between 0.1-0.9
            X[:, 1] = X[:, 1] * 20 + 15    # home_points_for between 15-35
            X[:, 2] = X[:, 2] * 15 + 15    # home_points_against between 15-30
            X[:, 3] = X[:, 3] * 0.8 + 0.1  # away_win_pct between 0.1-0.9
            X[:, 4] = X[:, 4] * 20 + 15    # away_points_for between 15-35
            X[:, 5] = X[:, 5] * 15 + 15    # away_points_against between 15-30
            X[:, 6] = 1                    # home_field_advantage
            
            # Generate targets based on reasonable probabilities
            home_advantage = 0.03  # Home field advantage factor
            home_win_probs = (X[:, 0] - X[:, 3]) * 0.5 + 0.5 + home_advantage
            home_win_probs = np.clip(home_win_probs, 0.1, 0.9)
            
            # Generate binary outcomes
            y = (np.random.rand(n_samples) < home_win_probs).astype(int)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=10
            )
            self.model.fit(X, y)
            
            # Save the model
            with open('nfl_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            st.success("✅ New model trained and saved successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            # Create a fallback model
            self.model = None
            return False
    
    def get_team_abbreviation(self, full_name):
        """Convert full team name to abbreviation"""
        return self.team_mapping.get(full_name, full_name)
    
    def create_sample_schedule(self):
        """Create sample schedule data"""
        sample_games = [
            {'home': 'Kansas City Chiefs', 'away': 'Philadelphia Eagles', 'date': '2024-11-15'},
            {'home': 'San Francisco 49ers', 'away': 'Dallas Cowboys', 'date': '2024-11-15'},
            {'home': 'Buffalo Bills', 'away': 'Miami Dolphins', 'date': '2024-11-15'},
            {'home': 'Baltimore Ravens', 'away': 'Cincinnati Bengals', 'date': '2024-11-15'},
            {'home': 'Detroit Lions', 'away': 'Green Bay Packers', 'date': '2024-11-15'},
            {'home': 'Los Angeles Rams', 'away': 'Seattle Seahawks', 'date': '2024-11-15'}
        ]
        self.schedule = pd.DataFrame(sample_games)
    
    def create_sample_odds(self):
        """Create sample odds data"""
        sample_odds = [
            {
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Philadelphia Eagles', 
                'market': 'h2h',
                'label': 'Kansas City Chiefs',
                'price': -150
            },
            {
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Philadelphia Eagles',
                'market': 'h2h', 
                'label': 'Philadelphia Eagles',
                'price': +130
            },
            {
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Philadelphia Eagles',
                'market': 'spreads',
                'label': 'Kansas City Chiefs',
                'point': -3.5,
                'price': -110
            },
            {
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Philadelphia Eagles',
                'market': 'totals',
                'label': 'Over',
                'point': 48.5,
                'price': -110
            },
            {
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Philadelphia Eagles',
                'market': 'totals',
                'label': 'Under', 
                'point': 48.5,
                'price': -110
            }
        ]
        self.odds_data = pd.DataFrame(sample_odds)
    
    def load_data(self):
        """Load schedule and odds data with SOS integration"""
        try:
            # Try to load schedule
            schedule_loaded = False
            for file in ['schedule.json', 'data/schedule.json']:
                try:
                    with open(file, 'r') as f:
                        schedule_data = json.load(f)
                    # Handle different schedule formats
                    if isinstance(schedule_data, dict):
                        if 'weeks' in schedule_data:
                            # Get first week
                            first_week = list(schedule_data['weeks'].keys())[0]
                            self.schedule = pd.DataFrame(schedule_data['weeks'][first_week])
                        else:
                            first_key = list(schedule_data.keys())[0]
                            self.schedule = pd.DataFrame(schedule_data[first_key])
                    else:
                        self.schedule = pd.DataFrame(schedule_data)
                    schedule_loaded = True
                    break
                except FileNotFoundError:
                    continue
            
            if not schedule_loaded:
                st.warning("⚠️ Schedule data not found. Using sample data.")
                self.create_sample_schedule()
            
            # Try to load odds
            odds_loaded = False
            for file in ['odds.json', 'data/odds.json']:
                try:
                    with open(file, 'r') as f:
                        odds_data = json.load(f)
                    self.odds_data = pd.DataFrame(odds_data)
                    odds_loaded = True
                    break
                except FileNotFoundError:
                    continue
            
            if not odds_loaded:
                st.warning("⚠️ Odds data not found. Using sample data.")
                self.create_sample_odds()
            
            # Load SOS data
            sos_data = self.load_sos_data()
            
            # Load weather integration
            self.weather_predictor.load_data()

            # Create default team stats with SOS
            default_stats = {
                'KC': [0.75, 28.5, 19.2, 0.5], 'BUF': [0.65, 26.8, 21.1, 0.5], 'SF': [0.80, 30.1, 18.5, 0.5],
                'PHI': [0.70, 27.3, 20.8, 0.5], 'DAL': [0.68, 26.9, 21.3, 0.5], 'BAL': [0.72, 27.8, 19.8, 0.5],
                'MIA': [0.66, 29.2, 23.1, 0.5], 'CIN': [0.62, 25.7, 22.4, 0.5], 'GB': [0.58, 24.3, 23.7, 0.5],
                'DET': [0.64, 26.1, 22.9, 0.5], 'LAR': [0.59, 25.8, 23.5, 0.5], 'SEA': [0.55, 24.2, 24.8, 0.5],
                'LV': [0.45, 21.8, 25.9, 0.5], 'DEN': [0.52, 23.1, 24.2, 0.5], 'LAC': [0.57, 25.3, 24.1, 0.5],
                'NE': [0.35, 18.9, 27.3, 0.5], 'NYJ': [0.42, 20.5, 26.1, 0.5], 'CHI': [0.48, 22.7, 25.3, 0.5],
                'MIN': [0.53, 24.8, 23.9, 0.5], 'NO': [0.51, 23.5, 24.4, 0.5], 'ATL': [0.49, 22.9, 24.7, 0.5],
                'CAR': [0.30, 17.8, 28.5, 0.5], 'JAX': [0.56, 24.6, 23.8, 0.5], 'IND': [0.54, 24.1, 24.0, 0.5],
                'HOU': [0.50, 23.3, 24.5, 0.5], 'TEN': [0.47, 22.4, 25.1, 0.5], 'CLE': [0.61, 25.2, 22.6, 0.5],
                'PIT': [0.58, 23.9, 23.4, 0.5], 'NYG': [0.40, 19.8, 26.8, 0.5], 'WAS': [0.43, 21.2, 26.3, 0.5],
                'ARI': [0.46, 22.1, 25.6, 0.5], 'TB': [0.55, 24.5, 24.2, 0.5]
            }
            
            # Integrate SOS into team stats if available
            for team_abbr in default_stats.keys():
                # Find full team name
                full_name = None
                for full, abbr in self.team_mapping.items():
                    if abbr == team_abbr:
                        full_name = full
                        break
                
                if full_name and full_name in sos_data:
                    sos_rating = sos_data[full_name].get('combined_sos', 0.5)
                    # Update SOS in team stats
                    default_stats[team_abbr][3] = sos_rating
            
            self.team_stats = default_stats
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            # Create minimal data as fallback
            self.create_sample_schedule()
            self.create_sample_odds()
            return True
    
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
        features = np.array([[
            home_stats[0], home_stats[1], home_stats[2],
            away_stats[0], away_stats[1], away_stats[2],
            1  # Home field advantage
        ]])
        
        try:
            if self.model is None:
                # Fallback prediction based on team stats
                home_win_prob = 0.5 + (home_stats[0] - away_stats[0]) * 0.3
                return max(0.1, min(0.9, home_win_prob))
            
            probabilities = self.model.predict_proba(features)[0]
            home_win_prob = probabilities[1]
            return home_win_prob
        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Fallback prediction
            home_win_prob = 0.5 + (home_stats[0] - away_stats[0]) * 0.3
            return max(0.1, min(0.9, home_win_prob))
    
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
        
        # Apply weather adjustment
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

def calculate_cover_probability(model_spread, vegas_spread):
    """Calculate probability of covering the spread"""
    # Simple heuristic: the closer model spread is to Vegas spread, the higher the confidence
    spread_diff = abs(model_spread - vegas_spread)
    if spread_diff <= 1:
        return 75  # High confidence
    elif spread_diff <= 3:
        return 65  # Medium confidence
    elif spread_diff <= 6:
        return 55  # Low confidence
    else:
        return 50  # Toss-up

def calculate_over_probability(model_total, vegas_total):
    """Calculate probability of over hitting"""
    # Simple heuristic based on difference between model and Vegas total
    total_diff = model_total - vegas_total
    if total_diff >= 3:
        return 70  # Strong over
    elif total_diff >= 1:
        return 60  # Lean over
    elif total_diff >= -1:
        return 50  # Toss-up
    elif total_diff >= -3:
        return 40  # Lean under
    else:
        return 30  # Strong under

def create_game_card(predictor, game, game_odds, home_win_prob, home_score, away_score, weather_data):
    """Create a professional game card matching the desired layout"""
    
    home_full = game['home']
    away_full = game['away']
    home_team = predictor.get_team_abbreviation(home_full)
    away_team = predictor.get_team_abbreviation(away_full)
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Score Header (like "Broncos 24 @ 21 Raiders")
        st.markdown(f'<div class="score-header">{away_team} {int(away_score)} @ {int(home_score)} {home_team}</div>', unsafe_allow_html=True)
        
        # Two Column Layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Projections Card
            st.markdown('<div class="projections-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Projections</div>', unsafe_allow_html=True)
            
            # Winner Projection
            model_winner = home_full if home_win_prob > 0.5 else away_full
            winner_abbr = home_team if home_win_prob > 0.5 else away_team
            st.markdown('<div class="projection-item">', unsafe_allow_html=True)
            st.markdown(f'**Winner:** {model_winner}')
            if game_odds:
                home_ml = game_odds.get('home_moneyline', 'N/A')
                away_ml = game_odds.get('away_moneyline', 'N/A')
                if home_win_prob > 0.5:
                    st.markdown(f'<span class="odds-value">{home_team}: {home_ml} | {away_team}: {away_ml}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="odds-value">{away_team}: {away_ml} | {home_team}: {home_ml}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Spread Projection
            model_spread = predictor.convert_prob_to_spread(home_win_prob)
            st.markdown('<div class="projection-item">', unsafe_allow_html=True)
            st.markdown(f'**Spread:** {model_spread:.1f}')
            if game_odds and game_odds.get('spread'):
                vegas_spread = game_odds['spread']
                st.markdown(f'<span class="odds-value">Spread: {vegas_spread}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Totals Projection
            model_total = predictor.predict_total_points(home_team, away_team)
            st.markdown('<div class="projection-item">', unsafe_allow_html=True)
            st.markdown(f'**Totals:** {model_total:.1f}')
            if game_odds and game_odds.get('total'):
                vegas_total = game_odds['total']
                st.markdown(f'<span class="odds-value">{vegas_total}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close projections-card
            
            # Weather Card
            if weather_data and weather_data.get('success'):
                st.markdown('<div class="weather-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Weather</div>', unsafe_allow_html=True)
                
                stadium_name = predictor.weather_predictor.weather_api.team_stadiums.get(home_full, "Unknown Stadium")
                stadium_info = predictor.weather_predictor.weather_api.stadiums.get(stadium_name, {})
                roof_type = stadium_info.get('roof_type', 'Unknown')
                
                st.write(f"**Venue:** {stadium_name}")
                st.write(f"**Conditions:** {weather_data.get('conditions', 'Unknown')}")
                st.write(f"**Temperature:** {weather_data.get('temperature', 'N/A')}°F")
                st.write(f"**Wind:** {weather_data.get('wind_speed', 'N/A')} mph")
                st.write(f"**Stadium Type:** {roof_type.title()}")
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close weather-card
        
        with col2:
            # Final Projections Card
            st.markdown('<div class="final-projections-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Final Projections</div>', unsafe_allow_html=True)
            
            # Winner Pick with Probability
            st.markdown('<div class="projection-item">', unsafe_allow_html=True)
            st.markdown(f'**Winner:** {model_winner}')
            win_prob_pct = home_win_prob * 100 if home_win_prob > 0.5 else (1 - home_win_prob) * 100
            st.markdown(f'<div class="probability-badge">WIN Probability: {win_prob_pct:.0f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Spread Pick
            if game_odds and game_odds.get('spread'):
                vegas_spread = game_odds['spread']
                model_spread = predictor.convert_prob_to_spread(home_win_prob)
                
                # Determine ATS pick
                if vegas_spread < 0:  # Home team favored
                    if model_spread <= vegas_spread:  # Model thinks home covers
                        ats_pick = f"{home_team} to cover"
                    else:  # Model thinks away covers
                        ats_pick = f"{away_team} to cover"
                else:  # Away team favored
                    if model_spread >= vegas_spread:  # Model thinks away covers
                        ats_pick = f"{away_team} to cover"
                    else:  # Model thinks home covers
                        ats_pick = f"{home_team} to cover"
                
                cover_prob = calculate_cover_probability(model_spread, vegas_spread)
                
                st.markdown('<div class="projection-item">', unsafe_allow_html=True)
                st.markdown(f'**Spread:** {ats_pick}')
                st.markdown(f'<div class="confidence-badge">{cover_prob}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Totals Pick
            if game_odds and game_odds.get('total'):
                vegas_total = game_odds['total']
                model_total = predictor.predict_total_points(home_team, away_team)
                
                if model_total > vegas_total:
                    totals_pick = "Lean Over"
                else:
                    totals_pick = "Lean Under"
                
                over_prob = calculate_over_probability(model_total, vegas_total)
                if totals_pick == "Lean Under":
                    over_prob = 100 - over_prob
                
                st.markdown('<div class="projection-item">', unsafe_allow_html=True)
                st.markdown(f'**Totals:** {totals_pick}')
                st.markdown(f'<div class="confidence-badge">{over_prob}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close final-projections-card
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close game-card

def main():
    # Header
    st.markdown('<div class="main-header">NFL Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Week 10 • Model Projections vs Vegas Odds</div>', unsafe_allow_html=True)
    
    # Initialize predictor with loading state
    with st.spinner('Loading NFL predictions...'):
        predictor = NFLPredictor()
        
        # Load model and data
        if not predictor.load_model():
            st.error("Failed to load model")
            st.stop()
            
        if not predictor.load_data():
            st.error("Failed to load data files")
            st.stop()
    
    # Add refresh button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Refresh Predictions", use_container_width=True):
            st.rerun()
    
    # Display predictions for each game
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
            st.warning(f"Could not generate prediction for {away_team} @ {home_team}")
            continue
        
        # Get scores with weather adjustment
        home_score, away_score, weather_data = predictor.predict_game_score(home_team, away_team, home_win_prob, game_date)
        
        # Create professional game card
        create_game_card(predictor, game, game_odds, home_win_prob, home_score, away_score, weather_data)
        
        st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
