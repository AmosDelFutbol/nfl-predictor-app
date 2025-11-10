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
    page_title="NFL Prediction Model",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to check for required data files
def check_required_files():
    """Check if all required data files exist"""
    required_files = [
        'teams_power_rating.csv',
        'week_10_schedule.json', 
        'week_10_odds.json',
        'nfl_strength_of_schedule.json',
        'spreadspoke_scores.json'
    ]
    
    missing_files = []
    for file in required_files:
        try:
            if file.endswith('.csv'):
                pd.read_csv(file)
            else:
                with open(file, 'r') as f:
                    json.load(f)
        except:
            missing_files.append(file)
    
    return missing_files

# Custom CSS for professional styling - CLEANED UP VERSION
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1E3A8A 0%, #3730A3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    .game-card {
        background: white;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.15), 0 4px 12px -4px rgba(0, 0, 0, 0.1);
        border: 1px solid #F3F4F6;
    }
    .score-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    .projections-section {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .efficiency-section {
        background: #FFFBEB;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 4px solid #D97706;
    }
    .final-section {
        background: #F0FDF4;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 4px solid #10B981;
    }
    .odds-section {
        background: #FEF7FF;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 4px solid #8B5CF6;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .metric-label {
        font-weight: 600;
        color: #6B7280;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-weight: 700;
        color: #1F2937;
        font-size: 1.1rem;
    }
    .probability-badge {
        background: #1E40AF;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .confidence-badge {
        background: #059669;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .elo-badge {
        background: #7C3AED;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .odds-value {
        font-weight: 700;
        color: #059669;
        font-size: 1.1rem;
    }
    .positive {
        color: #059669;
        font-weight: 600;
    }
    .negative {
        color: #DC2626;
        font-weight: 600;
    }
    .team-comparison {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.75rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .team-name {
        font-weight: 700;
        color: #1F2937;
    }
    .comparison-value {
        font-weight: 600;
        color: #6B7280;
    }
    .insight-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        border-right: 1px solid #E5E7EB;
        border-top: 1px solid #E5E7EB;
        border-bottom: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Weather Predictor Class
class WeatherPredictor:
    def __init__(self):
        self.weather_analysis = {}
        self.weather_api = WeatherAPI()
    
    def load_data(self):
        """Load weather data"""
        try:
            with open('weather_data.json', 'r') as f:
                self.weather_analysis = json.load(f)
            return True
        except:
            self.weather_analysis = {}
            return True
    
    def adjust_prediction_for_weather(self, home_full, away_full, home_score, away_score, game_date):
        """Apply weather adjustments to scores"""
        weather_data = {
            'success': False,
            'conditions': 'Unknown',
            'temperature': 'N/A', 
            'wind_speed': 'N/A'
        }
        return home_score, away_score, weather_data

class WeatherAPI:
    def __init__(self):
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
            'San Francisco 49ers': 'Levi\'s Stadium',
            'Seattle Seahawks': 'Lumen Field',
            'Tampa Bay Buccaneers': 'Raymond James Stadium',
            'Tennessee Titans': 'Nissan Stadium',
            'Washington Commanders': 'FedExField'
        }
        
        self.stadiums = {
            'State Farm Stadium': {'roof_type': 'retractable'},
            'Mercedes-Benz Stadium': {'roof_type': 'retractable'},
            'M&T Bank Stadium': {'roof_type': 'open'},
            'Highmark Stadium': {'roof_type': 'open'},
            'Bank of America Stadium': {'roof_type': 'open'},
            'Soldier Field': {'roof_type': 'open'},
            'Paycor Stadium': {'roof_type': 'open'},
            'Cleveland Browns Stadium': {'roof_type': 'open'},
            'AT&T Stadium': {'roof_type': 'retractable'},
            'Empower Field at Mile High': {'roof_type': 'open'},
            'Ford Field': {'roof_type': 'dome'},
            'Lambeau Field': {'roof_type': 'open'},
            'NRG Stadium': {'roof_type': 'retractable'},
            'Lucas Oil Stadium': {'roof_type': 'retractable'},
            'TIAA Bank Field': {'roof_type': 'open'},
            'Arrowhead Stadium': {'roof_type': 'open'},
            'Allegiant Stadium': {'roof_type': 'dome'},
            'SoFi Stadium': {'roof_type': 'open'},
            'Hard Rock Stadium': {'roof_type': 'open'},
            'U.S. Bank Stadium': {'roof_type': 'fixed'},
            'Gillette Stadium': {'roof_type': 'open'},
            'Caesars Superdome': {'roof_type': 'dome'},
            'MetLife Stadium': {'roof_type': 'open'},
            'Lincoln Financial Field': {'roof_type': 'open'},
            'Acrisure Stadium': {'roof_type': 'open'},
            'Levi\'s Stadium': {'roof_type': 'open'},
            'Lumen Field': {'roof_type': 'open'},
            'Raymond James Stadium': {'roof_type': 'open'},
            'Nissan Stadium': {'roof_type': 'open'},
            'FedExField': {'roof_type': 'open'}
        }

# ELO Processor Class
class ELOProcessor:
    def __init__(self):
        self.team_elos = {}
        self.base_elo = 1500
        self.k_factor = 20
        self.home_field_advantage = 24
        
    def load_elo_data(self):
        """Load and process the ELO CSV with all advanced metrics"""
        try:
            elo_df = pd.read_csv('teams_power_rating.csv')
            st.success("✅ Successfully loaded ELO data")
            
            for _, row in elo_df.iterrows():
                team = row['Team']
                
                # Map team names to abbreviations
                team_abbr = self._get_team_abbreviation(team)
                
                if team_abbr:
                    self.team_elos[team_abbr] = {
                        'nfelo': row['nfelo'],
                        'elo': row.get('Elo', row['nfelo']),
                        'qb_adj': row['QB Adj'],
                        'value': row['Value'],
                        'wow_change': row['WoW'],
                        'ytd_performance': row['YTD'],
                        'off_epa_play': row['Play'],
                        'off_epa_pass': row['Pass'],
                        'off_epa_rush': row['Rush'],
                        'def_epa_play': row['Play.1'],
                        'def_epa_pass': row['Pass.1'],
                        'def_epa_rush': row['Rush.1'],
                        'net_epa': row['Play.2'],
                        'points_for': row['For'],
                        'points_against': row['Against'],
                        'point_differential': row['Dif'],
                        'wins': row['Wins'],
                        'pythag_wins': row['Pythag'],
                        'film_grade': row.get('Film', 5),
                        'season': row.get('Season', 2025),
                        'team_name': team
                    }
            
            st.success(f"✅ Loaded ELO data for {len(self.team_elos)} teams")
            return True
            
        except Exception as e:
            st.error(f"❌ Could not load ELO data: {e}")
            return False
    
    def _get_team_abbreviation(self, team_name):
        """Convert team names from CSV to abbreviations"""
        team_mapping = {
            'KC': 'KC', 'Kansas City': 'KC', 'Kansas City Chiefs': 'KC',
            'LAR': 'LAR', 'Los Angeles Rams': 'LAR', 'LA Rams': 'LAR',
            'SF': 'SF', 'San Francisco': 'SF', 'San Francisco 49ers': 'SF',
            'BAL': 'BAL', 'Baltimore': 'BAL', 'Baltimore Ravens': 'BAL',
            'PHI': 'PHI', 'Philadelphia': 'PHI', 'Philadelphia Eagles': 'PHI',
            'DAL': 'DAL', 'Dallas': 'DAL', 'Dallas Cowboys': 'DAL',
            'BUF': 'BUF', 'Buffalo': 'BUF', 'Buffalo Bills': 'BUF',
            'MIA': 'MIA', 'Miami': 'MIA', 'Miami Dolphins': 'MIA',
            'DET': 'DET', 'Detroit': 'DET', 'Detroit Lions': 'DET',
            'GB': 'GB', 'Green Bay': 'GB', 'Green Bay Packers': 'GB',
            'MIN': 'MIN', 'Minnesota': 'MIN', 'Minnesota Vikings': 'MIN',
            'CHI': 'CHI', 'Chicago': 'CHI', 'Chicago Bears': 'CHI',
            'NO': 'NO', 'New Orleans': 'NO', 'New Orleans Saints': 'NO',
            'TB': 'TB', 'Tampa Bay': 'TB', 'Tampa Bay Buccaneers': 'TB',
            'ATL': 'ATL', 'Atlanta': 'ATL', 'Atlanta Falcons': 'ATL',
            'CAR': 'CAR', 'Carolina': 'CAR', 'Carolina Panthers': 'CAR',
            'SEA': 'SEA', 'Seattle': 'SEA', 'Seattle Seahawks': 'SEA',
            'ARI': 'ARI', 'Arizona': 'ARI', 'Arizona Cardinals': 'ARI',
            'LAC': 'LAC', 'Los Angeles Chargers': 'LAC', 'LA Chargers': 'LAC',
            'LV': 'LV', 'Las Vegas': 'LV', 'Las Vegas Raiders': 'LV',
            'DEN': 'DEN', 'Denver': 'DEN', 'Denver Broncos': 'DEN',
            'CIN': 'CIN', 'Cincinnati': 'CIN', 'Cincinnati Bengals': 'CIN',
            'CLE': 'CLE', 'Cleveland': 'CLE', 'Cleveland Browns': 'CLE',
            'PIT': 'PIT', 'Pittsburgh': 'PIT', 'Pittsburgh Steelers': 'PIT',
            'IND': 'IND', 'Indianapolis': 'IND', 'Indianapolis Colts': 'IND',
            'HOU': 'HOU', 'Houston': 'HOU', 'Houston Texans': 'HOU',
            'JAX': 'JAX', 'Jacksonville': 'JAX', 'Jacksonville Jaguars': 'JAX',
            'TEN': 'TEN', 'Tennessee': 'TEN', 'Tennessee Titans': 'TEN',
            'NYJ': 'NYJ', 'New York Jets': 'NYJ',
            'NYG': 'NYG', 'New York Giants': 'NYG',
            'NE': 'NE', 'New England': 'NE', 'New England Patriots': 'NE',
            'WAS': 'WAS', 'Washington': 'WAS', 'Washington Commanders': 'WAS'
        }
        return team_mapping.get(team_name, None)
    
    def get_team_elo(self, team_abbr):
        """Get ELO data for a team by abbreviation"""
        return self.team_elos.get(team_abbr, {
            'nfelo': self.base_elo, 'elo': self.base_elo, 'qb_adj': 0, 'value': 0,
            'wow_change': 0, 'ytd_performance': 0, 'off_epa_play': 0, 'off_epa_pass': 0,
            'off_epa_rush': 0, 'def_epa_play': 0, 'def_epa_pass': 0, 'def_epa_rush': 0,
            'net_epa': 0, 'points_for': 0, 'points_against': 0, 'point_differential': 0,
            'wins': 0, 'pythag_wins': 0, 'film_grade': 5, 'season': 2025, 'team_name': team_abbr
        })
    
    def calculate_elo_win_probability(self, home_team, away_team):
        """Calculate win probability based on ELO difference"""
        home_data = self.get_team_elo(home_team)
        away_data = self.get_team_elo(away_team)
        
        home_elo = home_data['nfelo'] + self.home_field_advantage
        away_elo = away_data['nfelo']
        
        # ELO win probability formula
        elo_diff = home_elo - away_elo
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        
        return win_prob
    
    def get_matchup_analysis(self, home_team, away_team):
        """Generate detailed matchup analysis using ELO metrics"""
        home_data = self.get_team_elo(home_team)
        away_data = self.get_team_elo(away_team)
        
        analysis = {
            'elo_advantage': home_data['nfelo'] - away_data['nfelo'],
            'offensive_matchup': home_data['off_epa_play'] - away_data['def_epa_play'],
            'defensive_matchup': away_data['off_epa_play'] - home_data['def_epa_play'],
            'qb_advantage': home_data['qb_adj'] - away_data['qb_adj'],
            'momentum_advantage': home_data['wow_change'] - away_data['wow_change'],
            'efficiency_advantage': home_data['net_epa'] - away_data['net_epa'],
            'home_elo': home_data['nfelo'],
            'away_elo': away_data['nfelo']
        }
        
        return analysis

# Enhanced NFLPredictor with ELO integration
class NFLPredictor:
    def __init__(self):
        self.model = None
        self.schedule = None
        self.odds_data = None
        self.team_stats = {}
        self.weather_predictor = WeatherPredictor()
        self.elo_processor = ELOProcessor()
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
            return {}

    def load_model(self):
        """Load or train a simple model"""
        try:
            with open('nfl_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            return True
        except:
            return self.train_simple_model()
    
    def train_simple_model(self):
        """Train a simple model quickly"""
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
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            return False
    
    def get_team_abbreviation(self, full_name):
        """Convert full team name to abbreviation"""
        return self.team_mapping.get(full_name, full_name)
    
    def load_data(self):
        """Load schedule and odds data with SOS and ELO integration"""
        try:
            # Load schedule
            with open('week_10_schedule.json', 'r') as f:
                schedule_data = json.load(f)
                self.schedule = pd.DataFrame(schedule_data['Week 10'])
            
            # Load odds
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
                self.odds_data = pd.DataFrame(odds_data)
            
            # Load SOS data
            sos_data = self.load_sos_data()
            
            # Load ELO data
            self.elo_processor.load_elo_data()
            
            # Load weather integration
            self.weather_predictor.load_data()

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

    def predict_game_with_elo_enhanced(self, home_team, away_team):
        """Enhanced prediction that actually uses ELO to improve accuracy"""
        # Get ML model prediction
        ml_win_prob = self.predict_game(home_team, away_team)
        
        if ml_win_prob is None:
            return None
            
        # Get ELO prediction
        elo_win_prob = self.elo_processor.calculate_elo_win_probability(home_team, away_team)
        
        # Get team ELO data for advanced adjustments
        home_elo_data = self.elo_processor.get_team_elo(home_team)
        away_elo_data = self.elo_processor.get_team_elo(away_team)
        
        # Calculate confidence factors based on ELO metrics
        confidence_factors = self._calculate_elo_confidence_factors(home_elo_data, away_elo_data)
        
        # Dynamic weighting based on confidence
        if confidence_factors['high_confidence']:
            # When ELO metrics strongly favor one team, trust ELO more
            elo_weight = 0.7
            ml_weight = 0.3
        elif confidence_factors['moderate_confidence']:
            # Balanced approach
            elo_weight = 0.5
            ml_weight = 0.5
        else:
            # When uncertain, lean slightly toward ML model
            elo_weight = 0.4
            ml_weight = 0.6
        
        # Apply EPA-based adjustments
        epa_adjustment = self._calculate_epa_based_adjustment(home_elo_data, away_elo_data)
        
        # Combine predictions with dynamic weighting and EPA adjustments
        base_combined = (ml_weight * ml_win_prob) + (elo_weight * elo_win_prob)
        final_win_prob = base_combined + epa_adjustment
        
        # Ensure probability stays within bounds
        final_win_prob = max(0.05, min(0.95, final_win_prob))
        
        return {
            'combined_win_prob': final_win_prob,
            'ml_win_prob': ml_win_prob,
            'elo_win_prob': elo_win_prob,
            'elo_weight': elo_weight,
            'ml_weight': ml_weight,
            'epa_adjustment': epa_adjustment,
            'home_elo': home_elo_data['nfelo'],
            'away_elo': away_elo_data['nfelo'],
            'confidence_level': confidence_factors['confidence_level']
        }
    
    def _calculate_elo_confidence_factors(self, home_data, away_data):
        """Determine how much to trust ELO based on team metrics"""
        elo_diff = abs(home_data['nfelo'] - away_data['nfelo'])
        net_epa_diff = abs(home_data['net_epa'] - away_data['net_epa'])
        
        # High confidence if both ELO and EPA strongly favor one team
        if elo_diff > 100 and net_epa_diff > 0.15:
            return {'high_confidence': True, 'moderate_confidence': False, 'confidence_level': 'high'}
        # Moderate confidence if either metric strongly favors one team
        elif elo_diff > 75 or net_epa_diff > 0.1:
            return {'high_confidence': False, 'moderate_confidence': True, 'confidence_level': 'moderate'}
        else:
            return {'high_confidence': False, 'moderate_confidence': False, 'confidence_level': 'low'}
    
    def _calculate_epa_based_adjustment(self, home_data, away_data):
        """Use EPA metrics to adjust win probability"""
        # Offensive matchup: home offense vs away defense
        offensive_matchup = home_data['off_epa_play'] - away_data['def_epa_play']
        
        # Defensive matchup: away offense vs home defense  
        defensive_matchup = away_data['off_epa_play'] - home_data['def_epa_play']
        
        # Net advantage
        net_advantage = (offensive_matchup - defensive_matchup) * 0.1  # Scale factor
        
        # QB advantage
        qb_advantage = (home_data['qb_adj'] - away_data['qb_adj']) * 0.005
        
        # Total adjustment (capped at ±0.1)
        total_adjustment = net_advantage + qb_advantage
        return max(-0.1, min(0.1, total_adjustment))
    
    def predict_game_score_enhanced(self, home_team, away_team, combined_win_prob, game_date):
        """Enhanced score prediction using ELO metrics"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None, None, None
        
        # Get ELO data for advanced scoring
        home_elo_data = self.elo_processor.get_team_elo(home_team)
        away_elo_data = self.elo_processor.get_team_elo(away_team)
        
        # Base scoring from team stats
        home_offense = self.team_stats[home_team][1]
        away_offense = self.team_stats[away_team][1]
        home_defense = self.team_stats[home_team][2]
        away_defense = self.team_stats[away_team][2]
        
        # Adjust for SOS if available
        home_sos = self.team_stats[home_team][3] if len(self.team_stats[home_team]) > 3 else 0.5
        away_sos = self.team_stats[away_team][3] if len(self.team_stats[away_team]) > 3 else 0.5
        
        # Base score prediction
        home_score = (home_offense + away_defense) / 2 + (combined_win_prob - 0.5) * 7 + (home_sos - 0.5) * 2
        away_score = (away_offense + home_defense) / 2 - (combined_win_prob - 0.5) * 7 + (away_sos - 0.5) * 2
        
        # ELO-based adjustments
        elo_adjustment = self._calculate_elo_scoring_adjustment(home_elo_data, away_elo_data)
        home_score += elo_adjustment['home_adjust']
        away_score += elo_adjustment['away_adjust']
        
        # EPA-based adjustments
        epa_adjustment = self._calculate_epa_scoring_adjustment(home_elo_data, away_elo_data)
        home_score += epa_adjustment['home_adjust']
        away_score += epa_adjustment['away_adjust']
        
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
    
    def _calculate_elo_scoring_adjustment(self, home_data, away_data):
        """Adjust scores based on ELO difference"""
        elo_diff = home_data['nfelo'] - away_data['nfelo']
        
        # ELO difference affects scoring margin
        if elo_diff > 100:
            return {'home_adjust': 2.0, 'away_adjust': -2.0}
        elif elo_diff > 50:
            return {'home_adjust': 1.0, 'away_adjust': -1.0}
        elif elo_diff < -100:
            return {'home_adjust': -2.0, 'away_adjust': 2.0}
        elif elo_diff < -50:
            return {'home_adjust': -1.0, 'away_adjust': 1.0}
        else:
            return {'home_adjust': 0, 'away_adjust': 0}
    
    def _calculate_epa_scoring_adjustment(self, home_data, away_data):
        """Adjust scores based on EPA efficiency"""
        # Teams with better offensive EPA score more
        home_off_boost = home_data['off_epa_play'] * 3
        away_off_boost = away_data['off_epa_play'] * 3
        
        # Teams with worse defensive EPA allow more points
        home_def_penalty = away_data['off_epa_play'] * 2  # Away offense vs home defense
        away_def_penalty = home_data['off_epa_play'] * 2  # Home offense vs away defense
        
        home_adjust = home_off_boost - home_def_penalty
        away_adjust = away_off_boost - away_def_penalty
        
        return {
            'home_adjust': home_adjust,
            'away_adjust': away_adjust
        }
    
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
    spread_diff = abs(model_spread - vegas_spread)
    if spread_diff <= 1:
        return 75
    elif spread_diff <= 3:
        return 65
    elif spread_diff <= 6:
        return 55
    else:
        return 50

def calculate_over_probability(model_total, vegas_total):
    """Calculate probability of over hitting"""
    total_diff = model_total - vegas_total
    if total_diff >= 3:
        return 70
    elif total_diff >= 1:
        return 60
    elif total_diff >= -1:
        return 50
    elif total_diff >= -3:
        return 40
    else:
        return 30

def create_game_card(predictor, game, game_odds, elo_prediction, home_score, away_score, weather_data):
    """Create a professional game card with clean design"""
    
    home_full = game['home']
    away_full = game['away']
    home_team = predictor.get_team_abbreviation(home_full)
    away_team = predictor.get_team_abbreviation(away_full)
    
    matchup_analysis = predictor.elo_processor.get_matchup_analysis(home_team, away_team)
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header with score and ELO info
        home_elo_data = predictor.elo_processor.get_team_elo(home_team)
        away_elo_data = predictor.elo_processor.get_team_elo(away_team)
        
        st.markdown(
            f'<div class="score-header">'
            f'🏈 {away_team} {int(away_score)} @ {home_team} {int(home_score)}<br>'
            f'<small style="font-size: 0.9rem; opacity: 0.9;">ELO Ratings: {away_team} {away_elo_data["nfelo"]:.0f} | {home_team} {home_elo_data["nfelo"]:.0f}</small>'
            f'</div>', 
            unsafe_allow_html=True
        )
        
        # Four Column Layout for Clean Organization
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            # Enhanced Projections Section
            st.markdown('<div class="projections-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Projections</div>', unsafe_allow_html=True)
            
            # Win Probability
            final_win_prob = elo_prediction['combined_win_prob']
            st.markdown('<div class="metric-item">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Win Probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{final_win_prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Model Weights
            st.markdown('<div class="metric-item">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Model Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{elo_prediction["confidence_level"].title()}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Key Insights
            st.markdown('<div style="margin-top: 1rem;"><strong>Key Insights:</strong></div>', unsafe_allow_html=True)
            
            if matchup_analysis['offensive_matchup'] > 0.1:
                st.markdown('<div class="insight-item">✅ Offensive Advantage</div>', unsafe_allow_html=True)
            elif matchup_analysis['offensive_matchup'] < -0.1:
                st.markdown('<div class="insight-item">✅ Defensive Advantage</div>', unsafe_allow_html=True)
                
            if matchup_analysis['qb_advantage'] > 5:
                st.markdown('<div class="insight-item">🎯 QB Advantage</div>', unsafe_allow_html=True)
            elif matchup_analysis['qb_advantage'] < -5:
                st.markdown('<div class="insight-item">🎯 QB Disadvantage</div>', unsafe_allow_html=True)
                
            if matchup_analysis['momentum_advantage'] > 0.5:
                st.markdown('<div class="insight-item">📈 Positive Momentum</div>', unsafe_allow_html=True)
            elif matchup_analysis['momentum_advantage'] < -0.5:
                st.markdown('<div class="insight-item">📉 Negative Momentum</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close projections-section
        
        with col2:
            # Efficiency Metrics Section
            st.markdown('<div class="efficiency-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⚡ Efficiency</div>', unsafe_allow_html=True)
            
            # Offensive EPA
            st.markdown('<div class="team-comparison">', unsafe_allow_html=True)
            st.markdown(f'<div class="team-name">{away_team}</div><div class="comparison-value">{away_elo_data["off_epa_play"]:+.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-name">{home_team}</div><div class="comparison-value">{home_elo_data["off_epa_play"]:+.3f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Defensive EPA
            st.markdown('<div class="team-comparison">', unsafe_allow_html=True)
            st.markdown(f'<div class="team-name">{away_team}</div><div class="comparison-value">{away_elo_data["def_epa_play"]:+.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-name">{home_team}</div><div class="comparison-value">{home_elo_data["def_epa_play"]:+.3f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Net EPA
            st.markdown('<div class="team-comparison">', unsafe_allow_html=True)
            st.markdown(f'<div class="team-name">{away_team}</div><div class="comparison-value">{away_elo_data["net_epa"]:+.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-name">{home_team}</div><div class="comparison-value">{home_elo_data["net_epa"]:+.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent Trends
            st.markdown('<div style="margin-top: 1rem;"><strong>Recent Trends:</strong></div>', unsafe_allow_html=True)
            away_trend = "📈" if away_elo_data["wow_change"] > 0 else "📉" if away_elo_data["wow_change"] < 0 else "➡️"
            home_trend = "📈" if home_elo_data["wow_change"] > 0 else "📉" if home_elo_data["wow_change"] < 0 else "➡️"
            st.markdown(f'<div class="team-comparison"><div class="team-name">{away_team}</div><div class="comparison-value">{away_trend} {away_elo_data["wow_change"]:+.1f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-comparison"><div class="team-name">{home_team}</div><div class="comparison-value">{home_trend} {home_elo_data["wow_change"]:+.1f}</div></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close efficiency-section
        
        with col3:
            # Final Projections Section
            st.markdown('<div class="final-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🏆 Picks</div>', unsafe_allow_html=True)
            
            # Winner Pick
            model_winner = home_team if final_win_prob > 0.5 else away_team
            win_prob_pct = final_win_prob * 100 if final_win_prob > 0.5 else (1 - final_win_prob) * 100
            
            st.markdown('<div class="metric-item">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Winner</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{model_winner}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="probability-badge" style="margin-top: 0.5rem;">{win_prob_pct:.0f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Spread Pick
            if game_odds and game_odds.get('spread'):
                vegas_spread = game_odds['spread']
                model_spread = predictor.convert_prob_to_spread(final_win_prob)
                
                # Determine ATS pick
                if vegas_spread < 0:  # Home team favored
                    if model_spread <= vegas_spread:
                        ats_pick = f"{home_team} to cover"
                    else:
                        ats_pick = f"{away_team} to cover"
                else:  # Away team favored
                    if model_spread >= vegas_spread:
                        ats_pick = f"{away_team} to cover"
                    else:
                        ats_pick = f"{home_team} to cover"
                
                cover_prob = calculate_cover_probability(model_spread, vegas_spread)
                
                st.markdown('<div class="metric-item">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Spread</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{ats_pick}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-badge" style="margin-top: 0.5rem;">{cover_prob}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Totals Pick
            if game_odds and game_odds.get('total'):
                vegas_total = game_odds['total']
                model_total = predictor.predict_total_points(home_team, away_team)
                
                if model_total > vegas_total:
                    totals_pick = "Over"
                else:
                    totals_pick = "Under"
                
                over_prob = calculate_over_probability(model_total, vegas_total)
                if totals_pick == "Under":
                    over_prob = 100 - over_prob
                
                st.markdown('<div class="metric-item">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{totals_pick}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-badge" style="margin-top: 0.5rem;">{over_prob}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close final-section
        
        with col4:
            # Odds Section
            st.markdown('<div class="odds-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">💰 Vegas Odds</div>', unsafe_allow_html=True)
            
            if game_odds:
                # Moneyline
                if game_odds.get('home_moneyline') and game_odds.get('away_moneyline'):
                    st.markdown('<div class="metric-item">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Moneyline</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="team-comparison"><div class="team-name">{home_team}</div><div class="odds-value">{game_odds["home_moneyline"]:+d}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="team-comparison"><div class="team-name">{away_team}</div><div class="odds-value">{game_odds["away_moneyline"]:+d}</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Spread
                if game_odds.get('spread'):
                    st.markdown('<div class="metric-item">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Spread</div>', unsafe_allow_html=True)
                    spread_display = f"{game_odds['spread']:+.1f}"
                    st.markdown(f'<div class="metric-value">{spread_display}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Total
                if game_odds.get('total'):
                    st.markdown('<div class="metric-item">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Points</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{game_odds["total"]:.1f}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-item">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Odds</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-value" style="color: #6B7280;">Not Available</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close odds-section
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close game-card

def main():
    # Header
    st.markdown('<div class="main-header">NFL Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Week 10 • Enhanced with ELO & Efficiency Metrics</div>', unsafe_allow_html=True)
    
    # Check for required files
    missing_files = check_required_files()
    if missing_files:
        st.error(f"❌ Missing required data files: {', '.join(missing_files)}")
        st.info("Please ensure all data files are available in the correct directory.")
        return
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Load model and data
    if not predictor.load_model():
        st.error("Failed to load model")
        st.stop()
        
    if not predictor.load_data():
        st.error("Failed to load data files")
        st.stop()
    
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
        
        # Use ENHANCED ELO prediction
        elo_prediction = predictor.predict_game_with_elo_enhanced(home_team, away_team)
        
        if elo_prediction is None:
            st.warning(f"Could not generate prediction for {home_team} vs {away_team}")
            continue
        
        # Get ENHANCED scores with ELO adjustments
        home_score, away_score, weather_data = predictor.predict_game_score_enhanced(
            home_team, away_team, elo_prediction['combined_win_prob'], game_date
        )
        
        # Create professional game card
        create_game_card(predictor, game, game_odds, elo_prediction, home_score, away_score, weather_data)

if __name__ == "__main__":
    main()
