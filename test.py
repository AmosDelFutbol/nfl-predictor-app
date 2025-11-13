import pandas as pd
import numpy as np
import json
import pickle
import streamlit as st

class EnhancedNFLProjector:
    def __init__(self):
        """Initialize the enhanced projector with all available data sources"""
        # Initialize all attributes first
        self.rb_data = None
        self.qb_data = None
        self.defense_data = None
        self.offense_data = None
        self.sos_data = None
        self.schedule_data = None
        self.odds_data = None
        self.nfl_model = None
        self.model_type = None
        
        try:
            # Load player data
            self.rb_data = pd.read_csv('RB_season.csv')
            self.qb_data = pd.read_csv('QB_season.csv')
            st.success("✅ Player data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading player data: {e}")
        
        # Load JSON data with error handling for each file
        self._load_json_data()
        
        # Load ML model
        self._load_ml_model()
        
        # Clean the data - fill NaN values
        self._clean_data()
    
    def _load_json_data(self):
        """Load JSON data files with individual error handling"""
        # Load defense data
        try:
            with open('2025_NFL_DEFENSE.json', 'r') as f:
                self.defense_data = json.load(f)
            st.success("✅ Defense data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load defense data: {e}")
            self.defense_data = []
        
        # Load offense data
        try:
            with open('2025_NFL_OFFENSE.json', 'r') as f:
                self.offense_data = json.load(f)
            st.success("✅ Offense data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load offense data: {e}")
            self.offense_data = []
        
        # Load strength of schedule data
        try:
            with open('nfl_strength_of_schedule.json', 'r') as f:
                data = json.load(f)
                self.sos_data = data.get('sos_rankings', {})
            st.success("✅ Strength of schedule data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load strength of schedule data: {e}")
            self.sos_data = {}
        
        # Load schedule data
        try:
            with open('schedule.json', 'r') as f:
                schedule_data = json.load(f)
                if 'Week 10' in schedule_data:
                    self.schedule_data = schedule_data['Week 10']
                else:
                    self.schedule_data = schedule_data
            st.success("✅ Schedule data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load schedule data: {e}")
            self.schedule_data = []
        
        # Load odds data - FIXED: Better debugging for odds data
        try:
            with open('odds.json', 'r') as f:
                self.odds_data = json.load(f)
            st.success("✅ Odds data loaded successfully!")
            st.info(f"📊 Loaded odds for {len(self.odds_data)} games")
        except Exception as e:
            st.warning(f"⚠️ Could not load odds data: {e}")
            self.odds_data = []
    
    def _load_ml_model(self):
        """Load the trained NFL model with better type detection"""
        try:
            with open('nfl_model.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            
            if hasattr(loaded_data, 'predict'):
                self.nfl_model = loaded_data
                self.model_type = 'sklearn'
                st.success("✅ ML model (scikit-learn) loaded successfully!")
            elif isinstance(loaded_data, dict):
                self.nfl_model = loaded_data
                self.model_type = 'dict'
                st.success("✅ Model parameters (dict) loaded successfully!")
            else:
                st.warning(f"⚠️ Unknown model type: {type(loaded_data)}")
                self.nfl_model = None
                self.model_type = None
                
        except Exception as e:
            st.warning(f"⚠️ Could not load ML model: {e}")
            self.nfl_model = None
            self.model_type = None
    
    def _clean_data(self):
        """Clean the data by filling NaN values with appropriate defaults"""
        if self.rb_data is not None:
            rb_numeric_columns = ['RushingYDS', 'RushingTD', 'TouchCarries', 'ReceivingYDS', 'ReceivingRec', 'ReceivingTD']
            for col in rb_numeric_columns:
                if col in self.rb_data.columns:
                    self.rb_data[col] = self.rb_data[col].fillna(0)
        
        if self.qb_data is not None:
            qb_numeric_columns = ['PassingYDS', 'PassingTD', 'PassingInt', 'RushingYDS', 'RushingTD']
            for col in qb_numeric_columns:
                if col in self.qb_data.columns:
                    self.qb_data[col] = self.qb_data[col].fillna(0)
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling NaN and None"""
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _get_defense_stats(self, team_name):
        """Get defense stats for a specific team"""
        if self.defense_data is None or not self.defense_data:
            return None
        for team in self.defense_data:
            if team['Team'] == team_name:
                return team
        return None
    
    def get_ml_prediction(self, player_name, opponent_team, position, stat_type):
        """Get ML model prediction if available"""
        if self.nfl_model is None:
            return None
        
        try:
            features = self._prepare_ml_features(player_name, opponent_team, position, stat_type)
            
            if features is not None and features.size > 0:
                if self.model_type == 'sklearn':
                    prediction = self.nfl_model.predict(features)
                    return prediction[0] if hasattr(prediction, '__len__') else prediction
                elif self.model_type == 'dict':
                    return self._predict_from_dict(features, position, stat_type)
                else:
                    return None
            else:
                return None
        except Exception as e:
            st.warning(f"ML prediction failed: {e}")
            return None
    
    def _predict_from_dict(self, features, position, stat_type):
        """Custom prediction logic for dictionary-based models - FIXED SCALING"""
        try:
            feature_vector = features.flatten()
            
            # Simple weighted average prediction with BETTER SCALING
            if self.nfl_model.get('model_type') == 'linear':
                weights = self.nfl_model.get('weights', [1.0] * len(feature_vector))
                bias = self.nfl_model.get('bias', 0.0)
                
                if len(weights) == len(feature_vector):
                    prediction = np.dot(feature_vector, weights) + bias
                    
                    # Apply REALISTIC position-specific adjustments
                    if position == 'QB' and stat_type == 'passing_yards':
                        prediction = max(150, min(350, prediction))  # More realistic range
                    elif position == 'RB' and stat_type == 'rushing_yards':
                        prediction = max(20, min(120, prediction))   # More realistic range
                    
                    return prediction
            
            # Fallback: simple average of features with BETTER position-based scaling
            avg_prediction = np.mean(feature_vector)
            
            # FIXED: Much more realistic scaling factors
            if position == 'QB':
                if stat_type == 'passing_yards':
                    return avg_prediction * 15 + 200   # More reasonable scaling
                elif stat_type == 'rushing_yards':
                    return avg_prediction * 3 + 10     # More reasonable scaling
            elif position == 'RB':
                if stat_type == 'rushing_yards':
                    return avg_prediction * 5 + 40     # More reasonable scaling
            
            return avg_prediction * 8 + 50  # More reasonable default scaling
            
        except Exception as e:
            st.warning(f"Dictionary prediction failed: {e}")
            return None
    
    def _prepare_ml_features(self, player_name, opponent_team, position, stat_type):
        """Prepare features for ML model prediction"""
        try:
            features = []
            
            # Get player stats
            player_stats = None
            if position == 'RB' and self.rb_data is not None:
                player_stats_df = self.rb_data[self.rb_data['PlayerName'] == player_name]
                if not player_stats_df.empty:
                    player_stats = player_stats_df.iloc[0]
            elif position == 'QB' and self.qb_data is not None:
                player_stats_df = self.qb_data[self.qb_data['PlayerName'] == player_name]
                if not player_stats_df.empty:
                    player_stats = player_stats_df.iloc[0]
            
            if player_stats is None:
                return None
            
            # Get defense stats
            defense_stats = self._get_defense_stats(opponent_team)
            if defense_stats is None:
                return None
            
            # Get game context
            player_team = player_stats['Team']
            game_context = self.get_game_context(player_team, opponent_team)
            
            # BASIC FEATURES - FIXED: Better normalization
            if position == 'RB':
                features.extend([
                    self._safe_float(player_stats.get('RushingYDS', 0)) / 100,  # Normalized
                    self._safe_float(player_stats.get('RushingTD', 0)) / 10,    # Normalized
                    self._safe_float(player_stats.get('TouchCarries', 0)) / 20, # Normalized
                ])
            elif position == 'QB':
                features.extend([
                    self._safe_float(player_stats.get('PassingYDS', 0)) / 300,  # Normalized
                    self._safe_float(player_stats.get('PassingTD', 0)) / 10,    # Normalized
                    self._safe_float(player_stats.get('RushingYDS', 0)) / 50,   # Normalized
                ])
            
            # Defense quality features - FIXED: Better normalization
            features.extend([
                self._safe_float(defense_stats.get('RUSHING YARDS PER GAME ALLOWED', 100)) / 150,
                self._safe_float(defense_stats.get('PASSING YARDS ALLOWED', 230)) / 300,
            ])
            
            # Game context features
            features.extend([
                game_context['expected_total'] / 50.0,
                (game_context['spread'] + 14) / 28.0,   # Normalized -14 to +14 spread
                game_context['sos_adjustment']
            ])
            
            if len(features) == 0:
                return None
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            st.warning(f"Feature preparation failed: {e}")
            return None
    
    def get_game_context(self, player_team, opponent_team):
        """Get game context including odds and SOS - FIXED ODDS PARSING"""
        context = {
            'expected_total': 45.0,
            'spread': 0.0,
            'sos_adjustment': 1.0
        }
        
        # Get SOS adjustment
        if self.sos_data and player_team in self.sos_data:
            sos_value = self.sos_data[player_team].get('combined_sos', 0.5)
            context['sos_adjustment'] = 1.0 + (self._safe_float(sos_value) - 0.5) * 0.3
        
        # FIXED: Better odds data parsing with debugging
        if hasattr(self, 'odds_data') and self.odds_data is not None:
            found_odds = False
            for odds in self.odds_data:
                # Debug: show what we're looking at
                home_team = odds.get('home_team', '')
                away_team = odds.get('away_team', '')
                market = odds.get('market', '')
                point = odds.get('point', '')
                
                # Check if this is the right game
                if (home_team == player_team and away_team == opponent_team) or \
                   (home_team == opponent_team and away_team == player_team):
                    
                    if market == 'totals' and point:
                        context['expected_total'] = self._safe_float(point, 45.0)
                        found_odds = True
                        st.sidebar.success(f"📊 Found totals: {point} for {home_team} vs {away_team}")
                    elif market == 'spreads' and point:
                        # Determine spread direction based on which team is which
                        if home_team == player_team:
                            context['spread'] = -self._safe_float(point, 0.0)  # Home team favored
                        else:
                            context['spread'] = self._safe_float(point, 0.0)   # Away team underdog
                        found_odds = True
                        st.sidebar.success(f"📊 Found spread: {point} for {home_team} vs {away_team}")
            
            if not found_odds:
                st.sidebar.warning(f"⚠️ No odds found for {player_team} vs {opponent_team}")
                st.sidebar.info(f"Looking for teams: {player_team} and {opponent_team}")
        
        return context
    
    def estimate_vegas_line(self, projection, position, stat_type):
        """Estimate what the Vegas line might have been based on projection"""
        safe_projection = self._safe_float(projection, 0.0)
        
        if position == 'RB':
            if stat_type == 'rushing_yards':
                estimated_line = round(safe_projection / 5) * 5
                return max(45, min(125, estimated_line))
            elif stat_type == 'rushing_tds':
                estimated_line = round(safe_projection * 2) / 2
                return max(0.5, min(2.5, estimated_line))
            elif stat_type == 'receiving_yards':
                estimated_line = round(safe_projection / 5) * 5
                return max(15, min(80, estimated_line))
            elif stat_type == 'receptions':
                estimated_line = round(safe_projection * 2) / 2
                return max(2.5, min(8.5, estimated_line))
        
        elif position == 'QB':
            if stat_type == 'passing_yards':
                estimated_line = round(safe_projection / 10) * 10
                return max(180, min(350, estimated_line))
            elif stat_type == 'passing_tds':
                estimated_line = round(safe_projection * 2) / 2
                return max(0.5, min(3.5, estimated_line))
            elif stat_type == 'rushing_yards':
                estimated_line = round(safe_projection)
                return max(10, min(60, estimated_line))
            elif stat_type == 'interceptions':
                estimated_line = round(safe_projection * 2) / 2
                return max(0.5, min(2.5, estimated_line))
        
        return round(safe_projection)
    
    def project_rushing_stats(self, player_name, opponent_team, games_played=9):
        """Enhanced rushing projections for RBs and rushing QBs"""
        if self.rb_data is None or self.qb_data is None:
            raise ValueError("Player data not loaded properly")
        
        rb_row = self.rb_data[self.rb_data['PlayerName'] == player_name]
        if not rb_row.empty:
            return self._project_rb_rushing(rb_row.iloc[0], opponent_team, games_played)
        
        qb_row = self.qb_data[self.qb_data['PlayerName'] == player_name]
        if not qb_row.empty:
            return self._project_qb_rushing(qb_row.iloc[0], opponent_team, games_played)
        
        raise ValueError(f"Player '{player_name}' not found in RB or QB data")
    
    def _project_rb_rushing(self, rb_stats, opponent_team, games_played):
        """Project rushing stats for running backs"""
        player_team = rb_stats['Team']
        
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            raise ValueError(f"Defense stats for '{opponent_team}' not found")
        
        game_context = self.get_game_context(player_team, opponent_team)
        
        projections = {}
        
        rb_rush_yds_pg = self._safe_float(rb_stats['RushingYDS']) / games_played
        def_rush_yds_allowed = self._safe_float(defense_stats.get('RUSHING YARDS PER GAME ALLOWED', 100))
        
        game_script = 1.15 if game_context['spread'] > 3 else 0.85 if game_context['spread'] < -3 else 1.0
        
        ml_prediction = self.get_ml_prediction(rb_stats['PlayerName'], opponent_team, 'RB', 'rushing_yards')
        
        if ml_prediction is not None:
            projections['RushingYards'] = ml_prediction
            projections['UsedML'] = True
        else:
            projections['RushingYards'] = (
                (rb_rush_yds_pg + def_rush_yds_allowed) / 2 * 
                game_script * game_context['sos_adjustment']
            )
            projections['UsedML'] = False
        
        rb_rush_td_pg = self._safe_float(rb_stats['RushingTD']) / games_played
        def_rush_td_allowed = self._safe_float(defense_stats.get('RUSHING TD PER GAME ALLOWED', 1.0))
        
        projections['RushingTDs'] = (
            (rb_rush_td_pg + def_rush_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        rb_carries_pg = self._safe_float(rb_stats['TouchCarries']) / games_played
        def_rush_attempts_allowed = self._safe_float(defense_stats.get('RUSHING ATTEMPTS ALLOWED', 25))
        
        projections['Carries'] = (
            (rb_carries_pg + def_rush_attempts_allowed * 0.3) / 1.3 * game_script
        )
        
        projections['FantasyPoints'] = (
            projections['RushingYards'] * 0.1 +
            projections['RushingTDs'] * 6
        )
        
        return projections, rb_stats, defense_stats, game_context, 'RB'
    
    def _project_qb_rushing(self, qb_stats, opponent_team, games_played):
        """Project rushing stats for quarterbacks"""
        player_team = qb_stats['Team']
        
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            raise ValueError(f"Defense stats for '{opponent_team}' not found")
        
        game_context = self.get_game_context(player_team, opponent_team)
        
        projections = {}
        
        qb_rush_yds_pg = self._safe_float(qb_stats['RushingYDS']) / games_played
        
        ml_prediction = self.get_ml_prediction(qb_stats['PlayerName'], opponent_team, 'QB', 'rushing_yards')
        
        if ml_prediction is not None:
            projections['RushingYards'] = ml_prediction
            projections['UsedML'] = True
        else:
            projections['RushingYards'] = qb_rush_yds_pg * game_context['sos_adjustment']
            projections['UsedML'] = False
        
        qb_rush_td_pg = self._safe_float(qb_stats['RushingTD']) / games_played
        
        projections['RushingTDs'] = qb_rush_td_pg * (game_context['expected_total'] / 45.0)
        
        projections['Carries'] = projections['RushingYards'] / 4.5
        
        projections['FantasyPoints'] = (
            projections['RushingYards'] * 0.1 +
            projections['RushingTDs'] * 6
        )
        
        return projections, qb_stats, defense_stats, game_context, 'QB'
    
    def project_passing_stats(self, qb_name, opponent_team, games_played=9):
        """Enhanced QB passing projections"""
        if self.qb_data is None:
            raise ValueError("QB data not loaded properly")
            
        qb_row = self.qb_data[self.qb_data['PlayerName'] == qb_name]
        if qb_row.empty:
            raise ValueError(f"QB '{qb_name}' not found")
        
        qb_stats = qb_row.iloc[0]
        player_team = qb_stats['Team']
        
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            raise ValueError(f"Defense stats for '{opponent_team}' not found")
        
        game_context = self.get_game_context(player_team, opponent_team)
        
        projections = {}
        
        qb_pass_yds_pg = self._safe_float(qb_stats['PassingYDS']) / games_played
        def_pass_yds_allowed = self._safe_float(defense_stats.get('PASSING YARDS ALLOWED', 230))
        
        ml_prediction = self.get_ml_prediction(qb_name, opponent_team, 'QB', 'passing_yards')
        
        if ml_prediction is not None:
            # FIXED: Use more reasonable ML prediction or fallback
            if ml_prediction > 400:  # If prediction is unrealistic
                st.warning("ML prediction seems unrealistic, using statistical calculation")
                projections['PassingYards'] = (
                    (qb_pass_yds_pg + def_pass_yds_allowed) / 2 *
                    (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
                )
                projections['UsedML'] = False
            else:
                projections['PassingYards'] = ml_prediction
                projections['UsedML'] = True
        else:
            projections['PassingYards'] = (
                (qb_pass_yds_pg + def_pass_yds_allowed) / 2 *
                (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
            )
            projections['UsedML'] = False
        
        qb_pass_td_pg = self._safe_float(qb_stats['PassingTD']) / games_played
        def_pass_td_allowed = self._safe_float(defense_stats.get('PASSING TD ALLOWED', 1.5))
        
        projections['PassingTDs'] = (
            (qb_pass_td_pg + def_pass_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        qb_int_pg = self._safe_float(qb_stats['PassingInt']) / games_played
        
        projections['Interceptions'] = (
            (qb_int_pg + self._safe_float(defense_stats.get('INTERCEPTIONS', 1.0))) / 2
        )
        
        def_attempts_allowed = self._safe_float(defense_stats.get('PASSING ATTEMPTS ALLOWED', 35))
        def_completions_allowed = self._safe_float(defense_stats.get('PASSING COMPLETIONS ALLOWED', 23))
        
        projections['PassAttempts'] = def_attempts_allowed * (game_context['expected_total'] / 45.0)
        defense_completion_pct = def_completions_allowed / def_attempts_allowed if def_attempts_allowed > 0 else 0.65
        projections['Completions'] = projections['PassAttempts'] * defense_completion_pct
        
        projections['FantasyPoints'] = (
            projections['PassingYards'] * 0.04 +
            projections['PassingTDs'] * 4 -
            projections['Interceptions'] * 2
        )
        
        return projections, qb_stats, defense_stats, game_context
    
    def get_available_rushers(self):
        """Get all players with rushing stats in alphabetical order"""
        if self.rb_data is None or self.qb_data is None:
            return []
        rushers = self.rb_data['PlayerName'].tolist() + self.qb_data['PlayerName'].tolist()
        return sorted(list(set(rushers)))
    
    def get_available_passers(self):
        """Get all quarterbacks in alphabetical order"""
        if self.qb_data is None:
            return []
        return sorted(self.qb_data['PlayerName'].tolist())
    
    def get_available_teams(self):
        """Get available teams from defense data in alphabetical order"""
        teams = set()
        if self.defense_data:
            for team in self.defense_data:
                if 'Team' in team:
                    teams.add(team['Team'])
        return sorted(list(teams))

def main():
    st.set_page_config(
        page_title="NFL Player Projections",
        page_icon="🏈", 
        layout="wide"
    )
    
    st.title("🏈 Enhanced NFL Player Projections")
    st.markdown("### Advanced Projections with Defense Matchups & Game Context")
    
    # Initialize projector
    if 'projector' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.projector = EnhancedNFLProjector()
    
    projector = st.session_state.projector
    
    # Display model info
    if projector.model_type:
        st.sidebar.info(f"🤖 Model Type: {projector.model_type.upper()}")
    
    # Debug odds data
    if projector.odds_data:
        st.sidebar.info(f"📊 Odds data: {len(projector.odds_data)} games loaded")
        if st.sidebar.checkbox("Show Odds Debug Info"):
            st.sidebar.write("First 3 odds entries:")
            for i, odds in enumerate(projector.odds_data[:3]):
                st.sidebar.write(f"{i+1}. {odds.get('home_team')} vs {odds.get('away_team')} - {odds.get('market')}: {odds.get('point')}")
    
    # Create tabs for different stats
    tab1, tab2 = st.tabs(["🏃‍♂️ Rushing", "🎯 Passing"])
    
    with tab1:
        st.subheader("Rushing Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_rushers = projector.get_available_rushers()
            if not available_rushers:
                st.error("No rusher data available. Please check your RB and QB CSV files.")
                rusher_name = st.selectbox("Select Rusher", [""], key="rusher")
            else:
                rusher_name = st.selectbox("Select Rusher", available_rushers, key="rusher")
            
            available_teams = projector.get_available_teams()
            if not available_teams:
                st.error("No team data available. Please check your defense JSON file.")
                opponent_team_rush = st.selectbox("Select Opponent Team", [""], key="rush_opponent")
            else:
                opponent_team_rush = st.selectbox("Select Opponent Team", available_teams, key="rush_opponent")
            
            games_played_rush = st.number_input("Games Played This Season", min_value=1, max_value=17, value=9, key="rush_games")
        
        with col2:
            if st.button("Generate Rushing Projection", type="primary", key="rush_btn"):
                try:
                    projections, player_stats, defense_stats, game_context, position = projector.project_rushing_stats(
                        rusher_name, opponent_team_rush, games_played_rush
                    )
                    
                    # Display results
                    st.success(f"📊 Rushing Projection for {rusher_name} ({position}) vs {opponent_team_rush}")
                    
                    # Game context
                    st.write(f"**Game Context:** Expected Total: {game_context['expected_total']} points, Spread: {game_context['spread']:+.1f}")
                    
                    # ML model info
                    if projections.get('UsedML'):
                        st.success("🤖 **ML Model**: Using machine learning prediction")
                    else:
                        st.info("📊 **ML Model**: Using statistical calculation")
                    
                    # Estimate Vegas lines
                    vegas_rush_yds = projector.estimate_vegas_line(projections['RushingYards'], position, 'rushing_yards')
                    vegas_rush_tds = projector.estimate_vegas_line(projections['RushingTDs'], position, 'rushing_tds')
                    
                    # Projections in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rushing Yards", f"{projections['RushingYards']:.1f}")
                        st.metric("Rushing TDs", f"{projections['RushingTDs']:.1f}")
                        st.metric("Estimated Vegas Line", f"{vegas_rush_yds} yards")
                    
                    with col2:
                        st.metric("Carries", f"{projections['Carries']:.1f}")
                        st.metric("Fantasy Points", f"{projections['FantasyPoints']:.1f}")
                        st.metric("TDs Vegas Line", f"{vegas_rush_tds}")
                    
                    with col3:
                        st.metric("Team", player_stats['Team'])
                        if position == 'RB':
                            st.metric("Season Rush Yds", int(projector._safe_float(player_stats['RushingYDS'])))
                        else:
                            st.metric("Season Rush Yds", int(projector._safe_float(player_stats['RushingYDS'])))
                    
                    # Analysis section
                    st.subheader("📈 Vegas Line Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if projections['RushingYards'] > vegas_rush_yds:
                            st.success(f"📈 **OVER PLAY**: Projection ({projections['RushingYards']:.1f}) > Vegas ({vegas_rush_yds})")
                        else:
                            st.error(f"📉 **UNDER PLAY**: Projection ({projections['RushingYards']:.1f}) < Vegas ({vegas_rush_yds})")
                    
                    with col2:
                        if projections['RushingTDs'] > vegas_rush_tds:
                            st.success(f"📈 **OVER PLAY**: Projection ({projections['RushingTDs']:.1f}) > Vegas ({vegas_rush_tds})")
                        else:
                            st.error(f"📉 **UNDER PLAY**: Projection ({projections['RushingTDs']:.1f}) < Vegas ({vegas_rush_tds})")
                    
                    # Defense info
                    with st.expander("View Defense Stats"):
                        st.write(f"**{opponent_team_rush} Rush Defense Allowed Per Game:**")
                        st.write(f"- Rushing: {projector._safe_float(defense_stats.get('RUSHING YARDS PER GAME ALLOWED', 0))} yds, {projector._safe_float(defense_stats.get('RUSHING TD PER GAME ALLOWED', 0))} TDs")
                        
                except Exception as e:
                    st.error(f"Error generating projection: {e}")
                    st.info("💡 This error might be due to missing data for this player. Try selecting a different player.")
    
    with tab2:
        st.subheader("Passing Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_passers = projector.get_available_passers()
            if not available_passers:
                st.error("No QB data available. Please check your QB CSV file.")
                qb_name = st.selectbox("Select Quarterback", [""], key="qb")
            else:
                qb_name = st.selectbox("Select Quarterback", available_passers, key="qb")
            
            available_teams = projector.get_available_teams()
            if not available_teams:
                st.error("No team data available. Please check your defense JSON file.")
                opponent_team_qb = st.selectbox("Select Opponent Team", [""], key="qb_opponent")
            else:
                opponent_team_qb = st.selectbox("Select Opponent Team", available_teams, key="qb_opponent")
            
            games_played_qb = st.number_input("Games Played This Season", min_value=1, max_value=17, value=9, key="qb_games")
        
        with col2:
            if st.button("Generate Passing Projection", type="primary", key="pass_btn"):
                try:
                    projections, qb_stats, defense_stats, game_context = projector.project_passing_stats(
                        qb_name, opponent_team_qb, games_played_qb
                    )
                    
                    # Display results
                    st.success(f"📊 Passing Projection for {qb_name} vs {opponent_team_qb}")
                    
                    # Game context
                    st.write(f"**Game Context:** Expected Total: {game_context['expected_total']} points, Spread: {game_context['spread']:+.1f}")
                    
                    # ML model info
                    if projections.get('UsedML'):
                        st.success("🤖 **ML Model**: Using machine learning prediction")
                    else:
                        st.info("📊 **ML Model**: Using statistical calculation")
                    
                    # Estimate Vegas lines
                    vegas_pass_yds = projector.estimate_vegas_line(projections['PassingYards'], 'QB', 'passing_yards')
                    vegas_pass_tds = projector.estimate_vegas_line(projections['PassingTDs'], 'QB', 'passing_tds')
                    vegas_ints = projector.estimate_vegas_line(projections['Interceptions'], 'QB', 'interceptions')
                    
                    # Projections in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Passing Yards", f"{projections['PassingYards']:.1f}")
                        st.metric("Passing TDs", f"{projections['PassingTDs']:.1f}")
                        st.metric("Interceptions", f"{projections['Interceptions']:.1f}")
                    
                    with col2:
                        st.metric("Pass Attempts", f"{projections['PassAttempts']:.1f}")
                        st.metric("Completions", f"{projections['Completions']:.1f}")
                        st.metric("Fantasy Points", f"{projections['FantasyPoints']:.1f}")
                    
                    with col3:
                        st.metric("Team", qb_stats['Team'])
                        st.metric("Season Pass Yds", int(projector._safe_float(qb_stats['PassingYDS'])))
                        st.metric("Vegas Yards Line", f"{vegas_pass_yds}")
                    
                    # Vegas lines display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Vegas TDs Line", f"{vegas_pass_tds}")
                    with col2:
                        st.metric("Vegas INTs Line", f"{vegas_ints}")
                    
                    # Analysis section
                    st.subheader("📈 Vegas Line Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if projections['PassingYards'] > vegas_pass_yds:
                            st.success(f"📈 **OVER PLAY**: Projection ({projections['PassingYards']:.1f}) > Vegas ({vegas_pass_yds})")
                        else:
                            st.error(f"📉 **UNDER PLAY**: Projection ({projections['PassingYards']:.1f}) < Vegas ({vegas_pass_yds})")
                    
                    with col2:
                        if projections['PassingTDs'] > vegas_pass_tds:
                            st.success(f"📈 **OVER PLAY**: Projection ({projections['PassingTDs']:.1f}) > Vegas ({vegas_pass_tds})")
                        else:
                            st.error(f"📉 **UNDER PLAY**: Projection ({projections['PassingTDs']:.1f}) < Vegas ({vegas_pass_tds})")
                    
                    with col3:
                        if projections['Interceptions'] > vegas_ints:
                            st.success(f"📈 **OVER PLAY**: Projection ({projections['Interceptions']:.1f}) > Vegas ({vegas_ints})")
                        else:
                            st.error(f"📉 **UNDER PLAY**: Projection ({projections['Interceptions']:.1f}) < Vegas ({vegas_ints})")
                    
                    # Defense info
                    with st.expander("View Defense Stats"):
                        st.write(f"**{opponent_team_qb} Pass Defense Allowed Per Game:**")
                        st.write(f"- Passing: {projector._safe_float(defense_stats.get('PASSING YARDS ALLOWED', 0))} yds, {projector._safe_float(defense_stats.get('PASSING TD ALLOWED', 0))} TDs")
                        st.write(f"- Interceptions: {projector._safe_float(defense_stats.get('INTERCEPTIONS', 0)):.1f}")
                        
                except Exception as e:
                    st.error(f"Error generating projection: {e}")
                    st.info("💡 This error might be due to missing data for this player. Try selecting a different player.")

if __name__ == "__main__":
    main()
