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
                    elif market == 'spreads' and point:
                        # Determine spread direction based on which team is which
                        if home_team == player_team:
                            context['spread'] = -self._safe_float(point, 0.0)  # Home team favored
                        else:
                            context['spread'] = self._safe_float(point, 0.0)   # Away team underdog
                        found_odds = True
        
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
    
    def get_players_playing_this_week(self):
        """Get all players who are playing this week based on schedule"""
        playing_players = {
            'rushers': [],
            'passers': []
        }
        
        if not self.schedule_data:
            st.warning("No schedule data available")
            return playing_players
        
        # Extract all teams playing this week
        playing_teams = set()
        for game in self.schedule_data:
            if 'home_team' in game and 'away_team' in game:
                playing_teams.add(game['home_team'])
                playing_teams.add(game['away_team'])
        
        # Get matchups for each team
        team_matchups = {}
        for game in self.schedule_data:
            if 'home_team' in game and 'away_team' in game:
                team_matchups[game['home_team']] = game['away_team']
                team_matchups[game['away_team']] = game['home_team']
        
        # Find rushers playing this week
        if self.rb_data is not None:
            for _, player in self.rb_data.iterrows():
                if player['Team'] in playing_teams:
                    playing_players['rushers'].append({
                        'name': player['PlayerName'],
                        'team': player['Team'],
                        'opponent': team_matchups.get(player['Team'], 'Unknown'),
                        'position': 'RB'
                    })
        
        # Find QBs playing this week (both for passing and rushing)
        if self.qb_data is not None:
            for _, player in self.qb_data.iterrows():
                if player['Team'] in playing_teams:
                    playing_players['passers'].append({
                        'name': player['PlayerName'],
                        'team': player['Team'],
                        'opponent': team_matchups.get(player['Team'], 'Unknown'),
                        'position': 'QB'
                    })
                    # Also add QBs to rushers list for rushing projections
                    playing_players['rushers'].append({
                        'name': player['PlayerName'],
                        'team': player['Team'],
                        'opponent': team_matchups.get(player['Team'], 'Unknown'),
                        'position': 'QB'
                    })
        
        return playing_players
    
    def calculate_bet_grade(self, projection, vegas_line, stat_type):
        """Calculate a bet grade from 0-100 based on projection vs vegas line"""
        difference = abs(projection - vegas_line)
        
        # Different thresholds for different stat types
        if stat_type in ['rushing_yards', 'passing_yards']:
            if difference >= 20:
                return 95
            elif difference >= 15:
                return 90
            elif difference >= 10:
                return 85
            elif difference >= 7:
                return 80
            elif difference >= 5:
                return 75
            elif difference >= 3:
                return 70
            else:
                return 65
        elif stat_type in ['rushing_tds', 'passing_tds']:
            if difference >= 0.8:
                return 95
            elif difference >= 0.6:
                return 90
            elif difference >= 0.4:
                return 85
            elif difference >= 0.3:
                return 80
            elif difference >= 0.2:
                return 75
            elif difference >= 0.1:
                return 70
            else:
                return 65
        else:
            # Default grading
            if difference >= 3:
                return 90
            elif difference >= 2:
                return 80
            elif difference >= 1:
                return 70
            else:
                return 60
    
    def generate_all_projections(self, games_played=9):
        """Generate projections for all players playing this week"""
        playing_players = self.get_players_playing_this_week()
        all_projections = []
        
        # Generate rushing projections
        for player_info in playing_players['rushers']:
            try:
                projections, player_stats, defense_stats, game_context, position = self.project_rushing_stats(
                    player_info['name'], player_info['opponent'], games_played
                )
                
                # Calculate Vegas lines
                vegas_rush_yds = self.estimate_vegas_line(projections['RushingYards'], position, 'rushing_yards')
                vegas_rush_tds = self.estimate_vegas_line(projections['RushingTDs'], position, 'rushing_tds')
                
                # Calculate bet grades
                rush_yds_grade = self.calculate_bet_grade(projections['RushingYards'], vegas_rush_yds, 'rushing_yards')
                rush_tds_grade = self.calculate_bet_grade(projections['RushingTDs'], vegas_rush_tds, 'rushing_tds')
                
                # Use the higher grade
                bet_grade = max(rush_yds_grade, rush_tds_grade)
                
                # Determine if over or under
                if projections['RushingYards'] > vegas_rush_yds or projections['RushingTDs'] > vegas_rush_tds:
                    bet_direction = "↑"  # Over
                    primary_line = f"{vegas_rush_yds} Rush Yds" if rush_yds_grade >= rush_tds_grade else f"{vegas_rush_tds} Rush TDs"
                else:
                    bet_direction = "↓"  # Under
                    primary_line = f"{vegas_rush_yds} Rush Yds" if rush_yds_grade >= rush_tds_grade else f"{vegas_rush_tds} Rush TDs"
                
                all_projections.append({
                    'Player': player_info['name'],
                    'Team': player_info['team'],
                    'Opponent': player_info['opponent'],
                    'Position': position,
                    'BetGrade': bet_grade,
                    'BetDirection': bet_direction,
                    'PrimaryLine': primary_line,
                    'RushingYards': projections['RushingYards'],
                    'RushingTDs': projections['RushingTDs'],
                    'VegasRushYds': vegas_rush_yds,
                    'VegasRushTDs': vegas_rush_tds,
                    'ProjectionDifferential': round(abs(projections['RushingYards'] - vegas_rush_yds), 1),
                    'Game': f"{player_info['team']} vs {player_info['opponent']}"
                })
                
            except Exception as e:
                continue
        
        # Generate passing projections for QBs
        for player_info in playing_players['passers']:
            try:
                projections, qb_stats, defense_stats, game_context = self.project_passing_stats(
                    player_info['name'], player_info['opponent'], games_played
                )
                
                # Calculate Vegas lines
                vegas_pass_yds = self.estimate_vegas_line(projections['PassingYards'], 'QB', 'passing_yards')
                vegas_pass_tds = self.estimate_vegas_line(projections['PassingTDs'], 'QB', 'passing_tds')
                
                # Calculate bet grades
                pass_yds_grade = self.calculate_bet_grade(projections['PassingYards'], vegas_pass_yds, 'passing_yards')
                pass_tds_grade = self.calculate_bet_grade(projections['PassingTDs'], vegas_pass_tds, 'passing_tds')
                
                # Use the higher grade
                bet_grade = max(pass_yds_grade, pass_tds_grade)
                
                # Determine if over or under
                if projections['PassingYards'] > vegas_pass_yds or projections['PassingTDs'] > vegas_pass_tds:
                    bet_direction = "↑"  # Over
                    primary_line = f"{vegas_pass_yds} Pass Yds" if pass_yds_grade >= pass_tds_grade else f"{vegas_pass_tds} Pass TDs"
                else:
                    bet_direction = "↓"  # Under
                    primary_line = f"{vegas_pass_yds} Pass Yds" if pass_yds_grade >= pass_tds_grade else f"{vegas_pass_tds} Pass TDs"
                
                all_projections.append({
                    'Player': player_info['name'],
                    'Team': player_info['team'],
                    'Opponent': player_info['opponent'],
                    'Position': 'QB',
                    'BetGrade': bet_grade,
                    'BetDirection': bet_direction,
                    'PrimaryLine': primary_line,
                    'PassingYards': projections['PassingYards'],
                    'PassingTDs': projections['PassingTDs'],
                    'VegasPassYds': vegas_pass_yds,
                    'VegasPassTDs': vegas_pass_tds,
                    'ProjectionDifferential': round(abs(projections['PassingYards'] - vegas_pass_yds), 1),
                    'Game': f"{player_info['team']} vs {player_info['opponent']}"
                })
                
            except Exception as e:
                continue
        
        # Sort by bet grade (highest first)
        return sorted(all_projections, key=lambda x: x['BetGrade'], reverse=True)

def create_player_card(player_data):
    """Create a card for a player similar to the reference image"""
    grade_color = "#00ff00" if player_data['BetGrade'] >= 85 else "#ffff00" if player_data['BetGrade'] >= 75 else "#ff4444"
    
    card_html = f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div style="flex: 1;">
            <div style="font-weight: bold; font-size: 16px;">
                {player_data['Player']} - {player_data['Position']}
            </div>
            <div style="color: #666; font-size: 14px; margin-top: 5px;">
                {player_data['BetDirection']} {player_data['PrimaryLine']}
            </div>
            <div style="color: #666; font-size: 12px; margin-top: 2px;">
                {player_data['Game']}
            </div>
        </div>
        
        <div style="text-align: center; margin: 0 20px;">
            <div style="
                background: {grade_color};
                color: white;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 18px;
                margin: 0 auto;
            ">
                {player_data['BetGrade']}
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                Bet Grade
            </div>
        </div>
        
        <div style="text-align: right; flex: 1;">
            <div style="font-size: 12px; color: #666;">
                Market Line
            </div>
            <div style="font-weight: bold; font-size: 14px;">
                {player_data['PrimaryLine'].split(' ')[-2]} {player_data['PrimaryLine'].split(' ')[-1]}
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 8px;">
                Proj Differential
            </div>
            <div style="font-weight: bold; font-size: 14px; color: {'#00aa00' if player_data['BetDirection'] == '↑' else '#aa0000'}">
                +{player_data['ProjectionDifferential']}
            </div>
        </div>
    </div>
    """
    return card_html

def main():
    st.set_page_config(
        page_title="NFL Player Projections",
        page_icon="🏈", 
        layout="wide"
    )
    
    st.title("🏈 NFL Player Projections")
    
    # Initialize projector
    if 'projector' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.projector = EnhancedNFLProjector()
    
    projector = st.session_state.projector
    
    # Create the card-based interface
    st.markdown("### Player Projections for This Week")
    
    # Filters at the top (similar to the reference image)
    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
    
    with col1:
        sport_filter = st.selectbox("SPORT", ["ALL ✓", "NFL"], key="sport_filter")
    
    with col2:
        position_filter = st.selectbox("POSITION", ["ALL ✓", "QB", "RB"], key="position_filter")
    
    with col3:
        team_filter = st.selectbox("TEAM MARKET", ["ALL ✓"] + projector.get_available_teams(), key="team_filter")
    
    with col4:
        show_count = st.selectbox("SHOW", ["50", "100", "ALL"], key="show_count")
    
    # Search bar
    search_query = st.text_input("🔍 Search players...", placeholder="Type to search players or teams")
    
    # Generate all projections button
    if st.button("🔄 Generate All Projections", type="primary"):
        with st.spinner("Generating projections for all players..."):
            all_projections = projector.generate_all_projections()
            st.session_state.all_projections = all_projections
    
    # Display projections in card format
    if 'all_projections' in st.session_state and st.session_state.all_projections:
        projections = st.session_state.all_projections
        
        # Apply filters
        filtered_projections = projections
        
        if position_filter != "ALL ✓":
            filtered_projections = [p for p in filtered_projections if p['Position'] == position_filter]
        
        if team_filter != "ALL ✓":
            filtered_projections = [p for p in filtered_projections if p['Team'] == team_filter]
        
        if search_query:
            filtered_projections = [p for p in filtered_projections 
                                 if search_query.lower() in p['Player'].lower() 
                                 or search_query.lower() in p['Team'].lower()
                                 or search_query.lower() in p['Opponent'].lower()]
        
        # Limit results
        if show_count != "ALL":
            filtered_projections = filtered_projections[:int(show_count)]
        
        # Display cards
        st.markdown(f"**Showing {len(filtered_projections)} players**")
        
        for player_data in filtered_projections:
            card_html = create_player_card(player_data)
            st.markdown(card_html, unsafe_allow_html=True)
    
    else:
        st.info("Click 'Generate All Projections' to see player projections for this week")

if __name__ == "__main__":
    main()
