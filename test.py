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
        
        # Load schedule data - IMPROVED PARSING
        try:
            with open('schedule.json', 'r') as f:
                schedule_data = json.load(f)
                
            # Debug: Show what the schedule data looks like
            st.info(f"📅 Raw schedule data type: {type(schedule_data)}")
            
            # Handle different schedule formats
            if isinstance(schedule_data, dict):
                st.info("📅 Schedule is a dictionary")
                # Try common keys
                if 'Week 10' in schedule_data:
                    self.schedule_data = schedule_data['Week 10']
                    st.success("✅ Found 'Week 10' in schedule data")
                elif 'week_10' in schedule_data:
                    self.schedule_data = schedule_data['week_10']
                    st.success("✅ Found 'week_10' in schedule data")
                elif 'games' in schedule_data:
                    self.schedule_data = schedule_data['games']
                    st.success("✅ Found 'games' in schedule data")
                else:
                    # Show available keys for debugging
                    st.info(f"📅 Available keys in schedule: {list(schedule_data.keys())}")
                    # Try to find any list structure
                    for key, value in schedule_data.items():
                        if isinstance(value, list):
                            self.schedule_data = value
                            st.success(f"✅ Using schedule data from key: {key}")
                            break
                    else:
                        # If no list found, use the entire dict as schedule
                        self.schedule_data = [schedule_data]
            elif isinstance(schedule_data, list):
                self.schedule_data = schedule_data
                st.success("✅ Schedule is a list")
            else:
                self.schedule_data = []
                st.warning("❓ Unknown schedule format")
            
            st.success(f"✅ Schedule data loaded! Found {len(self.schedule_data)} games")
            
            # Show first game for debugging
            if self.schedule_data and len(self.schedule_data) > 0:
                st.info(f"📅 First game sample: {self.schedule_data[0]}")
                
        except Exception as e:
            st.error(f"❌ Could not load schedule data: {e}")
            self.schedule_data = []
        
        # Load odds data
        try:
            with open('odds.json', 'r') as f:
                self.odds_data = json.load(f)
            st.success("✅ Odds data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load odds data: {e}")
            self.odds_data = []
    
    def _load_ml_model(self):
        """Load the trained NFL model"""
        try:
            with open('nfl_model.pkl', 'rb') as f:
                self.nfl_model = pickle.load(f)
            st.success("✅ ML model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load ML model: {e}")
            self.nfl_model = None
    
    def _clean_data(self):
        """Clean the data by filling NaN values"""
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
        """Safely convert value to float"""
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _get_defense_stats(self, team_name):
        """Get defense stats for a specific team"""
        if not self.defense_data:
            return None
        
        # Try different team name formats
        team_variations = [team_name, team_name.upper(), team_name.lower(), team_name.title()]
        
        for team in self.defense_data:
            defense_team = team.get('Team', '')
            if defense_team in team_variations:
                return team
        
        # If not found, try partial matching
        for team in self.defense_data:
            defense_team = team.get('Team', '')
            if team_name in defense_team or defense_team in team_name:
                return team
        
        return None
    
    def get_game_context(self, player_team, opponent_team):
        """Get game context including odds and SOS"""
        context = {
            'expected_total': 45.0,
            'spread': 0.0,
            'sos_adjustment': 1.0
        }
        
        # Get SOS adjustment
        if self.sos_data and player_team in self.sos_data:
            sos_value = self.sos_data[player_team].get('combined_sos', 0.5)
            context['sos_adjustment'] = 1.0 + (self._safe_float(sos_value) - 0.5) * 0.3
        
        # Get odds data
        if self.odds_data:
            for odds in self.odds_data:
                home_team = odds.get('home_team', '')
                away_team = odds.get('away_team', '')
                
                if (home_team == player_team and away_team == opponent_team) or \
                   (home_team == opponent_team and away_team == player_team):
                    
                    if odds.get('market') == 'totals' and odds.get('point'):
                        context['expected_total'] = self._safe_float(odds['point'], 45.0)
                    elif odds.get('market') == 'spreads' and odds.get('point'):
                        spread = self._safe_float(odds['point'], 0.0)
                        context['spread'] = -spread if home_team == player_team else spread
        
        return context
    
    def estimate_vegas_line(self, projection, position, stat_type):
        """Estimate Vegas line based on projection"""
        safe_projection = self._safe_float(projection, 0.0)
        
        if position == 'RB':
            if stat_type == 'rushing_yards':
                estimated_line = round(safe_projection / 5) * 5
                return max(45, min(125, estimated_line))
            elif stat_type == 'rushing_tds':
                estimated_line = round(safe_projection * 2) / 2
                return max(0.5, min(2.5, estimated_line))
        
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
        
        return round(safe_projection)
    
    def project_rushing_stats(self, player_name, opponent_team, games_played=9):
        """Project rushing stats for RBs and QBs"""
        if self.rb_data is None or self.qb_data is None:
            raise ValueError("Player data not loaded properly")
        
        # Check RB data first
        rb_row = self.rb_data[self.rb_data['PlayerName'] == player_name]
        if not rb_row.empty:
            return self._project_rb_rushing(rb_row.iloc[0], opponent_team, games_played)
        
        # Check QB data for rushing QBs
        qb_row = self.qb_data[self.qb_data['PlayerName'] == player_name]
        if not qb_row.empty:
            return self._project_qb_rushing(qb_row.iloc[0], opponent_team, games_played)
        
        raise ValueError(f"Player '{player_name}' not found")
    
    def _project_rb_rushing(self, rb_stats, opponent_team, games_played):
        """Project rushing stats for running backs"""
        player_team = rb_stats['Team']
        
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            # If no defense stats, use defaults
            defense_stats = {
                'RUSHING YARDS PER GAME ALLOWED': 100,
                'RUSHING TD PER GAME ALLOWED': 1.0,
                'RUSHING ATTEMPTS ALLOWED': 25
            }
        
        game_context = self.get_game_context(player_team, opponent_team)
        
        projections = {}
        
        # Rushing Yards
        rb_rush_yds_pg = self._safe_float(rb_stats['RushingYDS']) / max(1, games_played)
        def_rush_yds_allowed = self._safe_float(defense_stats.get('RUSHING YARDS PER GAME ALLOWED', 100))
        
        game_script = 1.15 if game_context['spread'] > 3 else 0.85 if game_context['spread'] < -3 else 1.0
        
        projections['RushingYards'] = (
            (rb_rush_yds_pg + def_rush_yds_allowed) / 2 * 
            game_script * game_context['sos_adjustment']
        )
        
        # Rushing TDs
        rb_rush_td_pg = self._safe_float(rb_stats['RushingTD']) / max(1, games_played)
        def_rush_td_allowed = self._safe_float(defense_stats.get('RUSHING TD PER GAME ALLOWED', 1.0))
        
        projections['RushingTDs'] = (
            (rb_rush_td_pg + def_rush_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # Carries
        rb_carries_pg = self._safe_float(rb_stats.get('TouchCarries', rb_stats.get('RushingATT', 0))) / max(1, games_played)
        def_rush_attempts_allowed = self._safe_float(defense_stats.get('RUSHING ATTEMPTS ALLOWED', 25))
        
        projections['Carries'] = (
            (rb_carries_pg + def_rush_attempts_allowed * 0.3) / 1.3 * game_script
        )
        
        return projections, rb_stats, defense_stats, game_context, 'RB'
    
    def _project_qb_rushing(self, qb_stats, opponent_team, games_played):
        """Project rushing stats for quarterbacks"""
        player_team = qb_stats['Team']
        
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            defense_stats = {
                'RUSHING YARDS PER GAME ALLOWED': 100,
                'RUSHING TD PER GAME ALLOWED': 1.0
            }
        
        game_context = self.get_game_context(player_team, opponent_team)
        
        projections = {}
        
        # Rushing Yards
        qb_rush_yds_pg = self._safe_float(qb_stats['RushingYDS']) / max(1, games_played)
        projections['RushingYards'] = qb_rush_yds_pg * game_context['sos_adjustment']
        
        # Rushing TDs
        qb_rush_td_pg = self._safe_float(qb_stats['RushingTD']) / max(1, games_played)
        projections['RushingTDs'] = qb_rush_td_pg * (game_context['expected_total'] / 45.0)
        
        # Carries (estimate)
        projections['Carries'] = projections['RushingYards'] / 4.5
        
        return projections, qb_stats, defense_stats, game_context, 'QB'
    
    def project_passing_stats(self, qb_name, opponent_team, games_played=9):
        """Project passing stats for QBs"""
        if self.qb_data is None:
            raise ValueError("QB data not loaded properly")
            
        qb_row = self.qb_data[self.qb_data['PlayerName'] == qb_name]
        if qb_row.empty:
            raise ValueError(f"QB '{qb_name}' not found")
        
        qb_stats = qb_row.iloc[0]
        player_team = qb_stats['Team']
        
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            defense_stats = {
                'PASSING YARDS ALLOWED': 230,
                'PASSING TD ALLOWED': 1.5,
                'PASSING ATTEMPTS ALLOWED': 35,
                'PASSING COMPLETIONS ALLOWED': 23
            }
        
        game_context = self.get_game_context(player_team, opponent_team)
        
        projections = {}
        
        # Passing Yards
        qb_pass_yds_pg = self._safe_float(qb_stats['PassingYDS']) / max(1, games_played)
        def_pass_yds_allowed = self._safe_float(defense_stats.get('PASSING YARDS ALLOWED', 230))
        
        projections['PassingYards'] = (
            (qb_pass_yds_pg + def_pass_yds_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # Passing TDs
        qb_pass_td_pg = self._safe_float(qb_stats['PassingTD']) / max(1, games_played)
        def_pass_td_allowed = self._safe_float(defense_stats.get('PASSING TD ALLOWED', 1.5))
        
        projections['PassingTDs'] = (
            (qb_pass_td_pg + def_pass_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # Interceptions
        qb_int_pg = self._safe_float(qb_stats['PassingInt']) / max(1, games_played)
        projections['Interceptions'] = qb_int_pg * game_context['sos_adjustment']
        
        # Pass Attempts & Completions
        def_attempts_allowed = self._safe_float(defense_stats.get('PASSING ATTEMPTS ALLOWED', 35))
        def_completions_allowed = self._safe_float(defense_stats.get('PASSING COMPLETIONS ALLOWED', 23))
        
        projections['PassAttempts'] = def_attempts_allowed * (game_context['expected_total'] / 45.0)
        defense_completion_pct = def_completions_allowed / def_attempts_allowed if def_attempts_allowed > 0 else 0.65
        projections['Completions'] = projections['PassAttempts'] * defense_completion_pct
        
        return projections, qb_stats, defense_stats, game_context
    
    def get_players_playing_this_week(self):
        """Get all players who are playing this week"""
        playing_players = {
            'rushers': [],
            'passers': []
        }
        
        # If no schedule data, use ALL players
        if not self.schedule_data:
            st.warning("⚠️ No schedule data found - using ALL players")
            if self.rb_data is not None:
                for _, player in self.rb_data.iterrows():
                    playing_players['rushers'].append({
                        'name': player['PlayerName'],
                        'team': player['Team'],
                        'opponent': 'TBD',  # Unknown opponent
                        'position': 'RB'
                    })
            
            if self.qb_data is not None:
                for _, player in self.qb_data.iterrows():
                    playing_players['passers'].append({
                        'name': player['PlayerName'],
                        'team': player['Team'],
                        'opponent': 'TBD',
                        'position': 'QB'
                    })
                    playing_players['rushers'].append({
                        'name': player['PlayerName'],
                        'team': player['Team'],
                        'opponent': 'TBD',
                        'position': 'QB'
                    })
            
            return playing_players
        
        # Extract teams playing this week
        playing_teams = set()
        team_matchups = {}
        
        st.info(f"🔍 Analyzing {len(self.schedule_data)} schedule entries...")
        
        for game in self.schedule_data:
            # Handle different schedule formats
            home_team = None
            away_team = None
            
            if isinstance(game, dict):
                # Try different key combinations
                home_team = game.get('home_team') or game.get('home') or game.get('HomeTeam') or game.get('Home')
                away_team = game.get('away_team') or game.get('away') or game.get('AwayTeam') or game.get('Away')
            
            if home_team and away_team:
                playing_teams.add(home_team)
                playing_teams.add(away_team)
                team_matchups[home_team] = away_team
                team_matchups[away_team] = home_team
        
        st.info(f"🏈 Found {len(playing_teams)} teams playing this week: {list(playing_teams)[:5]}...")
        
        # Find players on playing teams
        rb_count = 0
        qb_count = 0
        
        if self.rb_data is not None:
            for _, player in self.rb_data.iterrows():
                team = player['Team']
                if team in playing_teams:
                    playing_players['rushers'].append({
                        'name': player['PlayerName'],
                        'team': team,
                        'opponent': team_matchups.get(team, 'Unknown'),
                        'position': 'RB'
                    })
                    rb_count += 1
        
        if self.qb_data is not None:
            for _, player in self.qb_data.iterrows():
                team = player['Team']
                if team in playing_teams:
                    playing_players['passers'].append({
                        'name': player['PlayerName'],
                        'team': team,
                        'opponent': team_matchups.get(team, 'Unknown'),
                        'position': 'QB'
                    })
                    # Add QBs to rushers for rushing projections
                    playing_players['rushers'].append({
                        'name': player['PlayerName'],
                        'team': team,
                        'opponent': team_matchups.get(team, 'Unknown'),
                        'position': 'QB'
                    })
                    qb_count += 1
        
        st.success(f"✅ Found {rb_count} RBs and {qb_count} QBs playing this week")
        
        return playing_players
    
    def calculate_bet_grade(self, projection, vegas_line, stat_type):
        """Calculate bet grade from 0-100"""
        difference = abs(projection - vegas_line)
        
        if stat_type in ['rushing_yards', 'passing_yards']:
            if difference >= 20: return 95
            elif difference >= 15: return 90
            elif difference >= 10: return 85
            elif difference >= 7: return 80
            elif difference >= 5: return 75
            elif difference >= 3: return 70
            else: return 65
        elif stat_type in ['rushing_tds', 'passing_tds']:
            if difference >= 0.8: return 95
            elif difference >= 0.6: return 90
            elif difference >= 0.4: return 85
            elif difference >= 0.3: return 80
            elif difference >= 0.2: return 75
            elif difference >= 0.1: return 70
            else: return 65
        else:
            return 70
    
    def generate_all_projections(self, games_played=9):
        """Generate projections for all players playing this week"""
        playing_players = self.get_players_playing_this_week()
        all_projections = []
        
        total_players = len(playing_players['rushers']) + len(playing_players['passers'])
        st.info(f"🔄 Generating projections for {total_players} players...")
        
        if total_players == 0:
            st.error("❌ No players found to generate projections for!")
            return all_projections
        
        # Progress tracking
        progress_bar = st.progress(0)
        processed = 0
        
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
                
                # Determine bet direction and primary line
                if projections['RushingYards'] > vegas_rush_yds:
                    bet_direction = "↑"
                    primary_line = f"{vegas_rush_yds} Rush Yds"
                    proj_diff = projections['RushingYards'] - vegas_rush_yds
                else:
                    bet_direction = "↓"
                    primary_line = f"{vegas_rush_yds} Rush Yds"
                    proj_diff = vegas_rush_yds - projections['RushingYards']
                
                all_projections.append({
                    'Player': player_info['name'],
                    'Team': player_info['team'],
                    'Opponent': player_info['opponent'],
                    'Position': position,
                    'BetGrade': int(bet_grade),
                    'BetDirection': bet_direction,
                    'PrimaryLine': primary_line,
                    'ProjectionDifferential': f"+{proj_diff:.1f}",
                    'Game': f"{player_info['team']} vs {player_info['opponent']}",
                    'RushingYards': projections['RushingYards'],
                    'VegasLine': vegas_rush_yds
                })
                
            except Exception as e:
                st.warning(f"Could not generate projection for {player_info['name']}: {str(e)}")
                continue
            
            processed += 1
            progress_bar.progress(processed / total_players)
        
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
                
                # Determine bet direction and primary line
                if projections['PassingYards'] > vegas_pass_yds:
                    bet_direction = "↑"
                    primary_line = f"{vegas_pass_yds} Pass Yds"
                    proj_diff = projections['PassingYards'] - vegas_pass_yds
                else:
                    bet_direction = "↓"
                    primary_line = f"{vegas_pass_yds} Pass Yds"
                    proj_diff = vegas_pass_yds - projections['PassingYards']
                
                all_projections.append({
                    'Player': player_info['name'],
                    'Team': player_info['team'],
                    'Opponent': player_info['opponent'],
                    'Position': 'QB',
                    'BetGrade': int(bet_grade),
                    'BetDirection': bet_direction,
                    'PrimaryLine': primary_line,
                    'ProjectionDifferential': f"+{proj_diff:.1f}",
                    'Game': f"{player_info['team']} vs {player_info['opponent']}",
                    'PassingYards': projections['PassingYards'],
                    'VegasLine': vegas_pass_yds
                })
                
            except Exception as e:
                st.warning(f"Could not generate projection for {player_info['name']}: {str(e)}")
                continue
            
            processed += 1
            progress_bar.progress(processed / total_players)
        
        progress_bar.empty()
        
        # Sort by bet grade (highest first)
        return sorted(all_projections, key=lambda x: x['BetGrade'], reverse=True)

def create_player_card(player_data):
    """Create a card for a player"""
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
                {player_data['PrimaryLine']}
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 8px;">
                Proj Differential
            </div>
            <div style="font-weight: bold; font-size: 14px; color: {'#00aa00' if player_data['BetDirection'] == '↑' else '#aa0000'}">
                {player_data['ProjectionDifferential']}
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
    
    # Filters
    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
    
    with col1:
        sport_filter = st.selectbox("SPORT", ["ALL ✓", "NFL"], key="sport_filter")
    
    with col2:
        position_filter = st.selectbox("POSITION", ["ALL ✓", "QB", "RB"], key="position_filter")
    
    with col3:
        # Get available teams from player data
        available_teams = ["ALL ✓"]
        if projector.rb_data is not None:
            team_list = list(projector.rb_data['Team'].unique()) + list(projector.qb_data['Team'].unique())
            available_teams.extend(sorted(list(set(team_list))))
        team_filter = st.selectbox("TEAM MARKET", available_teams, key="team_filter")
    
    with col4:
        show_count = st.selectbox("SHOW", ["25", "50", "100", "ALL"], key="show_count")
    
    # Search bar
    search_query = st.text_input("🔍 Search players...", placeholder="Type to search players or teams")
    
    # Games played input
    games_played = st.number_input("Games Played This Season", min_value=1, max_value=17, value=9, key="games_played")
    
    # Generate all projections button
    if st.button("🔄 Generate All Projections", type="primary", use_container_width=True):
        with st.spinner("Generating projections for all players..."):
            all_projections = projector.generate_all_projections(games_played)
            st.session_state.all_projections = all_projections
            if all_projections:
                st.success(f"✅ Generated {len(all_projections)} projections!")
            else:
                st.error("❌ No projections were generated. Check the debug info below.")
    
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
        st.info("👆 Click 'Generate All Projections' to see player projections for this week")

if __name__ == "__main__":
    main()
