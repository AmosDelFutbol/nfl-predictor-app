import pandas as pd
import numpy as np
import json
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class EnhancedNFLProjector:
    def __init__(self):
        """Initialize the enhanced projector with all available data sources"""
        self.missing_files = []
        
        try:
            # Load base data files from GitHub
            self.rb_data = self._load_csv_file(['RB_season.csv', 'rb_data.csv', 'running_backs.csv'])
            self.qb_data = self._load_csv_file(['QB_season.csv', 'qb_data.csv', 'quarterbacks.csv'])
            
            # Load JSON defense and offense data
            self.defense_data = self._load_json_file(['2025_NFL_DEFENSE.json', 'defense_data.json', 'nfl_defense.json'])
            self.offense_data = self._load_json_file(['2025_NFL_OFFENSE.json', 'offense_data.json', 'nfl_offense.json'])
            
            # Load advanced data from our main model
            self.elo_data = self._load_elo_data()
            self.sos_data = self._load_sos_data()
            self.schedule_data = self._load_schedule_data()
            self.odds_data = self._load_odds_data()
            
            # Clean column names
            self._clean_data()
            
            # Team mapping for consistency
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
            
            # Show initialization status
            if not self.missing_files:
                st.success("✅ Enhanced NFL Projector initialized successfully!")
                st.info(f"Loaded: {len(self.rb_data)} RBs, {len(self.qb_data)} QBs")
                if self.defense_data:
                    st.info(f"Defense data loaded for {len(self.defense_data)} teams")
                if self.offense_data:
                    st.info(f"Offense data loaded for {len(self.offense_data)} teams")
                if self.elo_data:
                    st.info(f"Advanced metrics loaded for {len(self.elo_data)} teams")
            else:
                st.warning(f"⚠️ Projector initialized with missing files: {', '.join(self.missing_files)}")
            
        except Exception as e:
            st.error(f"❌ Error initializing projector: {e}")
            # Create empty dataframes to prevent further errors
            self.rb_data = pd.DataFrame()
            self.qb_data = pd.DataFrame()
            self.defense_data = {}
            self.offense_data = {}
            self.elo_data = {}
            self.sos_data = {}
    
    def _load_csv_file(self, possible_filenames):
        """Try to load a CSV file with multiple possible names"""
        for filename in possible_filenames:
            try:
                df = pd.read_csv(filename)
                st.success(f"✅ Loaded {filename}")
                return df
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Error loading {filename}: {e}")
                continue
        
        # If we get here, none of the files worked
        self.missing_files.extend(possible_filenames)
        return pd.DataFrame()
    
    def _load_json_file(self, possible_filenames):
        """Try to load a JSON file with multiple possible names"""
        for filename in possible_filenames:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                st.success(f"✅ Loaded {filename}")
                return data
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Error loading {filename}: {e}")
                continue
        
        # If we get here, none of the files worked
        self.missing_files.extend(possible_filenames)
        return {}
    
    def _load_elo_data(self):
        """Load ELO and efficiency metrics"""
        try:
            elo_df = pd.read_csv('teams_power_rating.csv')
            elo_data = {}
            for _, row in elo_df.iterrows():
                team_abbr = self._get_team_abbreviation_from_name(row['Team'])
                if team_abbr:
                    elo_data[team_abbr] = {
                        'nfelo': row['nfelo'],
                        'off_epa_play': row['Play'],
                        'def_epa_play': row['Play.1'],
                        'off_epa_pass': row['Pass'],
                        'def_epa_pass': row['Pass.1'],
                        'off_epa_rush': row['Rush'],
                        'def_epa_rush': row['Rush.1'],
                        'net_epa': row['Play.2'],
                        'qb_adj': row['QB Adj'],
                        'points_for': row['For'],
                        'points_against': row['Against'],
                        'wow_change': row['WoW']
                    }
            return elo_data
        except Exception as e:
            st.warning(f"⚠️ Could not load ELO data: {e}")
            return {}
    
    def _load_sos_data(self):
        """Load strength of schedule data"""
        try:
            with open('nfl_strength_of_schedule.json', 'r') as f:
                sos_data = json.load(f)
            return sos_data.get('sos_rankings', {})
        except:
            st.warning("⚠️ Could not load SOS data")
            return {}
    
    def _load_schedule_data(self):
        """Load schedule data for game context"""
        try:
            with open('week_10_schedule.json', 'r') as f:
                schedule_data = json.load(f)
            return schedule_data.get('Week 10', [])
        except:
            st.warning("⚠️ Could not load schedule data")
            return []
    
    def _load_odds_data(self):
        """Load odds data for game totals and spreads"""
        try:
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
            return odds_data
        except:
            st.warning("⚠️ Could not load odds data")
            return []
    
    def _clean_data(self):
        """Clean all dataframes"""
        if not self.rb_data.empty:
            self.rb_data.columns = self.rb_data.columns.str.strip()
        if not self.qb_data.empty:
            self.qb_data.columns = self.qb_data.columns.str.strip()
    
    def _get_team_abbreviation_from_name(self, team_name):
        """Convert team name to abbreviation"""
        return self.team_mapping.get(team_name, None)
    
    def _get_defense_stats(self, team_name):
        """Get defense stats from JSON data"""
        if not self.defense_data:
            return {}
        
        # Try different possible structures in the JSON
        if isinstance(self.defense_data, list):
            for team in self.defense_data:
                if team.get('Team') == team_name:
                    return team
        elif isinstance(self.defense_data, dict):
            # If it's a dict with team names as keys
            if team_name in self.defense_data:
                return self.defense_data[team_name]
            # If it's a dict with a list of teams
            elif 'teams' in self.defense_data:
                for team in self.defense_data['teams']:
                    if team.get('Team') == team_name:
                        return team
        
        return {}
    
    def _get_offense_stats(self, team_name):
        """Get offense stats from JSON data for enhanced projections"""
        if not self.offense_data:
            return {}
        
        # Try different possible structures in the JSON
        if isinstance(self.offense_data, list):
            for team in self.offense_data:
                if team.get('Team') == team_name:
                    return team
        elif isinstance(self.offense_data, dict):
            # If it's a dict with team names as keys
            if team_name in self.offense_data:
                return self.offense_data[team_name]
            # If it's a dict with a list of teams
            elif 'teams' in self.offense_data:
                for team in self.offense_data['teams']:
                    if team.get('Team') == team_name:
                        return team
        
        return {}
    
    def get_game_context(self, player_team, opponent_team):
        """Get advanced game context for projections"""
        context = {
            'expected_total': 45.0,
            'spread': 0.0,
            'player_team_elo': 1500,
            'opponent_team_elo': 1500,
            'player_team_off_epa': 0.0,
            'opponent_team_def_epa': 0.0,
            'sos_adjustment': 1.0,
            'player_offense_stats': {},
            'opponent_defense_stats': {}
        }
        
        # Get ELO data
        player_abbr = self._get_team_abbreviation_from_name(player_team)
        opponent_abbr = self._get_team_abbreviation_from_name(opponent_team)
        
        if player_abbr in self.elo_data:
            context['player_team_elo'] = self.elo_data[player_abbr]['nfelo']
            context['player_team_off_epa'] = self.elo_data[player_abbr]['off_epa_play']
        
        if opponent_abbr in self.elo_data:
            context['opponent_team_elo'] = self.elo_data[opponent_abbr]['nfelo']
            context['opponent_team_def_epa'] = self.elo_data[opponent_abbr]['def_epa_play']
        
        # Get offense and defense stats from JSON files
        context['player_offense_stats'] = self._get_offense_stats(player_team)
        context['opponent_defense_stats'] = self._get_defense_stats(opponent_team)
        
        # Get SOS adjustment
        if player_team in self.sos_data:
            context['sos_adjustment'] = 1.0 + (self.sos_data[player_team]['combined_sos'] - 0.5) * 0.3
        
        # Get game odds if available
        for odds in self.odds_data:
            if (odds.get('home_team') == player_team and odds.get('away_team') == opponent_team) or \
               (odds.get('home_team') == opponent_team and odds.get('away_team') == player_team):
                if odds.get('market') == 'totals' and odds.get('point'):
                    context['expected_total'] = odds['point']
                elif odds.get('market') == 'spreads' and odds.get('point'):
                    context['spread'] = odds['point']
        
        return context
    
    def calculate_confidence_factors(self, player_stats, defense_stats, game_context, position):
        """Calculate confidence level for projections"""
        confidence_score = 0.5  # Base confidence
        
        # Sample size confidence
        games_played = player_stats.get('games_played', 1)
        if games_played >= 8:
            confidence_score += 0.2
        elif games_played >= 4:
            confidence_score += 0.1
        
        # Matchup confidence (clear strengths/weaknesses)
        elo_diff = game_context['player_team_elo'] - game_context['opponent_team_elo']
        if abs(elo_diff) > 100:
            confidence_score += 0.2
        elif abs(elo_diff) > 50:
            confidence_score += 0.1
        
        # Data availability confidence
        if game_context['player_offense_stats'] and game_context['opponent_defense_stats']:
            confidence_score += 0.1
        
        return min(confidence_score, 1.0)
    
    def create_sample_data(self):
        """Create sample data if real data isn't available"""
        st.warning("📝 Creating sample data for demonstration...")
        
        # Sample RB data
        sample_rbs = [
            ['Christian McCaffrey', 'SF', 850, 8, 350, 3, 45, 12],
            ['Austin Ekeler', 'LAC', 650, 6, 400, 4, 55, 10],
            ['Derrick Henry', 'TEN', 800, 7, 150, 1, 15, 8],
            ['Saquon Barkley', 'NYG', 700, 5, 300, 2, 40, 9]
        ]
        self.rb_data = pd.DataFrame(sample_rbs, columns=['PlayerName', 'Team', 'RushingYDS', 'RushingTD', 'ReceivingYDS', 'ReceivingTD', 'ReceivingRec', 'TouchCarries'])
        
        # Sample QB data
        sample_qbs = [
            ['Patrick Mahomes', 'KC', 2800, 22, 5, 150, 2],
            ['Jalen Hurts', 'PHI', 2500, 18, 6, 400, 8],
            ['Josh Allen', 'BUF', 2700, 20, 8, 300, 4],
            ['Lamar Jackson', 'BAL', 2400, 16, 4, 600, 6]
        ]
        self.qb_data = pd.DataFrame(sample_qbs, columns=['PlayerName', 'Team', 'PassingYDS', 'PassingTD', 'PassingInt', 'RushingYDS', 'RushingTD'])
        
        # Sample defense data
        self.defense_data = [
            {'Team': 'San Francisco 49ers', 'RUSHING_YARDS_ALLOWED': 85, 'RUSHING_TD_ALLOWED': 0.7, 'PASSING_YARDS_ALLOWED': 210, 'PASSING_TD_ALLOWED': 1.2},
            {'Team': 'Kansas City Chiefs', 'RUSHING_YARDS_ALLOWED': 95, 'RUSHING_TD_ALLOWED': 0.9, 'PASSING_YARDS_ALLOWED': 230, 'PASSING_TD_ALLOWED': 1.4},
            {'Team': 'Buffalo Bills', 'RUSHING_YARDS_ALLOWED': 90, 'RUSHING_TD_ALLOWED': 0.8, 'PASSING_YARDS_ALLOWED': 225, 'PASSING_TD_ALLOWED': 1.3},
            {'Team': 'Dallas Cowboys', 'RUSHING_YARDS_ALLOWED': 100, 'RUSHING_TD_ALLOWED': 1.0, 'PASSING_YARDS_ALLOWED': 240, 'PASSING_TD_ALLOWED': 1.5}
        ]
        
        # Sample offense data
        self.offense_data = [
            {'Team': 'San Francisco 49ers', 'TOTAL_YARDS': 380, 'RUSHING_YARDS': 140, 'PASSING_YARDS': 240, 'POINTS': 28.5},
            {'Team': 'Kansas City Chiefs', 'TOTAL_YARDS': 370, 'RUSHING_YARDS': 110, 'PASSING_YARDS': 260, 'POINTS': 27.8},
            {'Team': 'Buffalo Bills', 'TOTAL_YARDS': 360, 'RUSHING_YARDS': 120, 'PASSING_YARDS': 240, 'POINTS': 26.5},
            {'Team': 'Dallas Cowboys', 'TOTAL_YARDS': 350, 'RUSHING_YARDS': 130, 'PASSING_YARDS': 220, 'POINTS': 25.8}
        ]
        
        st.info("✅ Sample data created for demonstration")
    
    def _get_json_stat(self, stats_dict, possible_keys, default=0):
        """Safely get a stat from JSON data with multiple possible key names"""
        if not stats_dict:
            return default
        
        for key in possible_keys:
            if key in stats_dict:
                return stats_dict[key]
        return default
    
    def project_rb_stats(self, rb_name, opponent_team, games_played=9):
        """Enhanced RB projections using advanced metrics"""
        
        if self.rb_data.empty:
            raise ValueError("RB data not loaded")
        
        # Get base data
        rb_row = self.rb_data[self.rb_data['PlayerName'] == rb_name]
        if rb_row.empty:
            raise ValueError(f"RB '{rb_name}' not found in data")
        
        rb_stats = rb_row.iloc[0]
        
        # Get player's team
        player_team = rb_stats.get('Team', 'Unknown')
        
        # Get advanced game context
        game_context = self.get_game_context(player_team, opponent_team)
        defense_stats = game_context['opponent_defense_stats']
        offense_stats = game_context['player_offense_stats']
        
        # Calculate confidence
        confidence = self.calculate_confidence_factors(
            {'games_played': games_played}, 
            defense_stats, 
            game_context, 
            'RB'
        )
        
        # Enhanced projections with advanced metrics
        projections = {}
        
        # Get defense stats with flexible key names
        def_rush_yds_allowed = self._get_json_stat(defense_stats, 
            ['RUSHING_YARDS_ALLOWED', 'RUSHING_YARDS_PER_GAME', 'RUSH_YARDS_ALLOWED'], 100)
        def_rush_td_allowed = self._get_json_stat(defense_stats,
            ['RUSHING_TD_ALLOWED', 'RUSH_TD_ALLOWED', 'RUSHING_TOUCHDOWNS_ALLOWED'], 1.0)
        def_pass_yds_allowed = self._get_json_stat(defense_stats,
            ['PASSING_YARDS_ALLOWED', 'PASS_YARDS_ALLOWED', 'PASSING_YARDS_PER_GAME'], 230)
        def_pass_td_allowed = self._get_json_stat(defense_stats,
            ['PASSING_TD_ALLOWED', 'PASS_TD_ALLOWED', 'PASSING_TOUCHDOWNS_ALLOWED'], 1.5)
        def_completions_allowed = self._get_json_stat(defense_stats,
            ['PASSING_COMPLETIONS_ALLOWED', 'COMPLETIONS_ALLOWED'], 24)
        def_rush_attempts_allowed = self._get_json_stat(defense_stats,
            ['RUSHING_ATTEMPTS_ALLOWED', 'RUSH_ATTEMPTS_ALLOWED'], 25)
        
        # Get offense stats for enhanced accuracy
        team_rush_yds = self._get_json_stat(offense_stats,
            ['RUSHING_YARDS', 'RUSH_YARDS', 'RUSHING_YARDS_PER_GAME'], 120)
        team_pass_yds = self._get_json_stat(offense_stats,
            ['PASSING_YARDS', 'PASS_YARDS', 'PASSING_YARDS_PER_GAME'], 240)
        team_points = self._get_json_stat(offense_stats,
            ['POINTS', 'POINTS_PER_GAME', 'SCORING'], 24.0)
        
        # RUSHING YARDS - Enhanced calculation with offense data
        rb_rush_yds_per_game = rb_stats['RushingYDS'] / games_played
        
        # Team rushing share adjustment (how much of team's rushing this RB gets)
        team_rush_share = min(1.0, rb_rush_yds_per_game / (team_rush_yds / games_played)) if team_rush_yds > 0 else 0.5
        
        # Apply efficiency adjustments
        off_rush_efficiency = max(0.5, min(2.0, 
            game_context['player_team_off_epa'] / 0.05 + 1.0)) if game_context['player_team_off_epa'] != 0 else 1.0
        def_rush_efficiency = max(0.5, min(2.0, 
            -game_context['opponent_team_def_epa'] / 0.05 + 1.0)) if game_context['opponent_team_def_epa'] != 0 else 1.0
        
        # Game script adjustment (based on spread)
        game_script_multiplier = 1.0
        if game_context['spread'] > 3:  # Heavy favorite - positive game script for rushing
            game_script_multiplier = 1.15
        elif game_context['spread'] < -3:  # Underdog - negative game script for rushing
            game_script_multiplier = 0.85
        
        projections['RushingYards'] = (
            (rb_rush_yds_per_game * off_rush_efficiency * team_rush_share + 
             def_rush_yds_allowed * def_rush_efficiency * 0.3) / 1.3 *
            game_script_multiplier * game_context['sos_adjustment']
        )
        
        # RUSHING TDs - Enhanced calculation
        rb_rush_td_per_game = rb_stats['RushingTD'] / games_played
        
        # Red zone efficiency adjustment
        red_zone_efficiency = min(2.0, team_points / 24.0)  # Use team scoring for TD probability
        
        projections['RushingTDs'] = (
            (rb_rush_td_per_game + def_rush_td_allowed * 0.5) / 1.5 *
            red_zone_efficiency * game_context['sos_adjustment']
        )
        
        # RECEIVING YARDS - Enhanced calculation with offense data
        rb_rec_yds_per_game = rb_stats['ReceivingYDS'] / games_played
        
        # Team receiving share for RBs
        team_rec_share_rb = min(1.0, rb_rec_yds_per_game / (team_pass_yds / games_played * 0.2)) if team_pass_yds > 0 else 0.1
        
        projections['ReceivingYards'] = (
            rb_rec_yds_per_game * team_rec_share_rb *
            (def_pass_yds_allowed / 231.8) *  # Normalize
            game_context['sos_adjustment']
        )
        
        # RECEPTIONS - Enhanced calculation
        rb_rec_per_game = rb_stats['ReceivingRec'] / games_played
        
        projections['Receptions'] = (
            rb_rec_per_game * team_rec_share_rb *
            (def_completions_allowed / 24.1)  # Normalize
        )
        
        # RECEIVING TDs - Enhanced calculation
        rb_rec_td_per_game = rb_stats['ReceivingTD'] / games_played
        
        projections['ReceivingTDs'] = (
            (rb_rec_td_per_game + def_pass_td_allowed * 0.2) / 1.2 *
            red_zone_efficiency * game_context['sos_adjustment']
        )
        
        # CARRIES - Enhanced calculation
        rb_carries_per_game = rb_stats['TouchCarries'] / games_played
        
        projections['Carries'] = (
            (rb_carries_per_game * off_rush_efficiency * team_rush_share + 
             def_rush_attempts_allowed * 0.25) / 1.25 *
            game_script_multiplier
        )
        
        # FANTASY POINTS - PPR scoring
        projections['FantasyPoints'] = (
            projections['RushingYards'] * 0.1 +
            projections['RushingTDs'] * 6 +
            projections['ReceivingYards'] * 0.1 +
            projections['Receptions'] * 0.5 +  # PPR scoring
            projections['ReceivingTDs'] * 6
        )
        
        # Add confidence metrics
        projections['Confidence'] = confidence
        projections['GameScript'] = 'Positive' if game_script_multiplier > 1.0 else 'Negative' if game_script_multiplier < 1.0 else 'Neutral'
        projections['ExpectedGameTotal'] = game_context['expected_total']
        
        return projections, rb_stats, defense_stats, game_context

    def project_qb_stats(self, qb_name, opponent_team, games_played=9):
        """Enhanced QB projections using advanced metrics"""
        
        if self.qb_data.empty:
            raise ValueError("QB data not loaded")
        
        # Get base data
        qb_row = self.qb_data[self.qb_data['PlayerName'] == qb_name]
        if qb_row.empty:
            raise ValueError(f"QB '{qb_name}' not found in data")
        
        qb_stats = qb_row.iloc[0]
        
        # Get player's team
        player_team = qb_stats.get('Team', 'Unknown')
        
        # Get advanced game context
        game_context = self.get_game_context(player_team, opponent_team)
        defense_stats = game_context['opponent_defense_stats']
        offense_stats = game_context['player_offense_stats']
        
        # Calculate confidence
        confidence = self.calculate_confidence_factors(
            {'games_played': games_played}, 
            defense_stats, 
            game_context, 
            'QB'
        )
        
        # Enhanced projections with advanced metrics
        projections = {}
        
        # Get defense stats with flexible key names
        def_pass_yds_allowed = self._get_json_stat(defense_stats,
            ['PASSING_YARDS_ALLOWED', 'PASS_YARDS_ALLOWED', 'PASSING_YARDS_PER_GAME'], 230)
        def_pass_td_allowed = self._get_json_stat(defense_stats,
            ['PASSING_TD_ALLOWED', 'PASS_TD_ALLOWED', 'PASSING_TOUCHDOWNS_ALLOWED'], 1.5)
        def_interceptions = self._get_json_stat(defense_stats,
            ['INTERCEPTIONS', 'INTS', 'PASSING_INTERCEPTIONS'], 0.8)
        def_attempts_allowed = self._get_json_stat(defense_stats,
            ['PASSING_ATTEMPTS_ALLOWED', 'PASS_ATTEMPTS_ALLOWED'], 35)
        def_completions_allowed = self._get_json_stat(defense_stats,
            ['PASSING_COMPLETIONS_ALLOWED', 'COMPLETIONS_ALLOWED'], 24)
        
        # Get offense stats for enhanced accuracy
        team_pass_yds = self._get_json_stat(offense_stats,
            ['PASSING_YARDS', 'PASS_YARDS', 'PASSING_YARDS_PER_GAME'], 240)
        team_points = self._get_json_stat(offense_stats,
            ['POINTS', 'POINTS_PER_GAME', 'SCORING'], 24.0)
        
        # PASSING YARDS - Enhanced calculation with offense data
        qb_pass_yds_per_game = qb_stats['PassingYDS'] / games_played
        
        # Team passing share (QB's contribution to team passing)
        team_pass_share = min(1.0, qb_pass_yds_per_game / (team_pass_yds / games_played)) if team_pass_yds > 0 else 1.0
        
        # Efficiency adjustments
        qb_efficiency = max(0.7, min(1.3, 
            game_context.get('player_team_off_epa', 0.0) / 0.05 + 1.0)) if game_context.get('player_team_off_epa', 0) != 0 else 1.0
        def_pass_efficiency = max(0.7, min(1.3, 
            -game_context.get('opponent_team_def_epa', 0.0) / 0.05 + 1.0)) if game_context.get('opponent_team_def_epa', 0) != 0 else 1.0
        
        # Game total adjustment
        total_adjustment = game_context['expected_total'] / 45.0
        
        projections['PassingYards'] = (
            (qb_pass_yds_per_game * qb_efficiency * team_pass_share + 
             def_pass_yds_allowed * def_pass_efficiency * 0.5) / 1.5 *
            total_adjustment * game_context['sos_adjustment']
        )
        
        # PASSING TDs - Enhanced calculation
        qb_pass_td_per_game = qb_stats['PassingTD'] / games_played
        
        # Red zone efficiency
        red_zone_efficiency = min(2.0, team_points / 24.0)
        
        projections['PassingTDs'] = (
            (qb_pass_td_per_game * qb_efficiency + 
             def_pass_td_allowed * def_pass_efficiency) / 2 *
            red_zone_efficiency * game_context['sos_adjustment']
        )
        
        # INTERCEPTIONS - Enhanced calculation
        qb_int_per_game = qb_stats['PassingInt'] / games_played
        
        # Pressure adjustment based on defense efficiency
        pressure_factor = max(0.5, min(2.0, 
            -game_context.get('opponent_team_def_epa', 0.0) / 0.03 + 1.0)) if game_context.get('opponent_team_def_epa', 0) != 0 else 1.0
        
        projections['Interceptions'] = (
            (qb_int_per_game + def_interceptions) / 2 *
            pressure_factor
        )
        
        # RUSHING YARDS - Enhanced calculation
        qb_rush_yds_per_game = qb_stats['RushingYDS'] / games_played
        
        # Game script adjustment (underdogs run more)
        rush_game_script = 1.2 if game_context['spread'] < -3 else 0.8 if game_context['spread'] > 3 else 1.0
        
        projections['RushingYards'] = (
            qb_rush_yds_per_game * rush_game_script * game_context['sos_adjustment']
        )
        
        # RUSHING TDs - Enhanced calculation
        qb_rush_td_per_game = qb_stats['RushingTD'] / games_played
        
        projections['RushingTDs'] = (
            qb_rush_td_per_game * rush_game_script * red_zone_efficiency
        )
        
        # PASS ATTEMPTS & COMPLETIONS - Enhanced calculation
        # Game script adjustment for pass attempts
        pass_attempts_script = 1.15 if game_context['spread'] < 0 else 0.85  # Underdogs pass more
        
        projections['PassAttempts'] = (
            def_attempts_allowed * pass_attempts_script * total_adjustment * team_pass_share
        )
        
        defense_completion_pct = def_completions_allowed / def_attempts_allowed if def_attempts_allowed > 0 else 0.65
        projections['Completions'] = projections['PassAttempts'] * defense_completion_pct
        
        # FANTASY POINTS - Standard QB scoring
        projections['FantasyPoints'] = (
            projections['PassingYards'] * 0.04 +
            projections['PassingTDs'] * 4 +
            projections['RushingYards'] * 0.1 +
            projections['RushingTDs'] * 6 -
            projections['Interceptions'] * 2
        )
        
        # Add confidence metrics
        projections['Confidence'] = confidence
        projections['GameScript'] = 'Positive' if game_context['spread'] > 0 else 'Negative' if game_context['spread'] < 0 else 'Neutral'
        projections['ExpectedGameTotal'] = game_context['expected_total']
        
        return projections, qb_stats, defense_stats, game_context
    
    def get_available_rbs(self):
        return self.rb_data['PlayerName'].tolist() if not self.rb_data.empty else []
    
    def get_available_qbs(self):
        return self.qb_data['PlayerName'].tolist() if not self.qb_data.empty else []
    
    def get_available_teams(self):
        """Get available teams from defense data"""
        teams = set()
        
        # Get teams from defense data
        if isinstance(self.defense_data, list):
            for team in self.defense_data:
                if 'Team' in team:
                    teams.add(team['Team'])
        elif isinstance(self.defense_data, dict):
            if 'teams' in self.defense_data and isinstance(self.defense_data['teams'], list):
                for team in self.defense_data['teams']:
                    if 'Team' in team:
                        teams.add(team['Team'])
            else:
                # Assume dict keys are team names
                teams.update(self.defense_data.keys())
        
        # Also get teams from player data
        if not self.rb_data.empty:
            teams.update(self.rb_data['Team'].unique())
        if not self.qb_data.empty:
            teams.update(self.qb_data['Team'].unique())
            
        return sorted(list(teams))

# [Rest of the Streamlit app code remains the same...]
