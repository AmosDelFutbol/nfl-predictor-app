import pandas as pd
import numpy as np
import json
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
        self.odds_data = None  # Make sure this is initialized
        
        try:
            # Load player data
            self.rb_data = pd.read_csv('RB_season.csv')
            self.qb_data = pd.read_csv('QB_season.csv')
            st.success("✅ Player data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading player data: {e}")
        
        # Load JSON data with error handling for each file
        self._load_json_data()
        
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
                # Handle different schedule formats
                if 'Week 10' in schedule_data:
                    self.schedule_data = schedule_data['Week 10']
                else:
                    self.schedule_data = schedule_data
            st.success("✅ Schedule data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load schedule data: {e}")
            self.schedule_data = []
        
        # Load odds data - FIXED: Make sure odds_data is always initialized
        try:
            with open('week_10_odds.json', 'r') as f:
                self.odds_data = json.load(f)
            st.success("✅ Odds data loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load odds data: {e}")
            self.odds_data = []  # Always initialize as empty list
    
    def _clean_data(self):
        """Clean the data by filling NaN values with appropriate defaults"""
        # Clean RB data
        if self.rb_data is not None:
            rb_numeric_columns = ['RushingYDS', 'RushingTD', 'TouchCarries', 'ReceivingYDS', 'ReceivingRec', 'ReceivingTD']
            for col in rb_numeric_columns:
                if col in self.rb_data.columns:
                    self.rb_data[col] = self.rb_data[col].fillna(0)
        
        # Clean QB data
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
    
    def _get_offense_stats(self, team_name):
        """Get offense stats for a specific team"""
        if self.offense_data is None or not self.offense_data:
            return None
        for team in self.offense_data:
            if team['Team'] == team_name:
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
        
        # Get game odds - FIXED: Check if odds_data exists and is not None
        if hasattr(self, 'odds_data') and self.odds_data is not None:
            for odds in self.odds_data:
                if (odds.get('home_team') == player_team and odds.get('away_team') == opponent_team) or \
                   (odds.get('home_team') == opponent_team and odds.get('away_team') == player_team):
                    if odds.get('market') == 'totals' and odds.get('point'):
                        context['expected_total'] = self._safe_float(odds['point'], 45.0)
                    elif odds.get('market') == 'spreads' and odds.get('point'):
                        context['spread'] = self._safe_float(odds['point'], 0.0)
        
        return context
    
    def estimate_vegas_line(self, projection, position, stat_type):
        """Estimate what the Vegas line might have been based on projection"""
        # Ensure projection is a valid number
        safe_projection = self._safe_float(projection, 0.0)
        
        # These are rough estimates based on common Vegas lines
        if position == 'RB':
            if stat_type == 'rushing_yards':
                # Vegas typically sets lines close to projections but rounded
                estimated_line = round(safe_projection / 5) * 5  # Round to nearest 5
                return max(45, min(125, estimated_line))  # Reasonable range for RBs
            elif stat_type == 'rushing_tds':
                estimated_line = round(safe_projection * 2) / 2  # Round to nearest 0.5
                return max(0.5, min(2.5, estimated_line))
            elif stat_type == 'receiving_yards':
                estimated_line = round(safe_projection / 5) * 5
                return max(15, min(80, estimated_line))
            elif stat_type == 'receptions':
                estimated_line = round(safe_projection * 2) / 2
                return max(2.5, min(8.5, estimated_line))
        
        elif position == 'QB':
            if stat_type == 'passing_yards':
                estimated_line = round(safe_projection / 10) * 10  # Round to nearest 10
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
        # Check if we have the necessary data
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
        
        raise ValueError(f"Player '{player_name}' not found in RB or QB data")
    
    def _project_rb_rushing(self, rb_stats, opponent_team, games_played):
        """Project rushing stats for running backs"""
        player_team = rb_stats['Team']
        
        # Get defense and offense stats
        defense_stats = self._get_defense_stats(opponent_team)
        if not defense_stats:
            raise ValueError(f"Defense stats for '{opponent_team}' not found")
        
        # Get game context
        game_context = self.get_game_context(player_team, opponent_team)
        
        # Calculate projections with safe float conversions
        projections = {}
        
        # Safely get stats with defaults
        rb_rush_yds_pg = self._safe_float(rb_stats['RushingYDS']) / games_played
        def_rush_yds_allowed = self._safe_float(defense_stats.get('RUSHING YARDS PER GAME ALLOWED', 100))
        
        # Game script adjustment
        game_script = 1.15 if game_context['spread'] > 3 else 0.85 if game_context['spread'] < -3 else 1.0
        
        projections['RushingYards'] = (
            (rb_rush_yds_pg + def_rush_yds_allowed) / 2 * 
            game_script * game_context['sos_adjustment']
        )
        
        # RUSHING TDs
        rb_rush_td_pg = self._safe_float(rb_stats['RushingTD']) / games_played
        def_rush_td_allowed = self._safe_float(defense_stats.get('RUSHING TD PER GAME ALLOWED', 1.0))
        
        projections['RushingTDs'] = (
            (rb_rush_td_pg + def_rush_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # CARRIES
        rb_carries_pg = self._safe_float(rb_stats['TouchCarries']) / games_played
        def_rush_attempts_allowed = self._safe_float(defense_stats.get('RUSHING ATTEMPTS ALLOWED', 25))
        
        projections['Carries'] = (
            (rb_carries_pg + def_rush_attempts_allowed * 0.3) / 1.3 * game_script
        )
        
        # FANTASY POINTS (Rushing only)
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
        
        # RUSHING YARDS
        qb_rush_yds_pg = self._safe_float(qb_stats['RushingYDS']) / games_played
        
        projections['RushingYards'] = qb_rush_yds_pg * game_context['sos_adjustment']
        
        # RUSHING TDs
        qb_rush_td_pg = self._safe_float(qb_stats['RushingTD']) / games_played
        
        projections['RushingTDs'] = qb_rush_td_pg * (game_context['expected_total'] / 45.0)
        
        # CARRIES (estimate based on rushing yards)
        projections['Carries'] = projections['RushingYards'] / 4.5  # Estimate 4.5 yards per carry
        
        # FANTASY POINTS (Rushing only)
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
        
        # PASSING YARDS
        qb_pass_yds_pg = self._safe_float(qb_stats['PassingYDS']) / games_played
        def_pass_yds_allowed = self._safe_float(defense_stats.get('PASSING YARDS ALLOWED', 230))
        
        projections['PassingYards'] = (
            (qb_pass_yds_pg + def_pass_yds_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # PASSING TDs
        qb_pass_td_pg = self._safe_float(qb_stats['PassingTD']) / games_played
        def_pass_td_allowed = self._safe_float(defense_stats.get('PASSING TD ALLOWED', 1.5))
        
        projections['PassingTDs'] = (
            (qb_pass_td_pg + def_pass_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # INTERCEPTIONS
        qb_int_pg = self._safe_float(qb_stats['PassingInt']) / games_played
        
        projections['Interceptions'] = (
            (qb_int_pg + self._safe_float(defense_stats.get('INTERCENTIONS', 1.0))) / 2
        )
        
        # PASS ATTEMPTS & COMPLETIONS
        def_attempts_allowed = self._safe_float(defense_stats.get('PASSING ATTEMPTS ALLOWED', 35))
        def_completions_allowed = self._safe_float(defense_stats.get('PASSING COMPLETIONS ALLOWED', 23))
        
        projections['PassAttempts'] = def_attempts_allowed * (game_context['expected_total'] / 45.0)
        defense_completion_pct = def_completions_allowed / def_attempts_allowed if def_attempts_allowed > 0 else 0.65
        projections['Completions'] = projections['PassAttempts'] * defense_completion_pct
        
        # FANTASY POINTS (Passing only)
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
        
        # Sort alphabetically
        playing_players['rushers'] = sorted(playing_players['rushers'], key=lambda x: x['name'])
        playing_players['passers'] = sorted(playing_players['passers'], key=lambda x: x['name'])
        
        return playing_players
    
    def generate_all_rushing_projections(self, games_played=9):
        """Generate rushing projections for all players playing this week"""
        playing_players = self.get_players_playing_this_week()
        projections = []
        
        for player_info in playing_players['rushers']:
            try:
                player_proj, player_stats, defense_stats, game_context, position = self.project_rushing_stats(
                    player_info['name'], player_info['opponent'], games_played
                )
                
                # Calculate total fantasy points (rushing + receiving if available)
                total_fantasy_points = player_proj['FantasyPoints']
                
                # Add receiving projections for RBs if available
                if position == 'RB' and 'ReceivingYDS' in player_stats and 'ReceivingTD' in player_stats:
                    receiving_yds_pg = self._safe_float(player_stats['ReceivingYDS']) / games_played
                    receiving_tds_pg = self._safe_float(player_stats['ReceivingTD']) / games_played
                    
                    # Simple receiving projection
                    receiving_yds_proj = receiving_yds_pg * game_context['sos_adjustment']
                    receiving_tds_proj = receiving_tds_pg * (game_context['expected_total'] / 45.0)
                    
                    receiving_points = receiving_yds_proj * 0.1 + receiving_tds_proj * 6
                    total_fantasy_points += receiving_points
                
                projections.append({
                    'Player': player_info['name'],
                    'Team': player_info['team'],
                    'Opponent': player_info['opponent'],
                    'Position': position,
                    'RushingYards': player_proj['RushingYards'],
                    'RushingTDs': player_proj['RushingTDs'],
                    'Carries': player_proj['Carries'],
                    'FantasyPoints': total_fantasy_points,
                    'GameTotal': game_context['expected_total'],
                    'Spread': game_context['spread']
                })
                
            except Exception as e:
                st.warning(f"Could not generate projection for {player_info['name']}: {e}")
                continue
        
        return sorted(projections, key=lambda x: x['FantasyPoints'], reverse=True)
    
    def generate_all_passing_projections(self, games_played=9):
        """Generate passing projections for all QBs playing this week"""
        playing_players = self.get_players_playing_this_week()
        projections = []
        
        for player_info in playing_players['passers']:
            try:
                player_proj, player_stats, defense_stats, game_context = self.project_passing_stats(
                    player_info['name'], player_info['opponent'], games_played
                )
                
                # Add rushing points for QBs
                try:
                    rush_proj, _, _, _, _ = self.project_rushing_stats(
                        player_info['name'], player_info['opponent'], games_played
                    )
                    rushing_points = rush_proj['FantasyPoints']
                except:
                    rushing_points = 0
                
                total_fantasy_points = player_proj['FantasyPoints'] + rushing_points
                
                projections.append({
                    'Player': player_info['name'],
                    'Team': player_info['team'],
                    'Opponent': player_info['opponent'],
                    'Position': 'QB',
                    'PassingYards': player_proj['PassingYards'],
                    'PassingTDs': player_proj['PassingTDs'],
                    'Interceptions': player_proj['Interceptions'],
                    'PassAttempts': player_proj['PassAttempts'],
                    'Completions': player_proj['Completions'],
                    'FantasyPoints': total_fantasy_points,
                    'GameTotal': game_context['expected_total'],
                    'Spread': game_context['spread']
                })
                
            except Exception as e:
                st.warning(f"Could not generate projection for {player_info['name']}: {e}")
                continue
        
        return sorted(projections, key=lambda x: x['FantasyPoints'], reverse=True)

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
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏃‍♂️ All Rushers", "🎯 All Passers", "🔍 Single Player"])
    
    with tab1:
        st.subheader("All Rushing Projections for This Week")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            games_played_rush = st.number_input("Games Played", min_value=1, max_value=17, value=9, key="all_rush_games")
            if st.button("Generate All Rushing Projections", type="primary", key="all_rush_btn"):
                with st.spinner("Generating projections for all rushers..."):
                    all_rush_projections = projector.generate_all_rushing_projections(games_played_rush)
                    st.session_state.all_rush_projections = all_rush_projections
        
        with col1:
            search_rush = st.text_input("🔍 Search Rushers", placeholder="Type to search players...")
        
        # Display all rushing projections
        if 'all_rush_projections' in st.session_state and st.session_state.all_rush_projections:
            projections_df = pd.DataFrame(st.session_state.all_rush_projections)
            
            # Filter based on search
            if search_rush:
                filtered_df = projections_df[projections_df['Player'].str.contains(search_rush, case=False, na=False)]
            else:
                filtered_df = projections_df
            
            # Display metrics
            st.metric("Total Players Projected", len(filtered_df))
            
            # Display table
            st.dataframe(
                filtered_df,
                column_config={
                    "Player": "Player",
                    "Team": "Team", 
                    "Opponent": "Opponent",
                    "Position": "Position",
                    "RushingYards": st.column_config.NumberColumn("Rush Yds", format="%.1f"),
                    "RushingTDs": st.column_config.NumberColumn("Rush TDs", format="%.1f"),
                    "Carries": st.column_config.NumberColumn("Carries", format="%.1f"),
                    "FantasyPoints": st.column_config.NumberColumn("FP", format="%.1f"),
                    "GameTotal": st.column_config.NumberColumn("Game Total", format="%.1f"),
                    "Spread": st.column_config.NumberColumn("Spread", format="%+.1f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Rushing Projections as CSV",
                data=csv,
                file_name="rushing_projections.csv",
                mime="text/csv"
            )
        else:
            st.info("Click 'Generate All Rushing Projections' to see all players")
    
    with tab2:
        st.subheader("All Passing Projections for This Week")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            games_played_pass = st.number_input("Games Played", min_value=1, max_value=17, value=9, key="all_pass_games")
            if st.button("Generate All Passing Projections", type="primary", key="all_pass_btn"):
                with st.spinner("Generating projections for all passers..."):
                    all_pass_projections = projector.generate_all_passing_projections(games_played_pass)
                    st.session_state.all_pass_projections = all_pass_projections
        
        with col1:
            search_pass = st.text_input("🔍 Search Passers", placeholder="Type to search QBs...")
        
        # Display all passing projections
        if 'all_pass_projections' in st.session_state and st.session_state.all_pass_projections:
            projections_df = pd.DataFrame(st.session_state.all_pass_projections)
            
            # Filter based on search
            if search_pass:
                filtered_df = projections_df[projections_df['Player'].str.contains(search_pass, case=False, na=False)]
            else:
                filtered_df = projections_df
            
            # Display metrics
            st.metric("Total QBs Projected", len(filtered_df))
            
            # Display table
            st.dataframe(
                filtered_df,
                column_config={
                    "Player": "Player",
                    "Team": "Team", 
                    "Opponent": "Opponent",
                    "Position": "Position",
                    "PassingYards": st.column_config.NumberColumn("Pass Yds", format="%.1f"),
                    "PassingTDs": st.column_config.NumberColumn("Pass TDs", format="%.1f"),
                    "Interceptions": st.column_config.NumberColumn("INTs", format="%.1f"),
                    "FantasyPoints": st.column_config.NumberColumn("FP", format="%.1f"),
                    "GameTotal": st.column_config.NumberColumn("Game Total", format="%.1f"),
                    "Spread": st.column_config.NumberColumn("Spread", format="%+.1f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Passing Projections as CSV",
                data=csv,
                file_name="passing_projections.csv",
                mime="text/csv"
            )
        else:
            st.info("Click 'Generate All Passing Projections' to see all QBs")
    
    with tab3:
        st.subheader("Single Player Projection")
        
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

if __name__ == "__main__":
    main()
