import pandas as pd
import numpy as np
import json
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class EnhancedNFLProjector:
    def __init__(self):
        """Initialize the enhanced projector with all available data sources"""
        try:
            # Load base data files from GitHub
            self.rb_data = pd.read_csv('RB_season.csv')
            self.qb_data = pd.read_csv('QB_season.csv')
            self.defense_data = pd.read_csv('2025_NFL_DEFENSE.csv')
            
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
            
            st.success("✅ Enhanced NFL Projector initialized successfully!")
            st.info(f"Advanced metrics loaded for {len(self.elo_data)} teams")
            
        except Exception as e:
            st.error(f"❌ Error initializing projector: {e}")
            # Create empty dataframes to prevent further errors
            self.rb_data = pd.DataFrame()
            self.qb_data = pd.DataFrame()
            self.defense_data = pd.DataFrame()
            self.elo_data = {}
            self.sos_data = {}
        
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
        if not self.defense_data.empty:
            self.defense_data.columns = self.defense_data.columns.str.strip()
    
    def _get_team_abbreviation_from_name(self, team_name):
        """Convert team name to abbreviation"""
        return self.team_mapping.get(team_name, None)
    
    def get_game_context(self, player_team, opponent_team):
        """Get advanced game context for projections"""
        context = {
            'expected_total': 45.0,
            'spread': 0.0,
            'player_team_elo': 1500,
            'opponent_team_elo': 1500,
            'player_team_off_epa': 0.0,
            'opponent_team_def_epa': 0.0,
            'sos_adjustment': 1.0
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
        
        return min(confidence_score, 1.0)
    
    def project_rb_stats(self, rb_name, opponent_team, games_played=9):
        """Enhanced RB projections using advanced metrics"""
        
        if self.rb_data.empty:
            raise ValueError("RB data not loaded")
        
        # Get base data
        rb_row = self.rb_data[self.rb_data['PlayerName'] == rb_name]
        if rb_row.empty:
            raise ValueError(f"RB '{rb_name}' not found in data")
        
        defense_row = self.defense_data[self.defense_data['Team'] == opponent_team]
        if defense_row.empty:
            raise ValueError(f"Team '{opponent_team}' not found in data")
        
        rb_stats = rb_row.iloc[0]
        defense_stats = defense_row.iloc[0]
        
        # Get player's team
        player_team = rb_stats.get('Team', 'Unknown')
        
        # Get advanced game context
        game_context = self.get_game_context(player_team, opponent_team)
        
        # Calculate confidence
        confidence = self.calculate_confidence_factors(
            {'games_played': games_played}, 
            defense_stats, 
            game_context, 
            'RB'
        )
        
        # Enhanced projections with advanced metrics
        projections = {}
        
        # RUSHING YARDS - Enhanced calculation
        rb_rush_yds_per_game = rb_stats['RushingYDS'] / games_played
        def_rush_yds_allowed = defense_stats['RUSHING YARDS PER GAME ALLOWED']
        
        # Apply efficiency adjustments
        off_rush_efficiency = max(0.5, min(2.0, 
            game_context['player_team_off_epa'] / 0.05 + 1.0))  # Normalize EPA
        def_rush_efficiency = max(0.5, min(2.0, 
            -game_context['opponent_team_def_epa'] / 0.05 + 1.0))  # Negative EPA is good for defense
        
        # Game script adjustment (based on spread)
        game_script_multiplier = 1.0
        if game_context['spread'] > 3:  # Heavy favorite - positive game script for rushing
            game_script_multiplier = 1.15
        elif game_context['spread'] < -3:  # Underdog - negative game script for rushing
            game_script_multiplier = 0.85
        
        projections['RushingYards'] = (
            (rb_rush_yds_per_game * off_rush_efficiency + 
             def_rush_yds_allowed * def_rush_efficiency) / 2 *
            game_script_multiplier * game_context['sos_adjustment']
        )
        
        # RUSHING TDs - Enhanced calculation
        rb_rush_td_per_game = rb_stats['RushingTD'] / games_played
        def_rush_td_allowed = defense_stats['RUSHING TD PER GAME ALLOWED']
        
        # Red zone efficiency adjustment
        red_zone_efficiency = min(2.0, game_context['expected_total'] / 45.0)  # Higher totals = more TDs
        
        projections['RushingTDs'] = (
            (rb_rush_td_per_game + def_rush_td_allowed) / 2 *
            red_zone_efficiency * game_context['sos_adjustment']
        )
        
        # RECEIVING YARDS - Enhanced calculation
        rb_rec_yds_per_game = rb_stats['ReceivingYDS'] / games_played
        def_pass_yds_allowed = defense_stats['PASSING YARDS ALLOWED']
        
        # Pass game efficiency adjustment
        qb_efficiency = max(0.7, min(1.3, 
            game_context.get('qb_rating', 1.0)))
        
        # Game script adjustment for receiving (inverse of rushing)
        rec_game_script = 1.0 / game_script_multiplier if game_script_multiplier != 1.0 else 1.0
        
        projections['ReceivingYards'] = (
            rb_rec_yds_per_game * 
            (def_pass_yds_allowed / 231.8) *  # Normalize
            qb_efficiency * rec_game_script * game_context['sos_adjustment']
        )
        
        # RECEPTIONS - Enhanced calculation
        rb_rec_per_game = rb_stats['ReceivingRec'] / games_played
        def_completions_allowed = defense_stats['PASSING COMPLETIONS ALLOWED']
        
        projections['Receptions'] = (
            rb_rec_per_game * 
            (def_completions_allowed / 24.1) *  # Normalize
            qb_efficiency * rec_game_script
        )
        
        # RECEIVING TDs - Enhanced calculation
        rb_rec_td_per_game = rb_stats['ReceivingTD'] / games_played
        def_pass_td_allowed = defense_stats['PASSING TD ALLOWED']
        
        projections['ReceivingTDs'] = (
            (rb_rec_td_per_game + def_pass_td_allowed * 0.3) / 2 *
            red_zone_efficiency * game_context['sos_adjustment']
        )
        
        # CARRIES - Enhanced calculation
        rb_carries_per_game = rb_stats['TouchCarries'] / games_played
        def_rush_attempts_allowed = defense_stats['RUSHING ATTEMPTS ALLOWED']
        
        projections['Carries'] = (
            (rb_carries_per_game * off_rush_efficiency + 
             def_rush_attempts_allowed * 0.3) / 2 *
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
        
        defense_row = self.defense_data[self.defense_data['Team'] == opponent_team]
        if defense_row.empty:
            raise ValueError(f"Team '{opponent_team}' not found in data")
        
        qb_stats = qb_row.iloc[0]
        defense_stats = defense_row.iloc[0]
        
        # Get player's team
        player_team = qb_stats.get('Team', 'Unknown')
        
        # Get advanced game context
        game_context = self.get_game_context(player_team, opponent_team)
        
        # Calculate confidence
        confidence = self.calculate_confidence_factors(
            {'games_played': games_played}, 
            defense_stats, 
            game_context, 
            'QB'
        )
        
        # Enhanced projections with advanced metrics
        projections = {}
        
        # PASSING YARDS - Enhanced calculation
        qb_pass_yds_per_game = qb_stats['PassingYDS'] / games_played
        def_pass_yds_allowed = defense_stats['PASSING YARDS ALLOWED']
        
        # Efficiency adjustments
        qb_efficiency = max(0.7, min(1.3, 
            game_context.get('player_team_off_epa', 0.0) / 0.05 + 1.0))
        def_pass_efficiency = max(0.7, min(1.3, 
            -game_context.get('opponent_team_def_epa', 0.0) / 0.05 + 1.0))
        
        # Game total adjustment
        total_adjustment = game_context['expected_total'] / 45.0
        
        projections['PassingYards'] = (
            (qb_pass_yds_per_game * qb_efficiency + 
             def_pass_yds_allowed * def_pass_efficiency) / 2 *
            total_adjustment * game_context['sos_adjustment']
        )
        
        # PASSING TDs - Enhanced calculation
        qb_pass_td_per_game = qb_stats['PassingTD'] / games_played
        def_pass_td_allowed = defense_stats['PASSING TD ALLOWED']
        
        # Red zone efficiency
        red_zone_efficiency = min(2.0, game_context['expected_total'] / 45.0)
        
        projections['PassingTDs'] = (
            (qb_pass_td_per_game * qb_efficiency + 
             def_pass_td_allowed * def_pass_efficiency) / 2 *
            red_zone_efficiency * game_context['sos_adjustment']
        )
        
        # INTERCEPTIONS - Enhanced calculation
        qb_int_per_game = qb_stats['PassingInt'] / games_played
        def_interceptions = defense_stats['INTERCENTIONS']
        
        # Pressure adjustment based on defense efficiency
        pressure_factor = max(0.5, min(2.0, 
            -game_context.get('opponent_team_def_epa', 0.0) / 0.03 + 1.0))
        
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
        def_attempts_allowed = defense_stats['PASSING ATTEMPTS ALLOWED']
        def_completions_allowed = defense_stats['PASSING COMPLETIONS ALLOWED']
        
        # Game script adjustment for pass attempts
        pass_attempts_script = 1.15 if game_context['spread'] < 0 else 0.85  # Underdogs pass more
        
        projections['PassAttempts'] = (
            def_attempts_allowed * pass_attempts_script * total_adjustment
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
        return self.defense_data['Team'].tolist() if not self.defense_data.empty else []

def main():
    st.set_page_config(
        page_title="NFL Player Projections",
        page_icon="🏈",
        layout="wide"
    )
    
    st.title("🏈 Enhanced NFL Player Projections")
    st.markdown("### Advanced Player Stats Projections with ELO & Efficiency Metrics")
    
    # Initialize the projector
    if 'projector' not in st.session_state:
        st.session_state.projector = EnhancedNFLProjector()
    
    projector = st.session_state.projector
    
    # Check if data loaded successfully
    if projector.rb_data.empty or projector.qb_data.empty or projector.defense_data.empty:
        st.error("❌ Could not load player data files. Please ensure RB_season.csv, QB_season.csv, and 2025_NFL_DEFENSE.csv are in your repository.")
        return
    
    # Sidebar for player selection
    st.sidebar.header("Player Selection")
    
    position = st.sidebar.selectbox("Select Position", ["Running Back", "Quarterback"])
    
    if position == "Running Back":
        available_rbs = projector.get_available_rbs()
        if available_rbs:
            rb_name = st.sidebar.selectbox("Select Running Back", available_rbs)
            available_teams = projector.get_available_teams()
            opponent_team = st.sidebar.selectbox("Select Opponent Team", available_teams)
            games_played = st.sidebar.number_input("Games Played This Season", min_value=1, max_value=17, value=9)
            
            if st.sidebar.button("Generate RB Projection"):
                try:
                    projections, rb_stats, defense_stats, game_context = projector.project_rb_stats(
                        rb_name, opponent_team, games_played
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"📊 {rb_name} Projection vs {opponent_team}")
                        
                        # Confidence indicator
                        confidence = projections['Confidence']
                        if confidence >= 0.7:
                            st.success(f"🟢 High Confidence Projection ({confidence:.0%})")
                        elif confidence >= 0.5:
                            st.warning(f"🟡 Medium Confidence Projection ({confidence:.0%})")
                        else:
                            st.error(f"🔴 Low Confidence Projection ({confidence:.0%})")
                        
                        # Game context
                        st.write(f"**Game Context:**")
                        st.write(f"- Expected Total: {projections['ExpectedGameTotal']} points")
                        st.write(f"- Game Script: {projections['GameScript']}")
                        st.write(f"- Spread: {game_context['spread']:+.1f}")
                    
                    with col2:
                        st.subheader("Projected Stats")
                        
                        # Create metrics
                        col2a, col2b, col2c = st.columns(3)
                        
                        with col2a:
                            st.metric("Rushing Yards", f"{projections['RushingYards']:.1f}")
                            st.metric("Rushing TDs", f"{projections['RushingTDs']:.1f}")
                            st.metric("Carries", f"{projections['Carries']:.1f}")
                        
                        with col2b:
                            st.metric("Receiving Yards", f"{projections['ReceivingYards']:.1f}")
                            st.metric("Receiving TDs", f"{projections['ReceivingTDs']:.1f}")
                            st.metric("Receptions", f"{projections['Receptions']:.1f}")
                        
                        with col2c:
                            st.metric("Fantasy Points", f"{projections['FantasyPoints']:.1f}")
                    
                    # Player stats and defense stats
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader(f"{rb_name} Season Stats")
                        st.write(f"**Rushing:** {rb_stats['RushingYDS']} yards, {rb_stats['RushingTD']} TDs")
                        st.write(f"**Receiving:** {rb_stats['ReceivingYDS']} yards, {rb_stats['ReceivingTD']} TDs, {rb_stats['ReceivingRec']} receptions")
                        st.write(f"**Per Game:** {rb_stats['RushingYDS']/games_played:.1f} rush yds, {rb_stats['ReceivingYDS']/games_played:.1f} rec yds")
                    
                    with col4:
                        st.subheader(f"{opponent_team} Defense")
                        st.write(f"**Rushing Allowed:** {defense_stats['RUSHING YARDS PER GAME ALLOWED']} yds/game, {defense_stats['RUSHING TD PER GAME ALLOWED']} TDs/game")
                        st.write(f"**Passing Allowed:** {defense_stats['PASSING YARDS ALLOWED']} yds/game, {defense_stats['PASSING TD ALLOWED']} TDs/game")
                        
                except ValueError as e:
                    st.error(f"Error: {e}")
    
    else:  # Quarterback
        available_qbs = projector.get_available_qbs()
        if available_qbs:
            qb_name = st.sidebar.selectbox("Select Quarterback", available_qbs)
            available_teams = projector.get_available_teams()
            opponent_team = st.sidebar.selectbox("Select Opponent Team", available_teams)
            games_played = st.sidebar.number_input("Games Played This Season", min_value=1, max_value=17, value=9)
            
            if st.sidebar.button("Generate QB Projection"):
                try:
                    projections, qb_stats, defense_stats, game_context = projector.project_qb_stats(
                        qb_name, opponent_team, games_played
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"📊 {qb_name} Projection vs {opponent_team}")
                        
                        # Confidence indicator
                        confidence = projections['Confidence']
                        if confidence >= 0.7:
                            st.success(f"🟢 High Confidence Projection ({confidence:.0%})")
                        elif confidence >= 0.5:
                            st.warning(f"🟡 Medium Confidence Projection ({confidence:.0%})")
                        else:
                            st.error(f"🔴 Low Confidence Projection ({confidence:.0%})")
                        
                        # Game context
                        st.write(f"**Game Context:**")
                        st.write(f"- Expected Total: {projections['ExpectedGameTotal']} points")
                        st.write(f"- Game Script: {projections['GameScript']}")
                        st.write(f"- Spread: {game_context['spread']:+.1f}")
                    
                    with col2:
                        st.subheader("Projected Stats")
                        
                        # Create metrics
                        col2a, col2b, col2c = st.columns(3)
                        
                        with col2a:
                            st.metric("Passing Yards", f"{projections['PassingYards']:.1f}")
                            st.metric("Passing TDs", f"{projections['PassingTDs']:.1f}")
                            st.metric("Interceptions", f"{projections['Interceptions']:.1f}")
                        
                        with col2b:
                            st.metric("Pass Attempts", f"{projections['PassAttempts']:.1f}")
                            st.metric("Completions", f"{projections['Completions']:.1f}")
                            st.metric("Rushing Yards", f"{projections['RushingYards']:.1f}")
                        
                        with col2c:
                            st.metric("Rushing TDs", f"{projections['RushingTDs']:.1f}")
                            st.metric("Fantasy Points", f"{projections['FantasyPoints']:.1f}")
                    
                    # Player stats and defense stats
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader(f"{qb_name} Season Stats")
                        st.write(f"**Passing:** {qb_stats['PassingYDS']} yards, {qb_stats['PassingTD']} TDs, {qb_stats['PassingInt']} INTs")
                        st.write(f"**Rushing:** {qb_stats['RushingYDS']} yards, {qb_stats['RushingTD']} TDs")
                        st.write(f"**Per Game:** {qb_stats['PassingYDS']/games_played:.1f} pass yds, {qb_stats['RushingYDS']/games_played:.1f} rush yds")
                    
                    with col4:
                        st.subheader(f"{opponent_team} Defense")
                        st.write(f"**Passing Allowed:** {defense_stats['PASSING YARDS ALLOWED']} yds/game, {defense_stats['PASSING TD ALLOWED']} TDs/game")
                        st.write(f"**Rushing Allowed:** {defense_stats['RUSHING YARDS PER GAME ALLOWED']} yds/game, {defense_stats['RUSHING TD PER GAME ALLOWED']} TDs/game")
                        st.write(f"**Interceptions:** {defense_stats['INTERCENTIONS']:.1f}/game")
                        
                except ValueError as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
