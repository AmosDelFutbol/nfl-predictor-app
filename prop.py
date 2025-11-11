import pandas as pd
import numpy as np
import json
import streamlit as st

class EnhancedNFLProjector:
    def __init__(self):
        """Initialize the enhanced projector with all available data sources"""
        try:
            # Load player data
            self.rb_data = pd.read_csv('RB_season.csv')
            self.qb_data = pd.read_csv('QB_season.csv')
            # self.wr_data = pd.read_csv('WR_season.csv')  # Add WR data
            # self.te_data = pd.read_csv('TE_season.csv')  # Add TE data
            
            # Load JSON defense and offense data
            with open('2025_NFL_DEFENSE.json', 'r') as f:
                self.defense_data = json.load(f)
            with open('2025_NFL_OFFENSE.json', 'r') as f:
                self.offense_data = json.load(f)
            
            # Load other data
            with open('nfl_strength_of_schedule.json', 'r') as f:
                self.sos_data = json.load(f)['sos_rankings']
            with open('week_10_schedule.json', 'r') as f:
                self.schedule_data = json.load(f)['Week 10']
            with open('week_10_odds.json', 'r') as f:
                self.odds_data = json.load(f)
            
            st.success("✅ All data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
    
    def _get_defense_stats(self, team_name):
        """Get defense stats for a specific team"""
        if self.defense_data is None:
            return None
        for team in self.defense_data:
            if team['Team'] == team_name:
                return team
        return None
    
    def _get_offense_stats(self, team_name):
        """Get offense stats for a specific team"""
        if self.offense_data is None:
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
            context['sos_adjustment'] = 1.0 + (self.sos_data[player_team]['combined_sos'] - 0.5) * 0.3
        
        # Get game odds
        if self.odds_data:
            for odds in self.odds_data:
                if (odds.get('home_team') == player_team and odds.get('away_team') == opponent_team) or \
                   (odds.get('home_team') == opponent_team and odds.get('away_team') == player_team):
                    if odds.get('market') == 'totals' and odds.get('point'):
                        context['expected_total'] = odds['point']
                    elif odds.get('market') == 'spreads' and odds.get('point'):
                        context['spread'] = odds['point']
        
        return context
    
    def project_rushing_stats(self, player_name, opponent_team, games_played=9):
        """Enhanced rushing projections for RBs and rushing QBs"""
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
        offense_stats = self._get_offense_stats(player_team)
        
        if not defense_stats:
            raise ValueError(f"Defense stats for '{opponent_team}' not found")
        
        # Get game context
        game_context = self.get_game_context(player_team, opponent_team)
        
        # Calculate projections
        projections = {}
        
        # RUSHING YARDS
        rb_rush_yds_pg = rb_stats['RushingYDS'] / games_played
        def_rush_yds_allowed = defense_stats['RUSHING YARDS PER GAME ALLOWED']
        
        # Game script adjustment
        game_script = 1.15 if game_context['spread'] > 3 else 0.85 if game_context['spread'] < -3 else 1.0
        
        projections['RushingYards'] = (
            (rb_rush_yds_pg + def_rush_yds_allowed) / 2 * 
            game_script * game_context['sos_adjustment']
        )
        
        # RUSHING TDs
        rb_rush_td_pg = rb_stats['RushingTD'] / games_played
        def_rush_td_allowed = defense_stats['RUSHING TD PER GAME ALLOWED']
        
        projections['RushingTDs'] = (
            (rb_rush_td_pg + def_rush_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # CARRIES
        rb_carries_pg = rb_stats['TouchCarries'] / games_played
        def_rush_attempts_allowed = defense_stats['RUSHING ATTEMPTS ALLOWED']
        
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
        qb_rush_yds_pg = qb_stats['RushingYDS'] / games_played
        
        projections['RushingYards'] = qb_rush_yds_pg * game_context['sos_adjustment']
        
        # RUSHING TDs
        qb_rush_td_pg = qb_stats['RushingTD'] / games_played
        
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
        qb_pass_yds_pg = qb_stats['PassingYDS'] / games_played
        def_pass_yds_allowed = defense_stats['PASSING YARDS ALLOWED']
        
        projections['PassingYards'] = (
            (qb_pass_yds_pg + def_pass_yds_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # PASSING TDs
        qb_pass_td_pg = qb_stats['PassingTD'] / games_played
        def_pass_td_allowed = defense_stats['PASSING TD ALLOWED']
        
        projections['PassingTDs'] = (
            (qb_pass_td_pg + def_pass_td_allowed) / 2 *
            (game_context['expected_total'] / 45.0) * game_context['sos_adjustment']
        )
        
        # INTERCEPTIONS
        qb_int_pg = qb_stats['PassingInt'] / games_played
        
        projections['Interceptions'] = (
            (qb_int_pg + defense_stats['INTERCENTIONS']) / 2
        )
        
        # PASS ATTEMPTS & COMPLETIONS
        def_attempts_allowed = defense_stats['PASSING ATTEMPTS ALLOWED']
        def_completions_allowed = defense_stats['PASSING COMPLETIONS ALLOWED']
        
        projections['PassAttempts'] = def_attempts_allowed * (game_context['expected_total'] / 45.0)
        defense_completion_pct = def_completions_allowed / def_attempts_allowed
        projections['Completions'] = projections['PassAttempts'] * defense_completion_pct
        
        # FANTASY POINTS (Passing only)
        projections['FantasyPoints'] = (
            projections['PassingYards'] * 0.04 +
            projections['PassingTDs'] * 4 -
            projections['Interceptions'] * 2
        )
        
        return projections, qb_stats, defense_stats, game_context
    
    def get_available_rushers(self):
        """Get all players with rushing stats"""
        rushers = self.rb_data['PlayerName'].tolist() + self.qb_data['PlayerName'].tolist()
        return sorted(list(set(rushers)))
    
    def get_available_passers(self):
        return self.qb_data['PlayerName'].tolist()
    
    def get_available_teams(self):
        """Get available teams from defense data"""
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
    
    # Create tabs for different stats - only Rushing and Passing for now
    tab1, tab2 = st.tabs(["🏃‍♂️ Rushing", "🎯 Passing"])  # Removed Receiving tab for now
    
    with tab1:
        st.subheader("Rushing Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_rushers = projector.get_available_rushers()
            rusher_name = st.selectbox("Select Rusher", available_rushers, key="rusher")
            
            available_teams = projector.get_available_teams()
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
                    
                    # Projections in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rushing Yards", f"{projections['RushingYards']:.1f}")
                        st.metric("Rushing TDs", f"{projections['RushingTDs']:.1f}")
                    
                    with col2:
                        st.metric("Carries", f"{projections['Carries']:.1f}")
                        st.metric("Fantasy Points", f"{projections['FantasyPoints']:.1f}")
                    
                    with col3:
                        st.metric("Team", player_stats['Team'])
                        if position == 'RB':
                            st.metric("Season Rush Yds", player_stats['RushingYDS'])
                        else:
                            st.metric("Season Rush Yds", player_stats['RushingYDS'])
                    
                    # Defense info
                    with st.expander("View Defense Stats"):
                        st.write(f"**{opponent_team_rush} Rush Defense Allowed Per Game:**")
                        st.write(f"- Rushing: {defense_stats['RUSHING YARDS PER GAME ALLOWED']} yds, {defense_stats['RUSHING TD PER GAME ALLOWED']} TDs")
                        
                except Exception as e:
                    st.error(f"Error generating projection: {e}")
    
    with tab2:
        st.subheader("Passing Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_passers = projector.get_available_passers()
            qb_name = st.selectbox("Select Quarterback", available_passers, key="qb")
            
            available_teams = projector.get_available_teams()
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
                        st.metric("Season Pass Yds", qb_stats['PassingYDS'])
                    
                    # Defense info
                    with st.expander("View Defense Stats"):
                        st.write(f"**{opponent_team_qb} Pass Defense Allowed Per Game:**")
                        st.write(f"- Passing: {defense_stats['PASSING YARDS ALLOWED']} yds, {defense_stats['PASSING TD ALLOWED']} TDs")
                        st.write(f"- Interceptions: {defense_stats['INTERCENTIONS']:.1f}")
                        
                except Exception as e:
                    st.error(f"Error generating projection: {e}")

    # Commented out Receiving tab for now
    """
    with tab3:
        st.subheader("Receiving Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_receivers = projector.get_available_receivers()
            receiver_name = st.selectbox("Select Receiver", available_receivers, key="receiver")
            
            available_teams = projector.get_available_teams()
            opponent_team_rec = st.selectbox("Select Opponent Team", available_teams, key="rec_opponent")
            
            games_played_rec = st.number_input("Games Played This Season", min_value=1, max_value=17, value=9, key="rec_games")
        
        with col2:
            if st.button("Generate Receiving Projection", type="primary", key="rec_btn"):
                try:
                    projections, player_stats, defense_stats, game_context, position = projector.project_receiving_stats(
                        receiver_name, opponent_team_rec, games_played_rec
                    )
                    
                    # Display results
                    st.success(f"📊 Receiving Projection for {receiver_name} ({position}) vs {opponent_team_rec}")
                    
                    # Game context
                    st.write(f"**Game Context:** Expected Total: {game_context['expected_total']} points, Spread: {game_context['spread']:+.1f}")
                    
                    # Projections in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Receiving Yards", f"{projections['ReceivingYards']:.1f}")
                        st.metric("Receiving TDs", f"{projections['ReceivingTDs']:.1f}")
                    
                    with col2:
                        st.metric("Receptions", f"{projections['Receptions']:.1f}")
                        st.metric("Targets", f"{projections['Targets']:.1f}")
                    
                    with col3:
                        st.metric("Fantasy Points", f"{projections['FantasyPoints']:.1f}")
                        st.metric("Team", player_stats['Team'])
                        st.metric("Position", position)
                    
                    # Defense info
                    with st.expander("View Defense Stats"):
                        st.write(f"**{opponent_team_rec} Pass Defense Allowed Per Game:**")
                        st.write(f"- Passing: {defense_stats['PASSING YARDS ALLOWED']} yds, {defense_stats['PASSING TD ALLOWED']} TDs")
                        st.write(f"- Completions: {defense_stats['PASSING COMPLETIONS ALLOWED']:.1f}")
                        
                except Exception as e:
                    st.error(f"Error generating projection: {e}")
    """

if __name__ == "__main__":
    main()
