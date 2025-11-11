import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class EnhancedNFLProjector:
    def __init__(self, rb_file, qb_file, defense_file):
        """Initialize the enhanced projector with all available data sources"""
        # Load base data files
        self.rb_data = pd.read_csv(rb_file)
        self.qb_data = pd.read_csv(qb_file)
        self.defense_data = pd.read_csv(defense_file)
        
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
        
        print("Enhanced NFL Projector initialized successfully!")
        print(f"Advanced metrics loaded for {len(self.elo_data)} teams")
        
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
            print(f"Warning: Could not load ELO data: {e}")
            return {}
    
    def _load_sos_data(self):
        """Load strength of schedule data"""
        try:
            with open('nfl_strength_of_schedule.json', 'r') as f:
                sos_data = json.load(f)
            return sos_data.get('sos_rankings', {})
        except:
            print("Warning: Could not load SOS data")
            return {}
    
    def _load_schedule_data(self):
        """Load schedule data for game context"""
        try:
            with open('week_10_schedule.json', 'r') as f:
                schedule_data = json.load(f)
            return schedule_data.get('Week 10', [])
        except:
            print("Warning: Could not load schedule data")
            return []
    
    def _load_odds_data(self):
        """Load odds data for game totals and spreads"""
        try:
            with open('week_10_odds.json', 'r') as f:
                odds_data = json.load(f)
            return odds_data
        except:
            print("Warning: Could not load odds data")
            return []
    
    def _clean_data(self):
        """Clean all dataframes"""
        self.rb_data.columns = self.rb_data.columns.str.strip()
        self.qb_data.columns = self.qb_data.columns.str.strip()
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
        
        # Consistency confidence (low variance in performance)
        if position == 'RB':
            rush_std = player_stats.get('rush_std', 20)
            if rush_std < 15:
                confidence_score += 0.1
        elif position == 'QB':
            pass_std = player_stats.get('pass_std', 50)
            if pass_std < 30:
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
        return self.rb_data['PlayerName'].tolist()
    
    def get_available_qbs(self):
        return self.qb_data['PlayerName'].tolist()
    
    def get_available_teams(self):
        return self.defense_data['Team'].tolist()
    
    def get_games_played(self, player_name, position):
        """Get games played with validation"""
        while True:
            try:
                games_input = input(f"How many games has {player_name} played this season? (default: 9): ").strip()
                if games_input == "":
                    return 9
                games = int(games_input)
                if games < 1 or games > 17:
                    print("Please enter a number between 1 and 17")
                    continue
                return games
            except ValueError:
                print("Please enter a valid number")
    
    def display_enhanced_rb_projections(self, rb_name, opponent_team):
        """Display enhanced RB projections with advanced metrics"""
        try:
            games_played = self.get_games_played(rb_name, 'RB')
            projections, rb_stats, defense_stats, game_context = self.project_rb_stats(rb_name, opponent_team, games_played)
            
            print(f"\n{'='*70}")
            print(f"ENHANCED RB PROJECTION: {rb_name} vs {opponent_team}")
            print(f"{'='*70}")
            
            # Basic stats
            print(f"\n{rb_name} 2025 Season Stats ({games_played} games):")
            print(f"  Rushing: {rb_stats['RushingYDS']} yards, {rb_stats['RushingTD']} TDs")
            print(f"  Receiving: {rb_stats['ReceivingYDS']} yards, {rb_stats['ReceivingTD']} TDs, {rb_stats['ReceivingRec']} receptions")
            print(f"  Per Game: {rb_stats['RushingYDS']/games_played:.1f} rush yds, {rb_stats['ReceivingYDS']/games_played:.1f} rec yds")
            
            # Advanced context
            print(f"\n🎯 ADVANCED GAME CONTEXT:")
            print(f"  Expected Total: {game_context['expected_total']} points")
            print(f"  Spread: {game_context['spread']:+.1f}")
            print(f"  Game Script: {projections['GameScript']}")
            print(f"  Projection Confidence: {projections['Confidence']:.0%}")
            
            print(f"\n{opponent_team} Defense Allowed Per Game:")
            print(f"  Rushing: {defense_stats['RUSHING YARDS PER GAME ALLOWED']} yards, {defense_stats['RUSHING TD PER GAME ALLOWED']} TDs")
            print(f"  Passing: {defense_stats['PASSING YARDS ALLOWED']} yards, {defense_stats['PASSING TD ALLOWED']} TDs")
            
            print(f"\n📊 ENHANCED PROJECTIONS vs {opponent_team}:")
            print(f"  Carries: {projections['Carries']:.1f}")
            print(f"  Rushing Yards: {projections['RushingYards']:.1f}")
            print(f"  Rushing TDs: {projections['RushingTDs']:.1f}")
            print(f"  Receptions: {projections['Receptions']:.1f}")
            print(f"  Receiving Yards: {projections['ReceivingYards']:.1f}")
            print(f"  Receiving TDs: {projections['ReceivingTDs']:.1f}")
            print(f"  Fantasy Points (PPR): {projections['FantasyPoints']:.1f}")
            
            # Confidence indicator
            confidence = projections['Confidence']
            if confidence >= 0.7:
                confidence_indicator = "🟢 HIGH CONFIDENCE"
            elif confidence >= 0.5:
                confidence_indicator = "🟡 MEDIUM CONFIDENCE"
            else:
                confidence_indicator = "🔴 LOW CONFIDENCE"
            
            print(f"\n{confidence_indicator}")
            
        except ValueError as e:
            print(f"Error: {e}")
    
    def display_enhanced_qb_projections(self, qb_name, opponent_team):
        """Display enhanced QB projections with advanced metrics"""
        try:
            games_played = self.get_games_played(qb_name, 'QB')
            projections, qb_stats, defense_stats, game_context = self.project_qb_stats(qb_name, opponent_team, games_played)
            
            print(f"\n{'='*70}")
            print(f"ENHANCED QB PROJECTION: {qb_name} vs {opponent_team}")
            print(f"{'='*70}")
            
            # Basic stats
            print(f"\n{qb_name} 2025 Season Stats ({games_played} games):")
            print(f"  Passing: {qb_stats['PassingYDS']} yards, {qb_stats['PassingTD']} TDs, {qb_stats['PassingInt']} INTs")
            print(f"  Rushing: {qb_stats['RushingYDS']} yards, {qb_stats['RushingTD']} TDs")
            print(f"  Per Game: {qb_stats['PassingYDS']/games_played:.1f} pass yds, {qb_stats['RushingYDS']/games_played:.1f} rush yds")
            
            # Advanced context
            print(f"\n🎯 ADVANCED GAME CONTEXT:")
            print(f"  Expected Total: {game_context['expected_total']} points")
            print(f"  Spread: {game_context['spread']:+.1f}")
            print(f"  Game Script: {projections['GameScript']}")
            print(f"  Projection Confidence: {projections['Confidence']:.0%}")
            
            print(f"\n{opponent_team} Defense Allowed Per Game:")
            print(f"  Passing: {defense_stats['PASSING YARDS ALLOWED']} yards, {defense_stats['PASSING TD ALLOWED']} TDs")
            print(f"  Rushing: {defense_stats['RUSHING YARDS PER GAME ALLOWED']} yards, {defense_stats['RUSHING TD PER GAME ALLOWED']} TDs")
            print(f"  Interceptions: {defense_stats['INTERCENTIONS']:.1f}")
            
            print(f"\n📊 ENHANCED PROJECTIONS vs {opponent_team}:")
            print(f"  Pass Attempts: {projections['PassAttempts']:.1f}")
            print(f"  Completions: {projections['Completions']:.1f}")
            print(f"  Passing Yards: {projections['PassingYards']:.1f}")
            print(f"  Passing TDs: {projections['PassingTDs']:.1f}")
            print(f"  Interceptions: {projections['Interceptions']:.1f}")
            print(f"  Rushing Yards: {projections['RushingYards']:.1f}")
            print(f"  Rushing TDs: {projections['RushingTDs']:.1f}")
            print(f"  Fantasy Points: {projections['FantasyPoints']:.1f}")
            
            # Confidence indicator
            confidence = projections['Confidence']
            if confidence >= 0.7:
                confidence_indicator = "🟢 HIGH CONFIDENCE"
            elif confidence >= 0.5:
                confidence_indicator = "🟡 MEDIUM CONFIDENCE"
            else:
                confidence_indicator = "🔴 LOW CONFIDENCE"
            
            print(f"\n{confidence_indicator}")
            
        except ValueError as e:
            print(f"Error: {e}")

def main():
    # Initialize the enhanced projector
    projector = EnhancedNFLProjector(
        rb_file='C:/Users/xboxl/OneDrive/Desktop/Main/NFL/NFLHenryModel/nfl_data/MainFiles/RB_season.csv',
        qb_file='C:/Users/xboxl/OneDrive/Desktop/Main/NFL/NFLHenryModel/nfl_data/MainFiles/QB_season.csv', 
        defense_file='C:/Users/xboxl/OneDrive/Desktop/Main/NFL/NFLHenryModel/nfl_data/MainFiles/2025_NFL_DEFENSE.csv'
    )
    
    while True:
        print(f"\n{'='*50}")
        print("ENHANCED NFL PLAYER PROJECTION TOOL")
        print("Now with Advanced Metrics & Game Context")
        print(f"{'='*50}")
        
        # Show available options
        available_rbs = projector.get_available_rbs()
        available_qbs = projector.get_available_qbs()
        available_teams = projector.get_available_teams()
        
        print(f"\nAvailable RBs ({len(available_rbs)}):")
        for i, rb in enumerate(available_rbs[:8]):
            print(f"  {i+1}. {rb}")
        if len(available_rbs) > 8:
            print(f"  ... and {len(available_rbs) - 8} more")
        
        print(f"\nAvailable QBs ({len(available_qbs)}):")
        for i, qb in enumerate(available_qbs[:8]):
            print(f"  {i+1}. {qb}")
        if len(available_qbs) > 8:
            print(f"  ... and {len(available_qbs) - 8} more")
        
        print(f"\nAvailable Teams ({len(available_teams)}):")
        for i, team in enumerate(available_teams):
            print(f"  {i+1}. {team}")
        
        # Get position choice
        print(f"\nChoose position:")
        print("  1. RB (Running Back) - Enhanced Projections")
        print("  2. QB (Quarterback) - Enhanced Projections")
        position_choice = input("Enter choice (1 or 2): ").strip()
        
        if position_choice == '1':
            player_name = input("Enter RB name (or 'quit' to exit): ").strip()
            if player_name.lower() == 'quit':
                return
            opponent_team = input("Enter opponent team: ").strip()
            projector.display_enhanced_rb_projections(player_name, opponent_team)
        
        elif position_choice == '2':
            player_name = input("Enter QB name (or 'quit' to exit): ").strip()
            if player_name.lower() == 'quit':
                return
            opponent_team = input("Enter opponent team: ").strip()
            projector.display_enhanced_qb_projections(player_name, opponent_team)
        
        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue
        
        # Ask to continue
        continue_choice = input("\nMake another projection? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break

if __name__ == "__main__":
    main()
