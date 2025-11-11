import pandas as pd
import numpy as np
import json
import streamlit as st

class SimpleNFLProjector:
    def __init__(self):
        """Simple projector that shows exactly what's loading"""
        st.header("🔍 File Loading Status")
        
        # Test loading each file and show status
        self.loading_status = {}
        
        # Player data files
        self.loading_status['RB_season.csv'] = self._test_load_csv('RB_season.csv')
        self.loading_status['QB_season.csv'] = self._test_load_csv('QB_season.csv')
        
        # JSON files
        self.loading_status['2025_NFL_DEFENSE.json'] = self._test_load_json('2025_NFL_DEFENSE.json')
        self.loading_status['2025_NFL_OFFENSE.json'] = self._test_load_json('2025_NFL_OFFENSE.json')
        
        # Other data files
        self.loading_status['teams_power_rating.csv'] = self._test_load_csv('teams_power_rating.csv')
        self.loading_status['nfl_strength_of_schedule.json'] = self._test_load_json('nfl_strength_of_schedule.json')
        self.loading_status['week_10_schedule.json'] = self._test_load_json('week_10_schedule.json')
        self.loading_status['week_10_odds.json'] = self._test_load_json('week_10_odds.json')
        
        # Display loading results
        self._show_loading_results()
        
        # Load the data that's available
        self._load_available_data()
    
    def _test_load_csv(self, filename):
        """Test if a CSV file can be loaded"""
        try:
            df = pd.read_csv(filename)
            return {'success': True, 'rows': len(df), 'columns': list(df.columns)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_load_json(self, filename):
        """Test if a JSON file can be loaded"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return {'success': True, 'type': type(data).__name__, 'keys': list(data.keys()) if isinstance(data, dict) else f'list with {len(data)} items'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _show_loading_results(self):
        """Show what files loaded successfully"""
        st.subheader("📊 File Loading Results")
        
        for filename, result in self.loading_status.items():
            if result['success']:
                if 'rows' in result:  # CSV file
                    st.success(f"✅ {filename}: {result['rows']} rows, {len(result['columns'])} columns")
                    st.code(f"Columns: {', '.join(result['columns'][:5])}..." if len(result['columns']) > 5 else f"Columns: {', '.join(result['columns'])}")
                else:  # JSON file
                    st.success(f"✅ {filename}: {result['type']}, {result['keys']}")
            else:
                st.error(f"❌ {filename}: {result['error']}")
    
    def _load_available_data(self):
        """Load the data that's available"""
        # Load player data
        try:
            self.rb_data = pd.read_csv('RB_season.csv')
            st.success("🎯 RB data loaded successfully!")
        except:
            self.rb_data = pd.DataFrame()
            st.warning("⚠️ Could not load RB data")
        
        try:
            self.qb_data = pd.read_csv('QB_season.csv')
            st.success("🎯 QB data loaded successfully!")
        except:
            self.qb_data = pd.DataFrame()
            st.warning("⚠️ Could not load QB data")
        
        # Load JSON data
        try:
            with open('2025_NFL_DEFENSE.json', 'r') as f:
                self.defense_data = json.load(f)
            st.success("🛡️ Defense data loaded successfully!")
        except Exception as e:
            self.defense_data = {}
            st.warning(f"⚠️ Could not load defense data: {e}")
        
        try:
            with open('2025_NFL_OFFENSE.json', 'r') as f:
                self.offense_data = json.load(f)
            st.success("⚡ Offense data loaded successfully!")
        except Exception as e:
            self.offense_data = {}
            st.warning(f"⚠️ Could not load offense data: {e}")
    
    def get_available_rbs(self):
        """Get available running backs"""
        if not self.rb_data.empty:
            return self.rb_data['PlayerName'].tolist()
        return []
    
    def get_available_qbs(self):
        """Get available quarterbacks"""
        if not self.qb_data.empty:
            return self.qb_data['PlayerName'].tolist()
        return []
    
    def get_available_teams(self):
        """Get available teams from loaded data"""
        teams = set()
        
        # From player data
        if not self.rb_data.empty:
            teams.update(self.rb_data['Team'].dropna().unique())
        if not self.qb_data.empty:
            teams.update(self.qb_data['Team'].dropna().unique())
        
        # From defense data
        if self.defense_data:
            if isinstance(self.defense_data, list):
                for team in self.defense_data:
                    if 'Team' in team:
                        teams.add(team['Team'])
            elif isinstance(self.defense_data, dict):
                # Try common structures
                if 'teams' in self.defense_data:
                    for team in self.defense_data['teams']:
                        if 'Team' in team:
                            teams.add(team['Team'])
                else:
                    # Assume keys are team names
                    teams.update(self.defense_data.keys())
        
        return sorted(list(teams))
    
    def simple_rb_projection(self, rb_name, opponent_team):
        """Simple RB projection for testing"""
        if self.rb_data.empty:
            return "No RB data loaded"
        
        rb_row = self.rb_data[self.rb_data['PlayerName'] == rb_name]
        if rb_row.empty:
            return f"RB {rb_name} not found"
        
        rb_stats = rb_row.iloc[0]
        
        # Simple projection based on averages
        games_played = 9  # Default
        rush_yds_pg = rb_stats['RushingYDS'] / games_played
        rush_td_pg = rb_stats['RushingTD'] / games_played
        rec_yds_pg = rb_stats['ReceivingYDS'] / games_played
        rec_td_pg = rb_stats['ReceivingTD'] / games_played
        receptions_pg = rb_stats['ReceivingRec'] / games_played
        
        return {
            'RushingYards': round(rush_yds_pg, 1),
            'RushingTDs': round(rush_td_pg, 1),
            'ReceivingYards': round(rec_yds_pg, 1),
            'ReceivingTDs': round(rec_td_pg, 1),
            'Receptions': round(receptions_pg, 1)
        }

def main():
    st.set_page_config(
        page_title="NFL Player Projections - Debug",
        page_icon="🏈",
        layout="wide"
    )
    
    st.title("🏈 NFL Player Projections - Debug Mode")
    st.markdown("### Let's see what files are actually loading...")
    
    # Initialize the projector
    if 'projector' not in st.session_state:
        st.session_state.projector = SimpleNFLProjector()
    
    projector = st.session_state.projector
    
    # Only show projection interface if we have the basic data
    if not projector.rb_data.empty and not projector.qb_data.empty:
        st.header("🎯 Projection Interface")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Running Back Projections")
            available_rbs = projector.get_available_rbs()
            if available_rbs:
                rb_name = st.selectbox("Select RB", available_rbs)
                available_teams = projector.get_available_teams()
                if available_teams:
                    opponent_team = st.selectbox("Select Opponent", available_teams)
                    if st.button("Project RB Stats"):
                        projection = projector.simple_rb_projection(rb_name, opponent_team)
                        if isinstance(projection, dict):
                            st.subheader(f"Projection for {rb_name} vs {opponent_team}")
                            for stat, value in projection.items():
                                st.metric(stat, value)
                        else:
                            st.error(projection)
                else:
                    st.warning("No teams available from loaded data")
            else:
                st.warning("No RBs available from loaded data")
        
        with col2:
            st.subheader("Data Preview")
            if not projector.rb_data.empty:
                st.write("RB Data Preview:")
                st.dataframe(projector.rb_data[['PlayerName', 'Team', 'RushingYDS', 'RushingTD']].head())
            
            if projector.defense_data:
                st.write("Defense Data Structure:")
                st.json(projector.defense_data if isinstance(projector.defense_data, (dict, list)) else {"data": "available"})
    
    else:
        st.error("""
        ❌ Cannot show projections - missing required data files.
        
        **Required files in your repository:**
        - `RB_season.csv` 
        - `QB_season.csv`
        
        **Optional but helpful:**
        - `2025_NFL_DEFENSE.json`
        - `2025_NFL_OFFENSE.json`
        
        Please check that these files are in your GitHub repository and try again.
        """)

if __name__ == "__main__":
    main()
