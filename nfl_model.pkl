# train_model.py
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

print("🚀 Starting NFL Model Training...")

# Load the data
print("📊 Loading data...")
with open('spreadspoke_scores.json', 'r') as f:
    games = pd.DataFrame(json.load(f))

print(f"Loaded {len(games)} historical games")

# Basic data cleaning
print("🧹 Cleaning data...")
games = games[games['score_home'].notna() & games['score_away'].notna()]
games['schedule_date'] = pd.to_datetime(games['schedule_date'])
games = games[games['schedule_date'].dt.year >= 2020]  # Last 5 years

# Create target variable (home team wins = 1, loses = 0)
games['home_win'] = (games['score_home'] > games['score_away']).astype(int)

print(f"Using {len(games)} games after cleaning")

# Calculate simple team stats (win percentage, avg points)
print("📈 Calculating team statistics...")

# Get all team names
all_teams = list(set(games['team_home'].unique()) | set(games['team_away'].unique()))

# Calculate rolling stats for each team
team_stats = {}
for team in all_teams:
    team_games = games[(games['team_home'] == team) | (games['team_away'] == team)].copy()
    team_games = team_games.sort_values('schedule_date')
    
    # Mark if team was home and if they won
    team_games['is_home'] = (team_games['team_home'] == team).astype(int)
    team_games['team_score'] = np.where(team_games['team_home'] == team, 
                                       team_games['score_home'], 
                                       team_games['score_away'])
    team_games['opponent_score'] = np.where(team_games['team_home'] == team, 
                                          team_games['score_away'], 
                                          team_games['score_home'])
    team_games['win'] = (team_games['team_score'] > team_games['opponent_score']).astype(int)
    
    # Calculate rolling averages (last 8 games)
    team_games['win_pct'] = team_games['win'].rolling(8, min_periods=1).mean()
    team_games['points_for_avg'] = team_games['team_score'].rolling(8, min_periods=1).mean()
    team_games['points_against_avg'] = team_games['opponent_score'].rolling(8, min_periods=1).mean()
    
    # Store by date for easy lookup
    for _, row in team_games.iterrows():
        date = row['schedule_date']
        if team not in team_stats:
            team_stats[team] = {}
        team_stats[team][date] = {
            'win_pct': row['win_pct'],
            'points_for_avg': row['points_for_avg'],
            'points_against_avg': row['points_against_avg']
        }

# Create features for model training
print("⚙️ Creating features...")
features = []
targets = []

for _, game in games.iterrows():
    home_team = game['team_home']
    away_team = game['team_away']
    game_date = game['schedule_date']
    
    # Find most recent stats before this game
    home_stats = None
    away_stats = None
    
    # Get home team stats (most recent before this game)
    if home_team in team_stats:
        previous_dates = [d for d in team_stats[home_team].keys() if d < game_date]
        if previous_dates:
            latest_date = max(previous_dates)
            home_stats = team_stats[home_team][latest_date]
    
    # Get away team stats (most recent before this game)
    if away_team in team_stats:
        previous_dates = [d for d in team_stats[away_team].keys() if d < game_date]
        if previous_dates:
            latest_date = max(previous_dates)
            away_stats = team_stats[away_team][latest_date]
    
    # Only include games where we have stats for both teams
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

# Convert to numpy arrays
X = np.array(features)
y = np.array(targets)

print(f"✅ Created {len(X)} training examples")

# Handle any NaN values
X = np.nan_to_num(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} games")
print(f"Test set: {len(X_test)} games")

# Train model
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Feature importance
feature_names = [
    'home_win_pct', 'home_offense', 'home_defense',
    'away_win_pct', 'away_offense', 'away_defense', 
    'home_field'
]

print("\n📊 Feature Importance:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.3f}")

# Save the model
print("💾 Saving model...")
with open('nfl_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'nfl_model.pkl'")

# Make some predictions on the test set
print("\n🔮 Sample Predictions:")
for i in range(min(5, len(X_test))):
    actual = y_test[i]
    prediction = model.predict_proba([X_test[i]])[0][1]  # Probability of home win
    home_win_prob = prediction
    away_win_prob = 1 - prediction
    
    print(f"Game {i+1}: Home win probability: {home_win_prob:.1%} | Actual: {'Home Win' if actual == 1 else 'Away Win'}")

print("\n🎉 Training complete! Ready to make predictions.")
