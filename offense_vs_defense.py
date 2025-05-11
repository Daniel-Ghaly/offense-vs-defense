import pandas as pd
import statsmodels.api as sm

years = list(range(1984, 2025))
all_dfs = []

for year in years:
    # Load regular season and playoff data for the given year
    season_file = f"data/{year}_season.csv"
    playoffs_file = f"data/{year}_playoffs.csv"
    
    df_season = pd.read_csv(season_file, header=1)
    df_playoffs = pd.read_csv(playoffs_file, header=1)
    
    # Standardize team column name
    if 'Tm' in df_season.columns:
        df_season.rename(columns={'Tm': 'Team'}, inplace=True)
    if 'Tm' in df_playoffs.columns:
        df_playoffs.rename(columns={'Tm': 'Team'}, inplace=True)

    # Clean team names (remove asterisks and leading/trailing spaces)
    df_season['Team'] = df_season['Team'].str.replace('*', '', regex=False).str.strip()
    df_playoffs['Team'] = df_playoffs['Team'].str.replace('*', '', regex=False).str.strip()
    
    # Merge regular season and playoff data on "Team" values
    merged = pd.merge(df_season, df_playoffs, on='Team')
    all_dfs.append(merged)

# Combine all years into one DataFrame
final_df = pd.concat(all_dfs, ignore_index=True)

# Select only relevant columns and rename them for clarity
columns_to_keep = ['Team', 'Year_x', 'ORtg_x', 'DRtg_x', 'W_x', 'L_x', 'W_y', 'L_y']
final_df_clean = final_df[columns_to_keep].rename(columns={
    'Year_x': 'Year',
    'ORtg_x': 'OffRtg',
    'DRtg_x': 'DefRtg',
    'W_x': 'SeasonWins',
    'L_x': 'SeasonLosses',
    'W_y': 'PlayoffWins',
    'L_y': 'PlayoffLosses'
})

# Fill missing playoff win/loss values and convert to integers
final_df_clean[['SeasonWins', 'SeasonLosses', 'PlayoffWins', 'PlayoffLosses']] = (
    final_df_clean[['SeasonWins', 'SeasonLosses', 'PlayoffWins', 'PlayoffLosses']]
    .fillna(0)
    .astype(int)
)

# Identify championship teams (team with most playoff wins per year)
final_df_clean['WonTitle'] = final_df_clean.groupby('Year')['PlayoffWins'].transform(
    lambda x: (x == x.max()).astype(int)
)

# Drop rows with missing championship data
final_df_clean = final_df_clean.dropna(subset=['WonTitle'])

# -------------------------------
# Linear Regression: Predict Playoff Wins
# -------------------------------
X = final_df_clean[['OffRtg', 'DefRtg']]
X = sm.add_constant(X)  # Add constant term (intercept)
y = final_df_clean['PlayoffWins']

model = sm.OLS(y, X).fit()
print(model.summary())

# -------------------------------
# Logistic Regression: Predict Championship (1 if won title, else 0)
# -------------------------------
y = final_df_clean['WonTitle']
X = final_df_clean[['OffRtg', 'DefRtg']]
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()
print(model.summary())