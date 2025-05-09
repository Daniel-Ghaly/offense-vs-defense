import pandas as pd
import statsmodels.api as sm

years = list(range(1984, 2025))
all_dfs = []

for year in years:
    # Load season and playoffs CSVs
    season_file = f"data/{year}_season.csv"
    playoffs_file = f"data/{year}_playoffs.csv"
    
    df_season = pd.read_csv(season_file, header=1)
    df_playoffs = pd.read_csv(playoffs_file, header=1)
    
    # Rename Tm → Team if needed
    if 'Tm' in df_season.columns:
        df_season.rename(columns={'Tm': 'Team'}, inplace=True)
    if 'Tm' in df_playoffs.columns:
        df_playoffs.rename(columns={'Tm': 'Team'}, inplace=True)

    df_season['Team'] = df_season['Team'].str.replace('*', '', regex=False).str.strip()
    df_playoffs['Team'] = df_playoffs['Team'].str.replace('*', '', regex=False).str.strip()
    
    # Merge on Team
    merged = pd.merge(df_season, df_playoffs, on='Team')
    
    all_dfs.append(merged)

# Concatenate all years
final_df = pd.concat(all_dfs, ignore_index=True)
columns_to_keep = ['Team', 'Year_x', 'ORtg_x', 'DRtg_x', 'W_x', 'L_x', 'W_y',  'L_y']
final_df_clean = final_df[columns_to_keep]
final_df_clean = final_df_clean.rename(columns={
    'Year_x': 'Year',
    'ORtg_x': 'OffRtg',
    'DRtg_x': 'DefRtg',
    'W_x': 'SeasonWins',
    'L_x': 'SeasonLosses',
    'W_y': 'PlayoffWins',
    'L_y': 'PlayoffLosses'
})
final_df_clean[['SeasonWins', 'SeasonLosses', 'PlayoffWins', 'PlayoffLosses']] = (
    final_df_clean[['SeasonWins', 'SeasonLosses', 'PlayoffWins', 'PlayoffLosses']]
    .fillna(0)
    .astype(int)
)
final_df_clean['WonTitle'] = final_df_clean.groupby('Year')['PlayoffWins'].transform(
    lambda x: (x == x.max()).astype(int)
)
final_df_clean = final_df_clean.dropna(subset=['WonTitle'])


print(final_df_clean[['Team', 'Year', 'PlayoffWins', 'WonTitle']])

print(final_df_clean['WonTitle'].unique())


# ✅ final_df now contains all seasons, merged, clean
print(final_df_clean.shape)
print(final_df_clean.head())
print(final_df_clean.columns.tolist())


# Define independent variables
X = final_df_clean[['OffRtg', 'DefRtg']]
X = sm.add_constant(X)  # adds intercept

# Define dependent variable
y = final_df_clean['PlayoffWins']

# Fit model
model = sm.OLS(y, X).fit()

# View summary
print(model.summary())


# Binary outcome (1 if won title, else 0)
y = final_df_clean['WonTitle']  

X = final_df_clean[['OffRtg', 'DefRtg']]
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()

print(model.summary())