import pandas as pd

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

# ✅ final_df now contains all seasons, merged, clean
print(final_df.shape)
print(final_df.head())
print(final_df.columns.tolist())
