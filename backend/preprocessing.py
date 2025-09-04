import pandas as pd

def load_and_clean_data(csv_path, start_date='2024-10-15'):
    """
    Load NBA games data and perform basic cleaning.
    Drops duplicates and missing values.
    Filters for games from start_date onwards to focus on recent team compositions.
    """
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime for filtering
    df['GAME_DATE_REAL'] = pd.to_datetime(df['GAME_DATE_REAL'])
    
    # Filter for games from start_date onwards (post-early season adjustments)
    start_date = pd.to_datetime(start_date)
    df = df[df['GAME_DATE_REAL'] >= start_date]
    print(f'Using games from {start_date.strftime("%B %d, %Y")} onwards')
    print(f'Total games after date filter: {len(df)}')
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Print missing value counts
    print('Missing values per column:')
    print(df.isnull().sum())
    
    # Drop rows with missing values
    df = df.dropna()
    
    print(f'Final dataset: {len(df)} games')
    return df 