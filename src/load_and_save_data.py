import pandas as pd
from pathlib import Path

def height_to_cm(height_str: str) -> int:
    feet, inches = height_str.split('-')
    return round(int(feet) * 30.48 + int(inches) * 2.54)

def parse_receiver_alignment(alignment: str) -> tuple:
    if pd.isna(alignment) or alignment == '':
        return (0, 0)
    parts = alignment.split('x')
    left = int(parts[0])
    right = int(parts[1])
    return (left, right)

def seconds_left_in_quarter(clock_str: str) -> int:
    minutes, seconds = map(int, clock_str.split(":"))
    time_left = minutes * 60 + seconds
    return time_left

def load_data(dir_path: Path) -> pd.DataFrame:
    train_path = dir_path / 'train'
    
    supplementary_file = dir_path / 'supplementary_data.csv'
    supplementary_cols_to_drop = ['week','game_date','game_time_eastern','penalty_yards','season','play_description','play_nullified_by_penalty',
                                  'pass_result','receiver_alignment','pre_penalty_yards_gained','yards_gained','game_clock','play_action']
    supplementary_df = pd.read_csv(supplementary_file)
    supplementary_df['home_team_abbr'] = supplementary_df['home_team_abbr'].astype('category').cat.codes
    supplementary_df['possession_team'] = supplementary_df['possession_team'].astype('category').cat.codes
    supplementary_df['defensive_team'] = supplementary_df['defensive_team'].astype('category').cat.codes
    supplementary_df['yardline_side'] = supplementary_df['yardline_side'].astype('category').cat.codes
    supplementary_df['visitor_team_abbr'] = supplementary_df['visitor_team_abbr'].astype('category').cat.codes
    supplementary_df['offense_formation'] = supplementary_df['offense_formation'].astype('category').cat.codes
    supplementary_df[['receivers_left', 'receivers_right']] = supplementary_df['receiver_alignment'].apply(
        lambda x: pd.Series(parse_receiver_alignment(x))
    )
    supplementary_df['game_clock'] = supplementary_df['game_clock'].apply(seconds_left_in_quarter)
    supplementary_df['route_of_targeted_receiver'] = supplementary_df['route_of_targeted_receiver'].astype('category').cat.codes
    supplementary_df['dropback_type'] = supplementary_df['dropback_type'].astype('category').cat.codes
    supplementary_df['pass_location_type'] = supplementary_df['pass_location_type'].astype('category').cat.codes
    supplementary_df['team_coverage_man_zone'] = supplementary_df['team_coverage_man_zone'].astype('category').cat.codes
    supplementary_df['team_coverage_type'] = supplementary_df['team_coverage_type'].astype('category').cat.codes
    supplementary_df.drop(columns=supplementary_cols_to_drop, inplace=True)
    
    dfs = []
    for i in range(1, 19):
        week_str = f'w{i:02d}'
        print(f"Loading week {i}...")
        df_input = pd.read_csv(train_path / f'input_2023_{week_str}.csv')
        df_output = pd.read_csv(train_path / f'output_2023_{week_str}.csv')
        
        df_input = df_input[df_input['player_to_predict'] == True]
        df_input['player_height_cm'] = df_input['player_height'].apply(height_to_cm)
        df_input['player_position'] = df_input['player_position'].astype('category').cat.codes
        df_input['player_side'] = df_input['player_side'].astype('category').cat.codes
        df_input['player_role'] = df_input['player_role'].astype('category').cat.codes
        df_input.drop(columns=['player_height', 'player_weight','player_birth_date','player_name','num_frames_output','player_to_predict','play_direction'], inplace=True)
        
        df_output = df_output.rename(columns={'x': 'target_x', 'y': 'target_y'})
        
        df_merged = pd.merge(df_input, df_output, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')
        df_final = pd.merge(df_merged, supplementary_df, on=['game_id', 'play_id'], how='left')
        dfs.append(df_final)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    frame_type_cols = [col for col in combined_df.columns if col.startswith('frame_type')]
    if frame_type_cols:
        combined_df.drop(columns=frame_type_cols, inplace=True)
    
    print(f"Total rows loaded: {len(combined_df):,}")
    return combined_df

def save_data(df: pd.DataFrame, file_path: Path) -> None:
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    dir = Path('./data')
    data = load_data(dir)
    save_data(data, Path('./data/combined_data.csv'))