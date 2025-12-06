import pandas as pd
import numpy as np
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

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-based ball trajectory features to dataset."""
    g = 9.8  # gravity (m/s^2)
    
    def calculate_throw_features(group):
        # Handle both string and categorical player_role
        qb_data = group[group['player_role'].astype(str) == 'Passer']
        if len(qb_data) == 0:
            return None
        
        # QB throw position (last frame)
        qb_last = qb_data.iloc[-1]
        x_throw, y_throw = qb_last['x'], qb_last['y']
        
        # Ball landing position
        x_land = group.iloc[0]['ball_land_x']
        y_land = group.iloc[0]['ball_land_y']
        
        # Horizontal distance
        horiz_dist = np.sqrt((x_land - x_throw)**2 + (y_land - y_throw)**2)
        
        # Flight time from output frames
        output_frames = group[group['target_x'].notna()]
        if len(output_frames) == 0:
            return None
            
        time_flight = output_frames['frame_id'].max() / 10.0
        
        if time_flight <= 0 or horiz_dist == 0:
            return None
        
        # Physics calculations
        v_xy = horiz_dist / time_flight
        v_z = (g * time_flight) / 2.0  # symmetric parabola
        launch_velocity = np.sqrt(v_xy**2 + v_z**2)
        launch_angle_deg = np.degrees(np.arctan2(v_z, v_xy))
        
        return {
            'v_xy': v_xy,
            'v_z': v_z,
            'launch_velocity': launch_velocity,
            'launch_angle_deg': launch_angle_deg,
            'time_to_target_s': time_flight
        }
    
    # Calculate and broadcast features
    play_features = {}
    for (game_id, play_id), group in df.groupby(['game_id', 'play_id']):
        features = calculate_throw_features(group)
        if features is not None:
            play_features[(game_id, play_id)] = features
    
    # Initialize columns
    for col in ['v_xy', 'v_z', 'launch_velocity', 'launch_angle_deg', 'time_to_target_s']:
        df[col] = np.nan
    
    # Broadcast to all rows in each play
    for (game_id, play_id), features in play_features.items():
        mask = (df['game_id'] == game_id) & (df['play_id'] == play_id)
        for col, value in features.items():
            df.loc[mask, col] = value
    
    return df

def load_data(dir_path: Path) -> pd.DataFrame:
    """Load and process NFL tracking data with physics features."""
    train_path = dir_path / 'train'
    
    # Load and process supplementary data
    supplementary_df = pd.read_csv(dir_path / 'supplementary_data.csv', low_memory=False)
    
    # Keep categorical columns as categories (preserve original values)
    cat_cols = ['home_team_abbr', 'possession_team', 'defensive_team', 'yardline_side', 
                'visitor_team_abbr', 'offense_formation', 'route_of_targeted_receiver', 
                'dropback_type', 'pass_location_type', 'team_coverage_man_zone', 'team_coverage_type']
    for col in cat_cols:
        supplementary_df[col] = supplementary_df[col].astype('category')
    
    # Process specific columns
    supplementary_df[['receivers_left', 'receivers_right']] = supplementary_df['receiver_alignment'].apply(
        lambda x: pd.Series(parse_receiver_alignment(x))
    )
    supplementary_df['game_clock'] = supplementary_df['game_clock'].apply(seconds_left_in_quarter)
    
    # Drop unnecessary columns
    drop_cols = ['week', 'game_date', 'game_time_eastern', 'penalty_yards', 'season', 
                 'play_description', 'play_nullified_by_penalty', 'pass_result', 'receiver_alignment',
                 'pre_penalty_yards_gained', 'yards_gained', 'game_clock', 'play_action']
    supplementary_df.drop(columns=drop_cols, inplace=True)
    
    # Process weekly data
    dfs = []
    for i in range(1, 19):
        print(f"Loading week {i}...")
        
        # Load input and output data
        df_input = pd.read_csv(train_path / f'input_2023_w{i:02d}.csv')
        df_output = pd.read_csv(train_path / f'output_2023_w{i:02d}.csv')
        
        # Process input data (keep all players for physics calculations)
        df_input['player_height_cm'] = df_input['player_height'].apply(height_to_cm)
        # Keep categorical columns as categories (preserve original values)
        df_input['player_position'] = df_input['player_position'].astype('category')
        df_input['player_side'] = df_input['player_side'].astype('category')
        # Keep player_role as category for feature engineering
        df_input['player_role'] = df_input['player_role'].astype('category')
        df_input.drop(columns=['player_height', 'player_weight', 'player_birth_date', 
                              'player_name', 'num_frames_output', 'play_direction'], inplace=True)
        
        # Merge data
        df_output = df_output.rename(columns={'x': 'target_x', 'y': 'target_y'})
        df_merged = pd.merge(df_input, df_output, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')
        df_final = pd.merge(df_merged, supplementary_df, on=['game_id', 'play_id'], how='left')

        # Add physics features (keep player_role as category for now)
        df_final = add_features(df_final)
        
        # Filter to prediction targets only
        df_final = df_final[df_final['player_to_predict'] == True]
        df_final.drop(columns=['player_to_predict'], inplace=True)
        
        dfs.append(df_final)
    
    # Combine all weeks
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove frame_type columns if they exist
    frame_type_cols = [col for col in combined_df.columns if col.startswith('frame_type')]
    if frame_type_cols:
        combined_df.drop(columns=frame_type_cols, inplace=True)
    
    print(f"Total rows loaded: {len(combined_df):,}")
    return combined_df

def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame to CSV file."""
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    data_dir = Path('./data')
    data = load_data(data_dir)
    save_data(data, data_dir / 'combined_data.csv')