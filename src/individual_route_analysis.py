"""
NFL Route Analysis with Physics-Based Modeling
Generates individual visualizations for each route type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class RouteAnalyzer:
    """Analyzes NFL route patterns and generates visualizations."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.play_data = None
        self.model = None
        self.model_metrics = {}
        
        self.route_descriptions = {
            'SLANT': 'Quick diagonal cut inside (3-5 yards)',
            'OUT': 'Break to the sideline (8-12 yards)', 
            'GO': 'Deep straight vertical route (15+ yards)',
            'HITCH': 'Stop and comeback route (6-10 yards)',
            'CROSS': 'Crossing pattern over the middle',
            'IN': 'Break to the inside (10-15 yards)',
            'POST': 'Deep route to the goal posts',
            'FLAT': 'Quick shallow route to the sideline',
            'COMEBACK': 'Deep comeback route (12+ yards)',
            'QUICK_OUT': 'Fast break to sideline (3-5 yards)'
        }
        
    def load_and_prepare_data(self):
        """Load and prepare the combined dataset."""
        print("Loading combined data...")
        self.play_data = pd.read_csv(self.data_path)
        
        # Add success metric (positive EPA as proxy for good plays)
        self.play_data['success'] = (
            self.play_data['expected_points_added'] > 0
        ).astype(int)
        
        print(f"Loaded {len(self.play_data):,} plays")
        return self.play_data
    
    def train_physics_model(self):
        """Train model to predict route success."""
        print("\nTraining physics-based success prediction model...")
        
        # Features for model
        features = ['launch_velocity', 'launch_angle_deg', 'v_xy', 'pass_length']
        X = self.play_data[features].fillna(self.play_data[features].median()) # type: ignore
        y = self.play_data['success'] # type: ignore
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        baseline = y_test.mean()
        
        print(f"Model Accuracy: {accuracy:.3f} ({accuracy-baseline:+.1%} vs baseline)")
        print(f"Baseline: {baseline:.3f}")
        performance = "Good" if accuracy > 0.58 else "Moderate" if accuracy > 0.55 else "Poor"
        print(f"Model Performance: {performance}")
        
        # Store model metrics for summary
        self.model_metrics = {
            'accuracy': accuracy,
            'baseline': baseline,
            'improvement': accuracy - baseline,
            'features': features
        }
        
    def predict_success(self, physics_params):
        """Predict success probability from physics parameters."""
        if self.model is None:
            return 0.5
        return self.model.predict_proba(np.array([physics_params]))[0][1]
    
    def draw_nfl_field(self, ax):
        """Draw professional NFL field."""
        # Field dimensions (120 yards x 53.3 yards)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        
        # Field color
        ax.add_patch(patches.Rectangle((0, 0), 120, 53.3, facecolor='#96b78c', alpha=0.8))
        
        # Yard lines
        for yard in range(10, 110, 10):
            ax.plot([yard, yard], [0, 53.3], 'white', linewidth=2, alpha=0.8)
            
        # Goal lines
        ax.plot([10, 10], [0, 53.3], 'white', linewidth=4)
        ax.plot([110, 110], [0, 53.3], 'white', linewidth=4)
        
        # Sidelines
        ax.plot([0, 120], [0, 0], 'white', linewidth=4)
        ax.plot([0, 120], [53.3, 53.3], 'white', linewidth=4)
        ax.plot([0, 0], [0, 53.3], 'white', linewidth=4)
        ax.plot([120, 120], [0, 53.3], 'white', linewidth=4)
        
        # End zones
        ax.add_patch(patches.Rectangle((0, 0), 10, 53.3, facecolor='#4CAF50', alpha=0.3))
        ax.add_patch(patches.Rectangle((110, 0), 10, 53.3, facecolor='#4CAF50', alpha=0.3))
        
        # Goal posts
        ax.plot([0, 0], [22, 31.3], 'gold', linewidth=8)
        ax.plot([120, 120], [22, 31.3], 'gold', linewidth=8)
        
        # 50-yard line
        ax.plot([60, 60], [0, 53.3], 'white', linewidth=4)
        ax.text(60, 26.5, '50', fontsize=24, color='white', weight='bold', 
                ha='center', va='center', bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def create_individual_route_images(self, output_dir: Path):
        """Create separate image for each route type."""
        output_dir.mkdir(exist_ok=True)
        
        print("\nCreating route visualizations...")
        
        # Get routes with sufficient sample size
        route_counts = self.play_data['route_of_targeted_receiver'].value_counts() # type: ignore
        top_routes = route_counts[route_counts >= 100].head(8).index
        
        # Create individual images
        for route in top_routes:
            self._create_single_route_image(route, output_dir)
            
        # Create summary comparison
        self._create_route_summary(output_dir)
        
        print(f"\nGenerated {len(top_routes)} route visualizations")
        
    def _create_single_route_image(self, route: str, output_dir: Path):
        """Create a single professional route visualization."""
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Draw field
        self.draw_nfl_field(ax)
        
        # Get route data
        route_plays = self.play_data[self.play_data['route_of_targeted_receiver'] == route] # type: ignore
        
        # Calculate comprehensive statistics
        stats = {
            'avg_velocity': route_plays['launch_velocity'].mean(),
            'avg_angle': route_plays['launch_angle_deg'].mean(),
            'avg_length': route_plays['pass_length'].mean(),
            'avg_flight_time': route_plays['time_to_target_s'].mean(),
            'success_rate': route_plays['success'].mean(),
            'avg_epa': route_plays['expected_points_added'].mean(),
            'num_plays': len(route_plays),
            'velocity_25': route_plays['launch_velocity'].quantile(0.25),
            'velocity_75': route_plays['launch_velocity'].quantile(0.75),
            'angle_25': route_plays['launch_angle_deg'].quantile(0.25),
            'angle_75': route_plays['launch_angle_deg'].quantile(0.75)
        }
        
        # Predict success
        physics_params = [stats['avg_velocity'], stats['avg_angle'], 
                         route_plays['v_xy'].mean(), stats['avg_length']]
        predicted_success = self.predict_success(physics_params)
        
        # Draw route pattern
        self._draw_route_pattern(ax, route, stats)
        
        # Add comprehensive info panel
        self._add_info_panel(ax, route, stats, predicted_success)
        
        # Add success indicator
        self._add_success_indicator(ax, stats['success_rate'])
        
        # Set title
        title = f"{route} Route Analysis - {self.route_descriptions.get(route, 'NFL Route Pattern')}"
        ax.set_title(title, fontsize=18, weight='bold', color='black', pad=20)
        
        # Save
        plt.tight_layout()
        filename = f'route_{route.lower()}_analysis.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated: {filename}")
        
    def _draw_route_pattern(self, ax, route: str, stats: dict):
        """Draw route-specific pattern with professional styling."""
        qb_x, qb_y = 60, 26.5
        receiver_color = '#FFD700'
        route_color = '#FF4500'
        qb_color = '#DC143C'
        
        # QB position
        ax.scatter(qb_x, qb_y, c=qb_color, s=500, marker='o', 
                  edgecolors='white', linewidth=4, zorder=5)
        ax.text(qb_x, qb_y, 'QB', ha='center', va='center', 
               fontsize=12, color='white', weight='bold', zorder=6)
        
        # Draw route patterns
        if route == 'SLANT':
            rx, ry = qb_x + 8, qb_y - 6
            ax.plot([qb_x, qb_x + 3], [qb_y, qb_y], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([qb_x + 3, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            
        elif route == 'OUT': 
            rx, ry = qb_x + 12, qb_y + 12
            ax.plot([qb_x, qb_x + 10], [qb_y, qb_y], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([qb_x + 10, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
                  
        elif route == 'GO':
            rx, ry = qb_x + 25, qb_y
            ax.plot([qb_x, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
                  
        elif route == 'HITCH':
            # Hitch: run out then comeback
            rx_initial, ry_initial = qb_x + 10, qb_y
            rx_final, ry_final = qb_x + 6, qb_y
            # Initial route out
            ax.plot([qb_x, rx_initial], [qb_y, ry_initial], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            # Comeback portion
            ax.plot([rx_initial, rx_final], [ry_initial, ry_final], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            # Add comeback arrow to show the stop and return
            ax.annotate('COMEBACK', xy=(rx_final, ry_final), xytext=(rx_initial, ry_initial),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=5, alpha=0.8),
                       fontsize=10, color='blue', weight='bold', ha='center')
            rx, ry = rx_final, ry_final
                      
        elif route == 'CROSS':
            rx, ry = qb_x + 12, qb_y - 10  
            ax.plot([qb_x, qb_x + 8], [qb_y, qb_y], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([qb_x + 8, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
                      
        elif route == 'IN':
            rx, ry = qb_x + 12, qb_y - 8
            ax.plot([qb_x, qb_x + 10], [qb_y, qb_y], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([qb_x + 10, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
                   
        elif route == 'POST':
            rx, ry = qb_x + 18, qb_y - 6
            ax.plot([qb_x, qb_x + 12], [qb_y, qb_y], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([qb_x + 12, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
                   
        elif route == 'FLAT':
            rx, ry = qb_x + 5, qb_y + 10
            ax.plot([qb_x, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
        else:
            # Generic route
            distance = min(stats['avg_length'] * 0.8, 20)
            rx, ry = qb_x + distance, qb_y
            ax.plot([qb_x, rx], [qb_y, ry], route_color, 
                   linewidth=8, alpha=0.9, solid_capstyle='round')
        
        # Receiver position
        ax.scatter(rx, ry, c=receiver_color, s=500, marker='o',
                  edgecolors='black', linewidth=4, zorder=5)
        ax.text(rx, ry, 'WR', ha='center', va='center', 
               fontsize=10, color='black', weight='bold', zorder=6)
    
    def _add_info_panel(self, ax, route: str, stats: dict, predicted_success: float):
        """Add comprehensive info panel with professional formatting."""
        success_color = ('#228B22' if stats['success_rate'] > 0.55 else 
                        '#FF6347' if stats['success_rate'] < 0.45 else '#4682B4')
        
        info_panel = f"""{route} ROUTE ANALYSIS

PERFORMANCE METRICS:
  Success Rate: {stats['success_rate']:.1%} (Positive EPA)
  Average EPA: {stats['avg_epa']:+.2f} points
  Model Prediction: {predicted_success:.1%}
  Sample Size: {stats['num_plays']:,} plays

PHYSICS PROFILE:
  Launch Velocity: {stats['avg_velocity']:.1f} m/s
  Velocity Range: {stats['velocity_25']:.1f}-{stats['velocity_75']:.1f} m/s
  Launch Angle: {stats['avg_angle']:.1f}°
  Angle Range: {stats['angle_25']:.1f}°-{stats['angle_75']:.1f}°
  Pass Length: {stats['avg_length']:.1f} yards
  Flight Time: {stats['avg_flight_time']:.2f} seconds

ROUTE DESCRIPTION:
{self.route_descriptions.get(route, 'NFL route pattern')}"""
        
        ax.text(5, 48, info_panel, fontsize=11, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=1.2', facecolor='white', 
                        edgecolor=success_color, linewidth=3, alpha=0.95),
               zorder=10)
    
    def _add_success_indicator(self, ax, success_rate: float):
        """Add visual success rate indicator."""
        if success_rate > 0.55:
            indicator = "HIGH SUCCESS"
            color = '#228B22'
        elif success_rate < 0.45:
            indicator = "LOW SUCCESS"  
            color = '#FF6347'
        else:
            indicator = "MEDIUM SUCCESS"
            color = '#4682B4'
            
        ax.text(115, 48, indicator, fontsize=16, va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.8', facecolor=color, 
                        edgecolor='white', linewidth=3, alpha=0.9),
               color='white', weight='bold', zorder=10)
    
    def _create_route_summary(self, output_dir: Path):
        """Create summary comparison of all routes."""
        route_stats = self.play_data.groupby('route_of_targeted_receiver').agg({ # type: ignore
            'success': 'mean',
            'expected_points_added': 'mean', 
            'launch_velocity': 'mean',
            'launch_angle_deg': 'mean',
            'game_id': 'count'
        }).round(3)
        
        route_stats = route_stats[route_stats['game_id'] >= 100].sort_values('success', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create horizontal bar chart
        colors = ['#228B22' if x > 0.55 else '#FF6347' if x < 0.45 else '#4682B4' 
                 for x in route_stats['success']]
        
        bars = ax.barh(range(len(route_stats)), route_stats['success'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(route_stats)))
        ax.set_yticklabels(route_stats.index, fontsize=12)
        ax.set_xlabel('Success Rate (Positive EPA)', fontsize=14, weight='bold')
        ax.set_title('NFL Route Success Rate Comparison', fontsize=16, weight='bold', pad=20)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, route_stats['success'])):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.1%}', ha='left', va='center', fontweight='bold')
                   
        ax.set_xlim(0, max(route_stats['success']) * 1.15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'route_summary_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print("Generated: route_summary_comparison.png")
    
    def run_analysis(self, output_dir: str = "results"):
        """Run complete individual route analysis."""
        output_path = Path(output_dir)
        
        print("NFL BIG DATA BOWL 2026 - ROUTE ANALYSIS")
        print("="*50)
        
        # Load data
        self.load_and_prepare_data()
        
        # Train model
        self.train_physics_model()
        
        # Create visualizations
        self.create_individual_route_images(output_path)
        
        # Generate comprehensive summary
        self._print_analysis_summary(output_path)
        
    def _print_analysis_summary(self, output_path: Path):
        """Print comprehensive analysis summary."""
        print("\n" + "="*80)
        print("NFL BIG DATA BOWL 2026 - ANALYSIS SUMMARY")
        print("="*80)
        
        print("DATASET OVERVIEW:")
        print(f"  Total Plays Analyzed: {len(self.play_data):,}") # type: ignore
        print(f"  Route Types: {self.play_data['route_of_targeted_receiver'].nunique()}") # type: ignore
        print(f"  Success Rate (Positive EPA): {self.play_data['success'].mean():.1%}") # type: ignore
        print(f"  Average EPA: {self.play_data['expected_points_added'].mean():+.3f}") # type: ignore
        
        print("\nPHYSICS-BASED MODEL:")
        print(f"  Algorithm: Logistic Regression")
        print(f"  Features: {', '.join(self.model_metrics['features'])}")
        print(f"  Accuracy: {self.model_metrics['accuracy']:.1%}")
        print(f"  Improvement over Baseline: {self.model_metrics['improvement']:+.1%}")
        
        # Route Insights
        route_summary = self.play_data.groupby('route_of_targeted_receiver').agg({ # type: ignore
            'success': 'mean',
            'expected_points_added': 'mean',
            'launch_velocity': 'mean',
            'pass_length': 'mean',
            'game_id': 'count'
        }).round(3)
        route_summary = route_summary[route_summary['game_id'] >= 100].sort_values('success', ascending=False)
        
        print("\nTOP PERFORMING ROUTES:")
        for i, (route, stats) in enumerate(route_summary.head(3).iterrows(), 1):
            print(f"  {i}. {route}: {stats['success']:.1%} success, {stats['expected_points_added']:+.2f} EPA")
        
        print("\nFILES GENERATED:")
        print(f"  Output Directory: {output_path}")
        print(f"  Individual Route Images: 8 files") 
        print(f"  Summary Comparison: 1 file")
        print(f"  Total Visualizations: 9 images")
        
        print("\nANALYSIS COMPLETE")
        print("="*80)


def main():
    """Main execution."""
    analyzer = RouteAnalyzer('data/combined_data.csv')
    analyzer.run_analysis()


if __name__ == "__main__":
    main()