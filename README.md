# CSE 482 Final Project: NFL Route Analysis

Physics-based modeling of NFL route patterns using big data analytics and machine learning.

## Data Setup

Data was downloaded from the NFL Big Data Bowl 2026 competition:

```bash
kaggle competitions download -c nfl-big-data-bowl-2026-analytics
```

The downloaded file was unzipped and renamed as `data/` directory.

## Usage

```bash
python src/individual_route_analysis.py
```

## Output

- 9 high-quality route visualizations in `results/`
- Physics-based logistic regression model (58.8% accuracy)
- Complete analysis documented in `REPORT.md`

## Project Structure

- `src/individual_route_analysis.py` - Main analysis script
- `data/combined_data.csv` - Processed NFL tracking data  
- `results/` - Generated visualizations
- `REPORT.md` - Complete project documentation
