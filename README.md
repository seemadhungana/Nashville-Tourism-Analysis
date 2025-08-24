# Nashville Tourism Analysis

A comprehensive analysis of Nashville's Airbnb market focusing on walkability, neighborhood quality, and bang-for-buck scores. This project combines Airbnb listings with local business data to identify the best value neighborhoods for travelers.

## Features

- **Walkability Analysis**: Calculates proximity-based scores using nearby restaurants and businesses
- **Neighborhood Quality**: Evaluates areas based on local business ratings and review density
- **Bang-for-Buck Scoring**: Combines walkability, quality, popularity, and price into actionable insights
- **Interactive Visualizations**: Maps and statistical plots for comprehensive analysis

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data** (expected files in input directory):
   - `listings.csv` - Airbnb listings data
   - `nashville_businesses.csv` - Local business data  
   - `nashville_restaurants.csv` - Restaurant data
   - `neighbourhoods.geojson` - Neighborhood boundaries

3. **Run the analysis**:
   ```bash
   python main.py --input data/ --outdir results/ --save-figures
   ```

## Example Commands

```bash
# Basic analysis with output
python main.py --input ./data --outdir ./output

# Save all visualizations
python main.py --input ./data --outdir ./results --save-figures

# Limit listings for testing
python main.py --input ./data --outdir ./test_output --limit 1000
```

## Configuration

Edit `config.py` to adjust analysis parameters:
- Walkability distance thresholds
- Neighborhood quality radius
- Bang-for-buck scoring weights
- Minimum review requirements

## Workflow Overview

1. **Data Loading**: Loads Airbnb listings, business data, and neighborhood boundaries
2. **Data Cleaning**: Handles missing values, outliers, and standardizes formats
3. **Spatial Matching**: Maps businesses and restaurants to neighborhoods using coordinates
4. **Feature Engineering**: Calculates walkability scores, neighborhood quality metrics
5. **Bang-for-Buck Analysis**: Combines multiple factors into comprehensive neighborhood rankings
6. **Visualization**: Generates interactive maps and statistical plots
7. **Output**: Saves results as CSV files and optional visualizations

## Output Files

- `neighborhood_rankings.csv` - Top neighborhoods by bang-for-buck score
- `listing_analysis.csv` - Individual listing scores and metrics
- `enhanced_airbnb_map.html` - Interactive neighborhood map (if --save-figures used)
- Various PNG plots showing price distributions and correlations (if --save-figures used)