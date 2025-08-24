"""
Main script for Nashville Tourism Analysis

Run comprehensive Airbnb analysis including walkability, neighborhood quality,
and bang-for-buck scoring with optional visualizations.
"""

import argparse
import os
import sys
import pandas as pd
import geopandas as gpd

from config import OUTPUT_FILES
from preprocessing import (
    load_data, clean_listings_data, setup_neighborhoods, 
    assign_neighborhoods_to_points, calculate_walkability,
    calculate_neighborhood_quality, calculate_bang_for_buck,
    get_neighborhood_rankings
)
from create_figures import save_figures


def save_results(results_df, neighborhood_rankings, outdir):
    """Save analysis results to CSV files."""
    print(f"Saving results to {outdir}...")
    
    os.makedirs(outdir, exist_ok=True)
    
    # Save neighborhood rankings
    rankings_path = os.path.join(outdir, OUTPUT_FILES['neighborhood_rankings'])
    neighborhood_rankings.to_csv(rankings_path, index=False)
    print(f"Saved neighborhood rankings: {rankings_path}")
    
    # Save detailed listing analysis
    analysis_path = os.path.join(outdir, OUTPUT_FILES['listing_analysis'])
    results_df.to_csv(analysis_path, index=False)
    print(f"Saved listing analysis: {analysis_path}")
    
    return rankings_path, analysis_path


def print_summary_statistics(results_df, neighborhood_rankings):
    """Print key summary statistics."""
    print("\n" + "="*80)
    print("NASHVILLE AIRBNB ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"• Total listings analyzed: {len(results_df):,}")
    print(f"• Total neighborhoods: {len(neighborhood_rankings):,}")
    print(f"• Average price: ${results_df['price'].mean():.2f}")
    print(f"• Price range: ${results_df['price'].min():.0f} - ${results_df['price'].max():.0f}")
    print(f"• Average bang-for-buck score: {results_df['bang_for_buck_score'].mean():.4f}")
    print(f"• Score range: {results_df['bang_for_buck_score'].min():.4f} to {results_df['bang_for_buck_score'].max():.4f}")
    
    print(f"\nTOP 10 BANG-FOR-BUCK NEIGHBORHOODS:")
    top_10 = neighborhood_rankings.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['neighborhood']:<25} "
              f"Score: {row['avg_bang_for_buck']:.3f} "
              f"(${row['avg_price']:.0f} avg, {row['listing_count']} listings)")
    
    print(f"\nSCORING COMPONENTS:")
    print("• Walkability: 25%")
    print("• Neighborhood Quality: 20%")
    print("• Business Diversity: 10%")
    print("• Business Density: 10%")
    print("• Popularity: 15%")
    print("• Price (inverted): 20%")
    
    if len(results_df) > 0:
        print(f"\nFEATURE CORRELATIONS WITH BANG-FOR-BUCK:")
        correlation_features = ['walkability_score', 'neighborhood_quality', 'enhanced_popularity', 'price']
        correlations = results_df[['bang_for_buck_score'] + correlation_features].corr()
        for feature in correlation_features:
            if feature in correlations.columns:
                corr = correlations.loc['bang_for_buck_score', feature]
                print(f"• {feature.replace('_', ' ').title():<25}: {corr:>6.3f}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Nashville Airbnb Tourism Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/ --outdir results/
  python main.py --input data/ --outdir results/ --save-figures
  python main.py --input data/ --outdir results/ --limit 1000
        """
    )
    
    parser.add_argument(
        '--input', 
        required=True,
        help='Input directory containing data files'
    )
    
    parser.add_argument(
        '--outdir',
        required=True, 
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save-figures',
        action='store_true',
        help='Generate and save visualization figures'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of listings for testing (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    try:
        print("Starting Nashville Tourism Analysis...")
        print("="*60)
        
        # Step 1: Load data
        listings_df, restaurants_df, businesses_df, neighborhoods_gdf = load_data(args.input)
        
        # Step 2: Clean and preprocess
        listings_df = clean_listings_data(listings_df, limit=args.limit)
        neighborhoods_gdf = setup_neighborhoods(neighborhoods_gdf)
        
        # Step 3: Assign neighborhoods to points
        listings_df = assign_neighborhoods_to_points(listings_df, neighborhoods_gdf) 
        businesses_df = assign_neighborhoods_to_points(businesses_df, neighborhoods_gdf)
        restaurants_df = assign_neighborhoods_to_points(restaurants_df, neighborhoods_gdf)
        
        # Step 4: Calculate walkability
        walkability_scores, diversity_scores = calculate_walkability(listings_df, businesses_df)
        listings_df['walkability_score'] = walkability_scores
        listings_df['business_diversity'] = diversity_scores
        
        # Step 5: Create GeoDataFrame for spatial operations
        listings_gdf = gpd.GeoDataFrame(
            listings_df,
            geometry=gpd.points_from_xy(listings_df['longitude'], listings_df['latitude']),
            crs="EPSG:4326"
        )
        
        # Step 6: Calculate neighborhood quality
        quality_scores, density_scores = calculate_neighborhood_quality(listings_gdf, businesses_df)
        listings_gdf['neighborhood_quality'] = quality_scores
        listings_gdf['business_density'] = density_scores
        
        # Step 7: Calculate bang-for-buck scores
        results_df = calculate_bang_for_buck(listings_gdf)
        
        if len(results_df) == 0:
            print("No valid listings found for analysis!")
            sys.exit(1)
        
        # Step 8: Generate neighborhood rankings
        neighborhood_rankings = get_neighborhood_rankings(results_df)
        
        # Step 9: Save results
        save_results(results_df, neighborhood_rankings, args.outdir)
        
        # Step 10: Generate visualizations if requested
        save_figures(results_df, neighborhood_rankings, neighborhoods_gdf, args.outdir, args.save_figures)
        
        # Step 11: Print summary
        print_summary_statistics(results_df, neighborhood_rankings)
        
        print(f"\n✅ Analysis complete! Results saved to: {args.outdir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()