"""
Data preprocessing functions for Nashville Tourism Analysis
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.errors import ShapelyDeprecationWarning
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import warnings
import os

from config import *

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def load_data(data_dir):
    """Load all required datasets from directory."""
    print("Loading datasets...")
    
    # Construct file paths
    files = {
        'listings': os.path.join(data_dir, DATA_FILES['listings']),
        'businesses': os.path.join(data_dir, DATA_FILES['businesses']),
        'restaurants': os.path.join(data_dir, DATA_FILES['restaurants']),
        'neighborhoods': os.path.join(data_dir, DATA_FILES['neighborhoods'])
    }
    
    # Check if files exist
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Load datasets
    listings_df = pd.read_csv(files['listings'])
    businesses_df = pd.read_csv(files['businesses'])
    restaurants_df = pd.read_csv(files['restaurants'])
    neighborhoods_gdf = gpd.read_file(files['neighborhoods'])
    
    print(f"Loaded {len(listings_df)} listings, {len(businesses_df)} businesses, "
          f"{len(restaurants_df)} restaurants, {len(neighborhoods_gdf)} neighborhoods")
    
    return listings_df, restaurants_df, businesses_df, neighborhoods_gdf


def clean_listings_data(listings_df, limit=None):
    """Clean and preprocess Airbnb listings data."""
    print("Cleaning listings data...")
    
    # Apply limit if specified (for testing)
    if limit:
        listings_df = listings_df.head(limit)
        print(f"Limited to {limit} listings for testing")
    
    # Fill missing values
    listings_df['reviews_per_month'] = listings_df['reviews_per_month'].fillna(0)
    listings_df['neighbourhood_group'] = listings_df['neighbourhood_group'].fillna('Unknown')
    listings_df['host_name'] = listings_df['host_name'].fillna('Unknown')
    listings_df['last_review'] = listings_df['last_review'].fillna('No reviews')
    
    # Drop unnecessary columns
    columns_to_drop = ['license', 'neighbourhood_group']
    listings_df = listings_df.drop(columns=[col for col in columns_to_drop if col in listings_df.columns])
    
    # Handle price outliers
    price_upper_limit = listings_df['price'].quantile(CLEANING_CONFIG['price_percentile_cutoff'])
    listings_df = listings_df[listings_df['price'] <= price_upper_limit]
    
    # Convert price to numeric and drop invalid entries
    listings_df['price'] = pd.to_numeric(listings_df['price'], errors='coerce')
    listings_df = listings_df.dropna(subset=['latitude', 'longitude', 'price'])
    
    # Clean text fields
    listings_df['neighbourhood'] = listings_df['neighbourhood'].str.strip().str.title()
    listings_df['room_type'] = listings_df['room_type'].str.strip().str.lower()
    
    print(f"After cleaning: {len(listings_df)} listings remaining")
    return listings_df


def setup_neighborhoods(neighborhoods_gdf):
    """Setup neighborhood boundaries and centroids."""
    print("Setting up neighborhoods...")
    
    # Re-project for accurate centroids
    neighborhoods_gdf = neighborhoods_gdf.to_crs(epsg=3857)
    neighborhoods_gdf['centroid'] = neighborhoods_gdf.geometry.centroid
    neighborhoods_gdf = neighborhoods_gdf.to_crs(epsg=4326)
    
    # Add neighborhood names
    if 'neighbourhood_name' not in neighborhoods_gdf.columns:
        neighborhoods_gdf['neighbourhood_name'] = neighborhoods_gdf['neighbourhood'].map(DISTRICT_LOCATIONS)
        neighborhoods_gdf['neighbourhood_name'] = neighborhoods_gdf['neighbourhood_name'].fillna(neighborhoods_gdf['neighbourhood'])
    
    return neighborhoods_gdf


def assign_neighborhoods_to_points(points_df, neighborhoods_gdf):
    """Assign neighborhood names to points using spatial join."""
    print(f"Assigning neighborhoods to {len(points_df)} points...")
    
    # Create points GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        points_df.copy(),
        geometry=gpd.points_from_xy(points_df['longitude'], points_df['latitude']),
        crs="EPSG:4326"
    )
    
    # Ensure neighborhood polygons are valid
    neighborhoods_clean = neighborhoods_gdf.to_crs("EPSG:4326").copy()
    neighborhoods_clean["geometry"] = neighborhoods_clean.buffer(0)
    
    name_col = "neighbourhood_name" if "neighbourhood_name" in neighborhoods_clean.columns else "neighbourhood"
    
    # Spatial join
    tagged = gpd.sjoin(
        points_gdf,
        neighborhoods_clean[[name_col, "geometry"]],
        how="left",
        predicate="within"
    )
    
    # Handle duplicates and write back
    names = (tagged[name_col]
             .groupby(level=0)
             .first()
             .reindex(points_df.index))
    
    points_df = points_df.copy()
    points_df["neighbourhood_name"] = names
    
    # Nearest neighbor fallback for points outside polygons
    missing = points_df["neighbourhood_name"].isna()
    if missing.any():
        print(f"Using nearest neighbor for {missing.sum()} points outside polygons")
        p_m = points_gdf.to_crs(3857)
        n_m = neighborhoods_clean.to_crs(3857)
        nearest = gpd.sjoin_nearest(
            p_m[missing],
            n_m[[name_col, "geometry"]],
            how="left",
            distance_col="dist_m",
            max_distance=500
        )
        nearest_names = (nearest[name_col]
                        .groupby(level=0)
                        .first())
        points_df.loc[missing, "neighbourhood_name"] = nearest_names
    
    # Drop rows with missing neighborhood names
    points_df = points_df.dropna(subset=['neighbourhood_name'])
    
    print(f"Successfully assigned neighborhoods to {len(points_df)} points")
    return points_df


def calculate_walkability(listings_df, businesses_df):
    """Calculate walkability scores based on proximity to businesses."""
    print("Calculating walkability scores...")
    
    business_coords = businesses_df[['latitude', 'longitude']].values
    tree = KDTree(business_coords)
    walkability_scores = []
    business_diversity_scores = []
    
    k_nearest = WALKABILITY_CONFIG['k_nearest_businesses']
    max_distance = WALKABILITY_CONFIG['max_distance_meters']
    distance_norm = WALKABILITY_CONFIG['distance_normalization']
    
    for idx, listing in listings_df.iterrows():
        listing_coords = [[listing['latitude'], listing['longitude']]]
        
        # Get more candidates for distance filtering
        k_search = min(k_nearest * 3, len(business_coords))
        distances, indices = tree.query(listing_coords, k=k_search)
        
        # Calculate distances and filter
        valid_businesses = []
        for i, dist_idx in enumerate(indices[0]):
            business = businesses_df.iloc[dist_idx]
            distance = distances[0][i] * 111000  # Rough conversion to meters
            
            if distance <= max_distance:
                valid_businesses.append({
                    'distance': distance,
                    'stars': business.get('stars', 3.0),
                    'review_count': business.get('review_count', 1),
                    'category': str(business.get('categories', 'Restaurant'))
                })
        
        # Sort by distance and take k_nearest
        valid_businesses = sorted(valid_businesses, key=lambda x: x['distance'])[:k_nearest]
        
        if len(valid_businesses) == 0:
            walkability_scores.append(0.0)
            business_diversity_scores.append(0.0)
        else:
            # Calculate quality-weighted walkability
            quality_weighted_score = 0
            total_weight = 0
            categories = set()
            
            for business in valid_businesses:
                # Distance decay
                distance_weight = 1 / (1 + business['distance'] / distance_norm)
                # Quality weight
                quality_weight = (business['stars'] / 5.0) * np.log(1 + business['review_count'])
                # Combined weight
                combined_weight = distance_weight * quality_weight
                quality_weighted_score += combined_weight
                total_weight += distance_weight
                
                # Track categories
                category = business['category']
                categories.add(category.split(',')[0].strip() if isinstance(category, str) else 'Unknown')
            
            walkability_score = quality_weighted_score / max(total_weight, 1)
            walkability_scores.append(float(walkability_score))
            
            # Business diversity score
            diversity_score = len(categories) / 10.0
            business_diversity_scores.append(float(min(diversity_score, 1.0)))
    
    print(f"Walkability calculation complete. Mean score: {np.mean(walkability_scores):.3f}")
    return walkability_scores, business_diversity_scores


def calculate_neighborhood_quality(listings_gdf, businesses_df):
    """Calculate neighborhood quality based on nearby business ratings."""
    print("Calculating neighborhood quality...")
    
    radius = NEIGHBORHOOD_QUALITY_CONFIG['quality_radius_meters']
    distance_norm = NEIGHBORHOOD_QUALITY_CONFIG['distance_normalization']
    max_businesses = NEIGHBORHOOD_QUALITY_CONFIG['max_businesses_for_density']
    
    businesses_gdf = gpd.GeoDataFrame(
        businesses_df,
        geometry=gpd.points_from_xy(businesses_df['longitude'], businesses_df['latitude']),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    listings_gdf_proj = listings_gdf.to_crs(epsg=3857)
    
    quality_scores = []
    business_density_scores = []
    
    for _, listing in listings_gdf_proj.iterrows():
        nearby_businesses = businesses_gdf[
            businesses_gdf.geometry.distance(listing.geometry) <= radius
        ].copy()
        
        if len(nearby_businesses) == 0:
            quality_scores.append(0.0)
            business_density_scores.append(0.0)
            continue
        
        # Distance-based weights
        distances = nearby_businesses.geometry.distance(listing.geometry)
        distance_weights = 1 / (1 + distances / distance_norm)
        
        # Handle missing data
        stars = nearby_businesses['stars'].fillna(3.0)
        review_counts = nearby_businesses['review_count'].fillna(1).clip(lower=1)
        
        # Quality calculation
        log_review_counts = np.log(1 + review_counts)
        weights = distance_weights * log_review_counts
        weighted_quality = (stars * weights).sum() / weights.sum()
        
        # Business density score
        density_score = min(len(nearby_businesses) / max_businesses, 1.0)
        
        quality_scores.append(float(weighted_quality))
        business_density_scores.append(float(density_score))
    
    print(f"Neighborhood quality complete. Mean score: {np.mean(quality_scores):.3f}")
    return quality_scores, business_density_scores


def calculate_bang_for_buck(listings_gdf):
    """Calculate enhanced bang-for-buck scores."""
    print("Calculating bang-for-buck scores...")
    
    min_reviews = CLEANING_CONFIG['min_reviews_for_analysis']
    
    # Filter valid listings
    valid_mask = (
        (listings_gdf['number_of_reviews'] >= min_reviews) &
        (listings_gdf['price'] > 0) &
        (~listings_gdf['price'].isna()) &
        (~listings_gdf['walkability_score'].isna()) &
        (~listings_gdf['neighborhood_quality'].isna()) &
        (listings_gdf['neighbourhood_name'].notna())
    )
    
    valid_listings = listings_gdf[valid_mask].copy()
    
    if len(valid_listings) == 0:
        print("No valid listings found!")
        return pd.DataFrame()
    
    # Enhanced popularity metric
    recent_weight = POPULARITY_CONFIG['recent_activity_weight']
    historical_weight = POPULARITY_CONFIG['historical_popularity_weight']
    
    review_velocity = valid_listings['reviews_per_month'].fillna(0)
    total_reviews = valid_listings['number_of_reviews'].fillna(0)
    valid_listings['enhanced_popularity'] = (
        review_velocity * 12 * recent_weight +
        total_reviews * historical_weight
    )
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = {}
    
    feature_columns = {
        'walkability_score': BANG_FOR_BUCK_WEIGHTS['walkability'],
        'neighborhood_quality': BANG_FOR_BUCK_WEIGHTS['neighborhood_quality'],
        'business_diversity': BANG_FOR_BUCK_WEIGHTS['business_diversity'],
        'business_density': BANG_FOR_BUCK_WEIGHTS['business_density'],
        'enhanced_popularity': BANG_FOR_BUCK_WEIGHTS['popularity'],
        'price': BANG_FOR_BUCK_WEIGHTS['price']
    }
    
    for feature, weight in feature_columns.items():
        if feature in valid_listings.columns:
            if feature == 'price':
                # Invert price (lower is better)
                price_values = valid_listings[feature].values
                transformed_price = -np.log(price_values / (price_values.min() + 1e-6))
                normalized_features[feature] = scaler.fit_transform(transformed_price.reshape(-1, 1)).flatten()
            else:
                feature_values = valid_listings[feature].fillna(valid_listings[feature].median()).values
                normalized_features[feature] = scaler.fit_transform(feature_values.reshape(-1, 1)).flatten()
    
    # Calculate composite score
    composite_score = np.zeros(len(valid_listings))
    for feature, weight in feature_columns.items():
        if feature in normalized_features:
            composite_score += normalized_features[feature] * abs(weight)
    
    valid_listings['composite_score'] = composite_score
    
    # Confidence adjustment
    confidence_full = POPULARITY_CONFIG['confidence_full_reviews']
    valid_listings['confidence'] = valid_listings['number_of_reviews'].apply(
        lambda x: min(1.0, np.log(1 + x) / np.log(confidence_full + 1))
    )
    
    valid_listings['bang_for_buck_score'] = valid_listings['composite_score'] * valid_listings['confidence']
    
    print(f"Bang-for-buck calculation complete for {len(valid_listings)} listings.")
    return valid_listings


def get_neighborhood_rankings(results_df):
    """Get comprehensive neighborhood rankings."""
    print("Calculating neighborhood rankings...")
    
    min_listings = CLEANING_CONFIG['min_listings_per_neighborhood']
    neighborhood_stats = []
    
    if 'neighbourhood_name' not in results_df.columns:
        print("Error: 'neighbourhood_name' column not found")
        return pd.DataFrame()
    
    for neighborhood in results_df['neighbourhood_name'].unique():
        neighborhood_data = results_df[results_df['neighbourhood_name'] == neighborhood]
        
        if len(neighborhood_data) < min_listings:
            continue
        
        bang_for_buck = neighborhood_data['bang_for_buck_score']
        
        stats_dict = {
            'neighborhood': neighborhood,
            'listing_count': len(neighborhood_data),
            'avg_bang_for_buck': bang_for_buck.mean(),
            'median_bang_for_buck': bang_for_buck.median(),
            'std_bang_for_buck': bang_for_buck.std(),
            'confidence_interval_lower': bang_for_buck.quantile(0.25),
            'confidence_interval_upper': bang_for_buck.quantile(0.75),
            'avg_price': neighborhood_data['price'].mean(),
            'median_price': neighborhood_data['price'].median(),
            'avg_walkability': neighborhood_data['walkability_score'].mean(),
            'avg_quality': neighborhood_data['neighborhood_quality'].mean(),
            'avg_diversity': neighborhood_data['business_diversity'].mean(),
            'avg_density': neighborhood_data['business_density'].mean(),
            'avg_popularity': neighborhood_data['enhanced_popularity'].mean(),
            'avg_confidence': neighborhood_data['confidence'].mean(),
            'total_reviews': neighborhood_data['number_of_reviews'].sum(),
        }
        
        # Composite ranking score
        consistency_bonus = 1 / (1 + stats_dict['std_bang_for_buck'] if not pd.isna(stats_dict['std_bang_for_buck']) else 1)
        stats_dict['composite_ranking'] = stats_dict['avg_bang_for_buck'] * consistency_bonus
        
        neighborhood_stats.append(stats_dict)
    
    neighborhood_df = pd.DataFrame(neighborhood_stats)
    neighborhood_df = neighborhood_df.sort_values('composite_ranking', ascending=False)
    
    print(f"Rankings calculated for {len(neighborhood_df)} neighborhoods")
    return neighborhood_df