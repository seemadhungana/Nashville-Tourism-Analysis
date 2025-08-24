"""
Configuration settings for Nashville Tourism Analysis
"""

# Data file names (expected in input directory)
DATA_FILES = {
    'listings': 'listings.csv',
    'businesses': 'nashville_businesses.csv', 
    'restaurants': 'nashville_restaurants.csv',
    'neighborhoods': 'neighbourhoods.geojson'
}

# Data cleaning parameters
CLEANING_CONFIG = {
    'price_percentile_cutoff': 0.95,  # Remove price outliers above this percentile
    'min_reviews_for_analysis': 3,   # Minimum reviews required for bang-for-buck analysis
    'min_listings_per_neighborhood': 3  # Minimum listings per neighborhood for rankings
}

# Walkability calculation parameters
WALKABILITY_CONFIG = {
    'k_nearest_businesses': 15,      # Number of nearest businesses to consider
    'max_distance_meters': 2000,     # Maximum distance to consider for walkability
    'distance_normalization': 200    # Meters for distance weight normalization
}

# Neighborhood quality parameters  
NEIGHBORHOOD_QUALITY_CONFIG = {
    'quality_radius_meters': 1000,   # Radius for nearby business quality assessment
    'distance_normalization': 100,   # Meters for distance weight normalization
    'max_businesses_for_density': 20 # Normalization factor for business density score
}

# Bang-for-buck scoring weights (must sum to 1.0)
BANG_FOR_BUCK_WEIGHTS = {
    'walkability': 0.25,
    'neighborhood_quality': 0.20,
    'business_diversity': 0.10,
    'business_density': 0.10, 
    'popularity': 0.15,
    'price': 0.20  # Note: price is inverted (lower is better)
}

# Popularity scoring parameters
POPULARITY_CONFIG = {
    'recent_activity_weight': 0.6,   # Weight for reviews_per_month * 12
    'historical_popularity_weight': 0.4,  # Weight for total number_of_reviews
    'confidence_full_reviews': 50    # Number of reviews for full confidence (1.0)
}

# Visualization parameters
VIZ_CONFIG = {
    'map_center_coords': [36.1627, -86.7816],  # Nashville center
    'map_zoom_start': 11,
    'top_neighborhoods_to_show': 15,
    'figure_dpi': 300,
    'figure_size': (16, 10)
}

# District location mapping (based on research)
DISTRICT_LOCATIONS = {
    "District 1": "Downtown Nashville",
    "District 6": "East Nashville",
    "District 19": "Vanderbilt/West End Area",
    "District 13": "12 South",
    "District 29": "Antioch",
    "District 21": "Music Row",
    "District 2": "Germantown", 
    "District 8": "The Nations",
    "District 31": "Brentwood Area",
    "District 32": "Nolensville",
    "District 22": "Bellevue",
    "District 23": "Green Hills",
    "District 33": "Cane Ridge",
    "District 34": "Forest Hills",
    "District 35": "Bellevue/West Meade",
    "District 30": "Oak Hill",
    "District 9": "Bordeaux",
    "District 4": "Smyrna Area",
    "District 14": "Edgehill",
    "District 15": "Donelson",
    "District 16": "Woodbine",
    "District 24": "Belle Meade",
    "District 25": "Sylvan Park",
    "District 26": "Berry Hill",
    "District 27": "Radnor",
    "District 28": "Creive Hall",
    "District 5": "Old Hickory",
    "District 7": "Inglewood",
    "District 3": "Joelton",
    "District 10": "Madison",
    "District 17": "Wedgewood-Houston",
    "District 18": "Melrose",
    "District 20": "The Gulch",
    "District 12": "Hermitage",
    "District 11": "West End/Charlotte Pike"
}

# Output file names
OUTPUT_FILES = {
    'neighborhood_rankings': 'neighborhood_rankings.csv',
    'listing_analysis': 'listing_analysis.csv', 
    'interactive_map': 'enhanced_airbnb_map.html',
    'price_distribution_plot': 'price_distribution_analysis.png',
    'statistical_analysis_plot': 'statistical_analysis.png'
}