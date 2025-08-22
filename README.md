# Nashville Tourism Analysis

> Analysis of Airbnb listings and restaurant data across Nashville neighborhoods, providing insights into pricing trends, neighborhood popularity, and walkability analysis. Perfect for travelers seeking budget-friendly accommodations and data scientists interested in urban analytics!

## 📊 Project Overview

This project conducts a comprehensive Exploratory Data Analysis (EDA) of Nashville's tourism landscape, focusing on:

- **Airbnb Market Analysis**: Price trends, room types, and neighborhood variations
- **Business & Restaurant Mapping**: Geographic distribution and category analysis  
- **Neighborhood Insights**: Comparative analysis of tourism hotspots
- **Value Discovery**: Identifying undervalued areas and investment opportunities

## 🗂️ Dataset Description

| Dataset | Records | Description |
|---------|---------|-------------|
| Airbnb Listings | 9,000+ | Short-term rental properties with pricing, reviews, and location data |
| Nashville Businesses | Various | Local business registrations and categories |
| Nashville Restaurants | Various | Restaurant listings with cuisine types and locations |
| Geographic Boundaries | Multiple | Zip codes and neighborhood boundary data (GeoJSON) |

## 🔍 Key Insights

- **Market Segmentation**: Clear pricing differentiation by room type and neighborhood
- **Geographic Patterns**: Distinct tourism zones with varying price-to-value ratios
- **Business Clusters**: Restaurant and business density correlates with tourism activity
- **Investment Opportunities**: Identified undervalued neighborhoods with growth potential
- **Seasonal Trends**: Price variations suggest market dynamics and demand patterns

## 📁 Repository Structure

```
Nashville-Tourism-Analysis/
├── EDA_consolidated_code.ipynb    # Main analysis notebook
├── data/                          # Raw datasets
│   ├── listings.csv              # Airbnb listings data
│   ├── nashville_businesses.csv  # Business registration data
│   ├── nashville_restaurants.csv # Restaurant listings
│   ├── nashville_zipcodes.geojson # Zip code boundaries
│   └── neighbourhoods.geojson    # Neighborhood boundaries
├── figures/                       # Generated visualizations
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required Python packages
pandas
numpy
matplotlib
seaborn
geopandas
scikit-learn
folium
geopy
scipy
```

### Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/seemadhungana/Nashville-Tourism-Analysis.git
   cd Nashville-Tourism-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn geopandas scikit-learn folium geopy scipy
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook EDA_consolidated_code.ipynb
   ```

## 📈 Analysis Highlights

### Price Distribution Analysis
- Room type significantly impacts pricing (Entire home > Private room > Shared room)
- Price distribution is right-skewed with most listings under $200/night
- Geographic location is the strongest price predictor

### Neighborhood Insights
- Premium neighborhoods have fewer but higher-priced listings
- Market concentration varies significantly across areas
- Strong correlation between location and value proposition

### Business Landscape
- Diverse restaurant and business ecosystem
- Geographic clustering around tourism hotspots
- Category distribution reflects local economy

## 🎯 Business Applications

- **For Travelers**: Identify budget-friendly neighborhoods with good amenities
- **For Investors**: Discover undervalued areas with growth potential
- **For City Planners**: Understand tourism impact and infrastructure needs
- **For Business Owners**: Optimal location selection based on market analysis

## 🔧 Technical Features

- **Reproducible Analysis**: Single seed (42) ensures consistent results
- **Interactive Visualizations**: Colorblind-safe plots with professional styling
- **Geographic Mapping**: Choropleth maps and boundary analysis
- **Statistical Analysis**: Correlation analysis and market metrics
- **Clean Code**: Well-documented, modular structure

## 📊 Sample Visualizations

The notebook generates several key visualizations:
- Price distribution by room type and neighborhood
- Geographic heat maps of listing density
- Business category analysis
- Market concentration scatter plots
- Comprehensive analytics dashboard

## 🔮 Future Enhancements

- **Predictive Modeling**: Machine learning models for price prediction
- **Temporal Analysis**: Seasonal trends and booking patterns
- **Interactive Dashboard**: Web-based exploration tool
- **Real-time Updates**: API integration for live data
- **Walkability Scores**: Integration with transportation data

## 📝 Data Sources

- Airbnb listing data (anonymized for privacy)
- Nashville Open Data Portal
- Geographic boundary files from local government sources
- Business registration databases

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Seema Dhungana**
- GitHub: [@seemadhungana](https://github.com/seemadhungana)
- Project Link: [Nashville Tourism Analysis](https://github.com/seemadhungana/Nashville-Tourism-Analysis)

## 🙏 Acknowledgments

- Nashville Metro Government for open data initiatives
- Airbnb for making anonymized data available for research
- Open source community for the excellent data analysis tools

---

*Perfect for travelers seeking budget-friendly accommodations and data scientists interested in urban analytics!*
