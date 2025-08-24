"""
Visualization functions for Nashville Tourism Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from scipy import stats
import os

from config import VIZ_CONFIG


def setup_plot_style():
    """Setup consistent plotting style."""
    sns.set_palette(['#E69F00', '#56B4E9', '#009E73', '#F0E442'])
    sns.set_theme(style="whitegrid", context="notebook")
    
    plt.rcParams.update({
        'figure.dpi': VIZ_CONFIG['figure_dpi'],
        'savefig.dpi': VIZ_CONFIG['figure_dpi'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': VIZ_CONFIG['figure_size'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.12,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white'
    })


def create_price_distribution_plot(listings_df):
    """Create price distribution visualization."""
    print("Creating price distribution plot...")
    setup_plot_style()
    
    fig = plt.figure(figsize=VIZ_CONFIG['figure_size'])
    fig.suptitle('Airbnb Pricing Analysis Dashboard', fontsize=22, weight='bold', y=0.95)
    
    gs = fig.add_gridspec(1, 2, top=0.88, bottom=0.12, left=0.08, right=0.92, wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']
    
    # Box plot by room type
    room_types = listings_df['room_type'].unique()
    price_95 = listings_df['price'].quantile(0.95)
    
    room_type_data = []
    for rt in room_types:
        rt_prices = listings_df[listings_df['room_type'] == rt]['price'].dropna()
        rt_prices_filtered = rt_prices[rt_prices <= price_95]
        room_type_data.append(rt_prices_filtered)
    
    # Violin plots for room types with >10 samples
    room_type_counts = [len(data) for data in room_type_data]
    violin_positions = []
    violin_data = []
    for i, (rt, data, count) in enumerate(zip(room_types, room_type_data, room_type_counts)):
        if count > 10:
            violin_positions.append(i)
            violin_data.append(data)
    
    if violin_data:
        parts = ax1.violinplot(violin_data, positions=violin_positions, widths=0.6,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            color_idx = violin_positions[i]
            pc.set_facecolor(colors[color_idx % len(colors)])
            pc.set_alpha(0.3)
            pc.set_edgecolor('none')
    
    # Box plots
    bp = ax1.boxplot(room_type_data, positions=range(len(room_types)), widths=0.25,
                    patch_artist=True, medianprops=dict(color='#2C3E50', linewidth=2))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticks(range(len(room_types)))
    ax1.set_xticklabels([rt.replace('_', ' ').title() for rt in room_types], rotation=30, ha='right')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Distribution by Room Type')
    ax1.grid(True, alpha=0.3)
    
    # Histogram with KDE
    price_filtered = listings_df['price'][listings_df['price'] <= price_95]
    n, bins, patches = ax2.hist(price_filtered, bins=40, alpha=0.7, color='#56B4E9', edgecolor='white')
    
    # KDE overlay
    kde = stats.gaussian_kde(price_filtered.dropna())
    x_range = np.linspace(price_filtered.min(), price_filtered.max(), 200)
    kde_values = kde(x_range)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_range, kde_values, color='#D55E00', linewidth=2.5)
    ax2_twin.set_ylabel('Density', color='#D55E00')
    ax2_twin.tick_params(axis='y', labelcolor='#D55E00')
    
    # Statistics lines
    mean_price = price_filtered.mean()
    median_price = price_filtered.median()
    ax2.axvline(mean_price, color='#E69F00', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.0f}')
    ax2.axvline(median_price, color='#CC79A7', linestyle='--', linewidth=2, label=f'Median: ${median_price:.0f}')
    
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Price Distribution')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    return fig


def create_statistical_analysis_plot(results_df, neighborhood_rankings):
    """Create comprehensive statistical analysis plots."""
    print("Creating statistical analysis plot...")
    setup_plot_style()
    
    top_n = VIZ_CONFIG['top_neighborhoods_to_show']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    if len(neighborhood_rankings) == 0:
        print("No neighborhood rankings available for plotting.")
        return fig
    
    # 1. Top neighborhoods by bang-for-buck
    top_neighborhoods = neighborhood_rankings.head(top_n)
    
    ax1 = axes[0, 0]
    bars = ax1.barh(range(len(top_neighborhoods)), top_neighborhoods['avg_bang_for_buck'])
    
    # Error bars for confidence intervals
    xerr_lower = top_neighborhoods['avg_bang_for_buck'] - top_neighborhoods['confidence_interval_lower']
    xerr_upper = top_neighborhoods['confidence_interval_upper'] - top_neighborhoods['avg_bang_for_buck']
    ax1.errorbar(top_neighborhoods['avg_bang_for_buck'], range(len(top_neighborhoods)),
                xerr=[xerr_lower.fillna(0), xerr_upper.fillna(0)], 
                fmt='none', color='black', alpha=0.5, capsize=3)
    
    ax1.set_yticks(range(len(top_neighborhoods)))
    ax1.set_yticklabels(top_neighborhoods['neighborhood'])
    ax1.set_xlabel('Bang-for-Buck Score (with 25-75% confidence intervals)')
    ax1.set_title(f'Top {top_n} Neighborhoods by Bang-for-Buck\n(with Statistical Confidence)',
                  fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Color bars by score
    colors = plt.cm.RdYlGn(top_neighborhoods['avg_bang_for_buck'] / top_neighborhoods['avg_bang_for_buck'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.8)
    
    # 2. Price vs Bang-for-Buck scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        neighborhood_rankings['avg_price'],
        neighborhood_rankings['avg_bang_for_buck'],
        s=neighborhood_rankings['listing_count'] * 15,
        c=neighborhood_rankings['avg_confidence'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add trend line if possible
    try:
        z = np.polyfit(neighborhood_rankings['avg_price'], neighborhood_rankings['avg_bang_for_buck'], 1)
        p = np.poly1d(z)
        ax2.plot(neighborhood_rankings['avg_price'], p(neighborhood_rankings['avg_price']),
                 "r--", alpha=0.8, linewidth=2, label='Trend Line')
        ax2.legend()
    except np.linalg.LinAlgError:
        print("Could not fit trend line (insufficient data)")
    
    ax2.set_xlabel('Average Price ($)')
    ax2.set_ylabel('Average Bang-for-Buck Score')
    ax2.set_title('Price vs Bang-for-Buck\n(Size=Listing Count, Color=Confidence)',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Average Confidence', rotation=270, labelpad=20)
    
    # 3. Component correlation heatmap
    ax3 = axes[1, 0]
    correlation_features = ['bang_for_buck_score', 'walkability_score', 'neighborhood_quality',
                          'business_diversity', 'business_density', 'enhanced_popularity', 'price']
    existing_correlation_features = [f for f in correlation_features if f in results_df.columns]
    
    if len(existing_correlation_features) > 1:
        correlation_matrix = results_df[existing_correlation_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 4. Distribution analysis
    ax4 = axes[1, 1]
    
    # Violin plot of scores by top neighborhoods
    top_5_neighborhoods = top_neighborhoods.head(5)['neighborhood'].tolist()
    violin_data = []
    labels = []
    
    for neighborhood in top_5_neighborhoods:
        neighborhood_data = results_df[results_df['neighbourhood_name'] == neighborhood]
        if len(neighborhood_data) > 3:
            violin_data.append(neighborhood_data['bang_for_buck_score'].values)
            labels.append(neighborhood[:15])  # Truncate long names
    
    if violin_data:
        parts = ax4.violinplot(violin_data, positions=range(len(violin_data)),
                              widths=0.6, showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plt.cm.RdYlGn(0.8 - i * 0.15))
            pc.set_alpha(0.7)
    
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_ylabel('Bang-for-Buck Score')
    ax4.set_title('Score Distribution for Top 5 Neighborhoods', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    return fig


def create_enhanced_map(results_df, neighborhoods_gdf=None):
    """Create enhanced interactive map."""
    print("Creating enhanced interactive map...")
    
    center_coords = VIZ_CONFIG['map_center_coords']
    zoom_start = VIZ_CONFIG['map_zoom_start']
    
    # Create base map
    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_start,
        tiles=None
    )
    
    # Add multiple tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    
    # Color mapping for bang-for-buck scores
    min_score = results_df['bang_for_buck_score'].min()
    max_score = results_df['bang_for_buck_score'].max()
    
    # Add listings as circles
    for idx, listing in results_df.iterrows():
        # Color based on bang-for-buck score
        score_normalized = (listing['bang_for_buck_score'] - min_score) / (max_score - min_score) if (max_score - min_score) != 0 else 0.5
        
        if score_normalized >= 0.8:
            color = '#2E8B57'  # Sea green (excellent)
        elif score_normalized >= 0.6:
            color = '#32CD32'  # Lime green (good)
        elif score_normalized >= 0.4:
            color = '#FFD700'  # Gold (average)
        elif score_normalized >= 0.2:
            color = '#FF8C00'  # Dark orange (below average)
        else:
            color = '#DC143C'  # Crimson (poor)
        
        # Size based on confidence
        radius = 3 + (listing['confidence'] * 7)  # 3-10 pixel radius
        
        # Create popup with detailed information
        popup_html = f"""
        <div style="width: 300px;">
            <h4>{listing['neighbourhood_name']}</h4>
            <hr>
            <b>Bang-for-Buck Score:</b> {listing['bang_for_buck_score']:.3f}<br>
            <b>Price:</b> ${listing['price']:.0f}/night<br>
            <b>Walkability:</b> {listing['walkability_score']:.2f}<br>
            <b>Neighborhood Quality:</b> {listing['neighborhood_quality']:.2f}<br>
            <b>Business Diversity:</b> {listing['business_diversity']:.2f}<br>
            <b>Reviews:</b> {listing['number_of_reviews']:.0f}<br>
            <b>Confidence:</b> {listing['confidence']:.2f}
        </div>
        """
        
        folium.CircleMarker(
            location=[listing['latitude'], listing['longitude']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=350),
            color='white',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed;
                top: 10px; right: 10px; width: 200px; height: 140px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <h4>Bang-for-Buck Score</h4>
    <p><i class="fa fa-circle" style="color:#2E8B57"></i> Excellent (0.8+)</p>
    <p><i class="fa fa-circle" style="color:#32CD32"></i> Good (0.6-0.8)</p>
    <p><i class="fa fa-circle" style="color:#FFD700"></i> Average (0.4-0.6)</p>
    <p><i class="fa fa-circle" style="color:#FF8C00"></i> Below Average (0.2-0.4)</p>
    <p><i class="fa fa-circle" style="color:#DC143C"></i> Poor (0-0.2)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def save_figures(results_df, neighborhood_rankings, neighborhoods_gdf, outdir, save_figures=False):
    """Save all visualization outputs."""
    if not save_figures:
        print("Skipping figure generation (--save-figures not specified)")
        return
    
    print(f"Saving figures to {outdir}...")
    
    # Create price distribution plot
    try:
        price_fig = create_price_distribution_plot(results_df)
        price_path = os.path.join(outdir, 'price_distribution_analysis.png')
        price_fig.savefig(price_path, dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(price_fig)
        print(f"Saved price distribution plot: {price_path}")
    except Exception as e:
        print(f"Error creating price distribution plot: {e}")
    
    # Create statistical analysis plot
    try:
        stats_fig = create_statistical_analysis_plot(results_df, neighborhood_rankings)
        stats_path = os.path.join(outdir, 'statistical_analysis.png')
        stats_fig.savefig(stats_path, dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close(stats_fig)
        print(f"Saved statistical analysis plot: {stats_path}")
    except Exception as e:
        print(f"Error creating statistical analysis plot: {e}")
    
    # Create interactive map
    try:
        interactive_map = create_enhanced_map(results_df, neighborhoods_gdf)
        map_path = os.path.join(outdir, 'enhanced_airbnb_map.html')
        interactive_map.save(map_path)
        print(f"Saved interactive map: {map_path}")
    except Exception as e:
        print(f"Error creating interactive map: {e}")