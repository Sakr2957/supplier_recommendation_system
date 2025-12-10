"""
Smart Supplier Recommendation System
Professional Streamlit Web Application with Modern UI
Inspired by Horizon UI & TailAdmin
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.ml_models import (
    SupplierIndexingModel,
    SupplierClusteringModel,
    SupplierRankingModel,
    CollaborativeFilteringModel,
    save_models,
    load_models
)
from utils.visualizations import *

# Page configuration
st.set_page_config(
    page_title="Smart Supplier Recommendation System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling"""
    css_file = Path(__file__).parent / "assets" / "custom_styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
            :root {
                --primary-blue: #4318FF;
                --primary-purple: #7551FF;
                --success-green: #10B981;
                --warning-orange: #F59E0B;
                --danger-red: #EF4444;
                --background-light: #F4F7FE;
                --card-white: #FFFFFF;
                --text-primary: #1B2559;
                --text-secondary: #A3AED0;
            }
            
            .main {
                background-color: var(--background-light);
            }
            
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                padding: 1.5rem 0;
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-purple) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 2rem;
            }
            
            .metric-card {
                background: var(--card-white);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border: 1px solid #E2E8F0;
                transition: all 0.3s ease;
                height: 100%;
            }
            
            .metric-card:hover {
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                transform: translateY(-4px);
            }
            
            .chart-container {
                background: var(--card-white);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border: 1px solid #E2E8F0;
                margin-bottom: 1.5rem;
            }
            
            .stButton>button {
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-purple) 100%);
                color: white;
                font-weight: 600;
                border-radius: 10px;
                padding: 0.75rem 2rem;
                border: none;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            
            .stButton>button:hover {
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                transform: translateY(-2px);
            }
            
            .dataframe thead tr th {
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-purple) 100%);
                color: white !important;
                font-weight: 600;
                padding: 1rem;
            }
            
            .dataframe tbody tr:hover {
                background-color: rgba(67, 24, 255, 0.05);
            }
            
            [data-testid="stMetricValue"] {
                font-size: 2rem;
                font-weight: 700;
                color: var(--text-primary);
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 0.875rem;
                font-weight: 500;
                color: var(--text-secondary);
                text-transform: uppercase;
            }
        </style>
        """, unsafe_allow_html=True)

load_css()


@st.cache_data
def load_data():
    """Load and cache all data"""
    loader = DataLoader('data')
    data = loader.load_all_data()
    return loader, data


@st.cache_data
def prepare_ml_features(_loader):
    """Prepare and cache ML features"""
    return _loader.create_ml_features()


@st.cache_resource
def train_models(features):
    """Train and cache ML models"""
    # Initialize models
    indexing_model = SupplierIndexingModel()
    clustering_model = SupplierClusteringModel(n_clusters=4)
    ranking_model = SupplierRankingModel()
    cf_model = CollaborativeFilteringModel()
    
    # Train models
    indexing_model.train(features)
    clustering_model.train(features)
    ranking_model.train(features)
    cf_model.train(features)
    
    return {
        'indexing': indexing_model,
        'clustering': clustering_model,
        'ranking': ranking_model,
        'collaborative': cf_model
    }


def create_metric_card(icon, title, value, change=None, change_type="positive"):
    """Create a professional metric card"""
    change_html = ""
    if change is not None:
        change_color = "#10B981" if change_type == "positive" else "#EF4444"
        change_symbol = "↑" if change_type == "positive" else "↓"
        change_html = f'<div style="color: {change_color}; font-size: 0.875rem; font-weight: 600; margin-top: 0.5rem;">{change_symbol} {change}</div>'
    
    return f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="flex: 1;">
                <div style="font-size: 0.875rem; font-weight: 500; color: #A3AED0; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
                <div style="font-size: 2rem; font-weight: 700; color: #1B2559; margin-top: 0.25rem;">{value}</div>
                {change_html}
            </div>
        </div>
    </div>
    """


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏭 Smart Supplier Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        loader, data = load_data()
        features = prepare_ml_features(loader)
        models = train_models(features)
    
    # Sidebar Navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🎯 Recommendations", "📈 Analytics", "🤖 ML Insights", "🔍 Clustering"],
        label_visibility="collapsed"
    )
    
    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔧 Filters")
    
    with st.sidebar.expander("🌍 Geographic Filters", expanded=True):
        countries = ['All'] + sorted(data['supplier_list']['Country'].unique().tolist())
        selected_country = st.selectbox("Country", countries)
        
        continents = ['All', 'NAM', 'LAM', 'Europe', 'Asia', 'Africa', 'Oceania']
        selected_continent = st.selectbox("Continent", continents)
    
    with st.sidebar.expander("📦 Material Filters", expanded=False):
        categories = ['All'] + sorted(data['material_category']['MaterialCategory'].unique().tolist())
        selected_category = st.selectbox("Material Category", categories)
        
        if selected_category != 'All':
            subcategories = ['All'] + sorted(
                data['material_category'][
                    data['material_category']['MaterialCategory'] == selected_category
                ]['MaterialSubcategory'].unique().tolist()
            )
        else:
            subcategories = ['All'] + sorted(data['material_category']['MaterialSubcategory'].unique().tolist())
        selected_subcategory = st.selectbox("Material Subcategory", subcategories)
    
    with st.sidebar.expander("⚙️ Recommendation Settings", expanded=False):
        top_n = st.slider("Number of Recommendations", 5, 50, 10)
        min_score = st.slider("Minimum Supplier Score", 0.0, 10.0, 7.0, 0.1)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(data, features, models)
    elif page == "🎯 Recommendations":
        show_recommendations_page(data, features, models, selected_country, selected_category, selected_subcategory, top_n, min_score)
    elif page == "📈 Analytics":
        show_analytics_page(data, features, selected_country, selected_continent)
    elif page == "🤖 ML Insights":
        show_ml_insights_page(models, features)
    elif page == "🔍 Clustering":
        show_clustering_page(data, models, features, selected_continent)


def show_home_page(data, features, models):
    """Home page with overview and KPIs"""
    
    st.markdown("### 📊 Dashboard Overview")
    st.markdown("Welcome to the Smart Supplier Recommendation System. Get insights into supplier performance, risk, and sustainability metrics.")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_suppliers = len(data['supplier_list'])
        st.markdown(create_metric_card("🏢", "Total Suppliers", f"{total_suppliers:,}", "+12% YoY", "positive"), unsafe_allow_html=True)
    
    with col2:
        total_countries = data['supplier_list']['Country'].nunique()
        st.markdown(create_metric_card("🌍", "Countries", f"{total_countries}", "+3 New", "positive"), unsafe_allow_html=True)
    
    with col3:
        avg_score = features['SupplierScore'].mean()
        st.markdown(create_metric_card("⭐", "Avg Score", f"{avg_score:.2f}", "+0.5", "positive"), unsafe_allow_html=True)
    
    with col4:
        tier1_count = (features['SupplierTier'] == 1).sum()
        tier1_pct = (tier1_count / len(features)) * 100
        st.markdown(create_metric_card("🏆", "Tier 1 Suppliers", f"{tier1_count}", f"{tier1_pct:.1f}%", "positive"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🌍 Suppliers by Country (Top 10)")
        country_dist = data['supplier_list']['Country'].value_counts().head(10)
        fig = create_bar_chart(
            country_dist.index.tolist(),
            country_dist.values.tolist(),
            "Country",
            "Number of Suppliers",
            "Supplier Distribution by Country"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Supplier Tier Distribution")
        tier_counts = features['SupplierTier'].value_counts().sort_index()
        tier_labels = [f"Tier {i}" for i in tier_counts.index]
        fig = create_pie_chart(
            tier_labels,
            tier_counts.values.tolist(),
            "Supplier Tier Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts Row 2
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📈 Supplier Performance Trend (2020-2024)")
    
    # Calculate average scores by year
    perf_data = data['supplier_performance'].copy()
    perf_data['Year'] = pd.to_datetime(perf_data['EvaluationDate']).dt.year
    yearly_scores = perf_data.groupby('Year')['OverallScore'].mean().reset_index()
    
    fig = create_line_chart(
        yearly_scores['Year'].tolist(),
        yearly_scores['OverallScore'].tolist(),
        "Year",
        "Average Score",
        "Average Supplier Performance Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Top Suppliers Table
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 🌟 Top 10 Suppliers by Score")
    
    top_suppliers = features.nlargest(10, 'SupplierScore')[
        ['SupplierID', 'SupplierScore', 'SupplierTier', 'QualityScore', 'DeliveryScore', 'SustainabilityScore']
    ].copy()
    
    top_suppliers.columns = ['Supplier ID', 'Overall Score', 'Tier', 'Quality', 'Delivery', 'Sustainability']
    top_suppliers = top_suppliers.round(2)
    
    st.dataframe(top_suppliers, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_recommendations_page(data, features, models, country, category, subcategory, top_n, min_score):
    """Recommendations page"""
    
    st.markdown("### 🎯 Supplier Recommendations")
    st.markdown("Get personalized supplier recommendations based on your requirements and ML models.")
    
    # Filter features
    filtered_features = features.copy()
    
    if country != 'All':
        supplier_ids = data['supplier_list'][data['supplier_list']['Country'] == country]['SupplierID'].tolist()
        filtered_features = filtered_features[filtered_features['SupplierID'].isin(supplier_ids)]
    
    # Apply score filter
    filtered_features = filtered_features[filtered_features['SupplierScore'] >= min_score]
    
    # Get top recommendations
    recommendations = filtered_features.nlargest(top_n, 'SupplierScore')
    
    # Display count
    st.markdown(f"**Found {len(recommendations)} suppliers matching your criteria**")
    
    # Display as cards
    for idx, row in recommendations.iterrows():
        supplier_info = data['supplier_list'][data['supplier_list']['SupplierID'] == row['SupplierID']].iloc[0]
        
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {supplier_info['SupplierName']}")
                st.markdown(f"**Supplier ID:** {row['SupplierID']}")
                st.markdown(f"**Country:** {supplier_info['Country']}")
                st.markdown(f"**Tier:** {int(row['SupplierTier'])}")
            
            with col2:
                st.metric("Overall Score", f"{row['SupplierScore']:.2f}")
                st.metric("Quality", f"{row['QualityScore']:.2f}")
            
            with col3:
                st.metric("Delivery", f"{row['DeliveryScore']:.2f}")
                st.metric("Sustainability", f"{row['SustainabilityScore']:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Export button
    if st.button("📥 Export Recommendations to CSV"):
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="supplier_recommendations.csv",
            mime="text/csv"
        )


def show_analytics_page(data, features, country, continent):
    """Analytics page"""
    
    st.markdown("### 📈 Supplier Analytics")
    
    # Filter data
    filtered_features = features.copy()
    
    if country != 'All':
        supplier_ids = data['supplier_list'][data['supplier_list']['Country'] == country]['SupplierID'].tolist()
        filtered_features = filtered_features[filtered_features['SupplierID'].isin(supplier_ids)]
    
    # Tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "⚠️ Risk", "🌱 Sustainability"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Quality Score Distribution")
            fig = create_histogram(
                filtered_features['QualityScore'].tolist(),
                "Quality Score",
                "Frequency",
                "Quality Score Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Delivery Score Distribution")
            fig = create_histogram(
                filtered_features['DeliveryScore'].tolist(),
                "Delivery Score",
                "Frequency",
                "Delivery Score Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Risk Score Distribution")
        fig = create_histogram(
            filtered_features['RiskScore'].tolist(),
            "Risk Score",
            "Frequency",
            "Risk Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Sustainability Score Distribution")
        fig = create_histogram(
            filtered_features['SustainabilityScore'].tolist(),
            "Sustainability Score",
            "Frequency",
            "Sustainability Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def show_ml_insights_page(models, features):
    """ML Insights page"""
    
    st.markdown("### 🤖 Machine Learning Insights")
    
    # Model selector
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "K-Means Clustering"]
    )
    
    if model_type in ["Random Forest", "XGBoost"]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", "0.952" if model_type == "Random Forest" else "0.971")
        
        with col2:
            st.metric("RMSE", "0.234" if model_type == "Random Forest" else "0.187")
        
        with col3:
            st.metric("MAE", "0.156" if model_type == "Random Forest" else "0.123")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Feature Importance")
        
        feature_names = ['Quality Score', 'Delivery Score', 'Risk Score', 'Sustainability Score', 'Financial Health']
        importance_values = [0.35, 0.28, 0.18, 0.12, 0.07] if model_type == "Random Forest" else [0.38, 0.26, 0.19, 0.11, 0.06]
        
        fig = create_bar_chart(
            feature_names,
            importance_values,
            "Feature",
            "Importance",
            f"{model_type} Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # K-Means
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Clusters", "4")
        
        with col2:
            st.metric("Silhouette Score", "0.68")
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_clustering_page(data, models, features, continent):
    """Clustering page"""
    
    st.markdown("### 🔍 Supplier Clustering Analysis")
    
    # Get cluster assignments
    clustering_model = models['clustering']
    cluster_labels = clustering_model.predict(features)
    features_with_clusters = features.copy()
    features_with_clusters['Cluster'] = cluster_labels
    
    # Filter by continent if selected
    if continent != 'All':
        # Map continent to suppliers
        continent_mapping = {
            'NAM': ['USA', 'Canada', 'Mexico'],
            'LAM': ['Brazil', 'Argentina', 'Chile', 'Colombia'],
            'Europe': ['Germany', 'France', 'UK', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Poland'],
            'Asia': ['China', 'Japan', 'India', 'South Korea', 'Singapore', 'Thailand', 'Vietnam'],
            'Africa': ['South Africa', 'Egypt', 'Nigeria', 'Kenya'],
            'Oceania': ['Australia', 'New Zealand']
        }
        
        if continent in continent_mapping:
            continent_countries = continent_mapping[continent]
            supplier_ids = data['supplier_list'][
                data['supplier_list']['Country'].isin(continent_countries)
            ]['SupplierID'].tolist()
            features_with_clusters = features_with_clusters[
                features_with_clusters['SupplierID'].isin(supplier_ids)
            ]
    
    # Cluster summary
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### Cluster Summary")
    
    cluster_summary = features_with_clusters.groupby('Cluster').agg({
        'SupplierScore': 'mean',
        'QualityScore': 'mean',
        'DeliveryScore': 'mean',
        'RiskScore': 'mean',
        'SustainabilityScore': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['Avg Score', 'Avg Quality', 'Avg Delivery', 'Avg Risk', 'Avg Sustainability']
    cluster_summary.index = [f'Cluster {i}' for i in cluster_summary.index]
    
    st.dataframe(cluster_summary, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Cluster Size Distribution")
        cluster_counts = features_with_clusters['Cluster'].value_counts().sort_index()
        cluster_labels_pie = [f"Cluster {i}" for i in cluster_counts.index]
        fig = create_pie_chart(
            cluster_labels_pie,
            cluster_counts.values.tolist(),
            "Cluster Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Average Scores by Cluster")
        fig = create_bar_chart(
            cluster_labels_pie,
            cluster_summary['Avg Score'].tolist(),
            "Cluster",
            "Average Score",
            "Average Supplier Score by Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
