"""
Smart Supplier Recommendation System
Professional Streamlit Web Application
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

# Custom CSS for professional light theme
st.markdown("""
<style>
    .main { background-color: #F4F7FE; }
    .main-header {
        font-size: 2.5rem; font-weight: 700; text-align: center; padding: 1.5rem 0;
        background: linear-gradient(135deg, #4318FF 0%, #7551FF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem; font-weight: 600; color: #1B2559; margin-top: 1rem;
        border-bottom: 3px solid #4318FF; padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #FFFFFF; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #E2E8F0;
        margin: 0.5rem 0; transition: all 0.3s ease;
    }
    .metric-card:hover { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); transform: translateY(-4px); }
    .stButton>button {
        background: linear-gradient(135deg, #4318FF 0%, #7551FF 100%); color: white;
        font-weight: 600; border-radius: 10px; padding: 0.75rem 2rem; border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); transition: all 0.3s ease;
    }
    .stButton>button:hover { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); transform: translateY(-2px); }
    .filter-section {
        background-color: #FFFFFF; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #E2E8F0; margin-bottom: 1rem;
    }
    .dataframe thead tr th {
        background: linear-gradient(135deg, #4318FF 0%, #7551FF 100%);
        color: white !important; font-weight: 600; padding: 1rem;
    }
    .dataframe tbody tr:hover { background-color: rgba(67, 24, 255, 0.05); }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #1B2559; }
    [data-testid="stMetricLabel"] { font-size: 0.875rem; font-weight: 500; color: #A3AED0; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


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
def train_models(_features_df, _interactions_df):
    """Train and cache all ML models"""
    models = {}
    
    # 1. Supplier Indexing Model
    indexing_model = SupplierIndexingModel()
    features_with_index = indexing_model.calculate_supplier_index(_features_df)
    models['indexing'] = indexing_model
    
    # 2. Clustering Model
    clustering_model = SupplierClusteringModel(n_clusters=4)
    clusters, pca_data = clustering_model.fit(features_with_index)
    cluster_profiles = clustering_model.get_cluster_profiles(features_with_index, clusters)
    models['clustering'] = clustering_model
    models['clusters'] = clusters
    models['pca_data'] = pca_data
    models['cluster_profiles'] = cluster_profiles
    
    # 3. Random Forest Model
    rf_model = SupplierRankingModel(model_type='random_forest')
    rf_model.train(features_with_index, target_col='Supplier_Score')
    models['random_forest'] = rf_model
    
    # 4. XGBoost Model
    xgb_model = SupplierRankingModel(model_type='xgboost')
    xgb_model.train(features_with_index, target_col='Supplier_Score')
    models['xgboost'] = xgb_model
    
    # 5. Collaborative Filtering
    if _interactions_df is not None and len(_interactions_df) > 0:
        cf_model = CollaborativeFilteringModel()
        cf_model.fit(_interactions_df)
        models['collaborative_filtering'] = cf_model
    
    return models, features_with_index


def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Smart Supplier Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Procurement Intelligence Platform")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        with st.spinner("🔄 Loading data and training models... This may take a minute..."):
            loader, data = load_data()
            features_df = prepare_ml_features(loader)
            interactions_df = loader.prepare_recommendation_data()
            models, features_with_index = train_models(features_df, interactions_df)
            
            st.session_state.data_loaded = True
            st.session_state.loader = loader
            st.session_state.data = data
            st.session_state.features_df = features_with_index
            st.session_state.interactions_df = interactions_df
            st.session_state.models = models
        
        st.success("✅ Data loaded and models trained successfully!")
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home & Overview", "🎯 Supplier Recommendations", "📈 Supplier Analytics", "🔬 ML Model Insights", "📊 Clustering Analysis"]
    )
    
    # Route to selected page
    if page == "🏠 Home & Overview":
        show_home_page()
    elif page == "🎯 Supplier Recommendations":
        show_recommendation_page()
    elif page == "📈 Supplier Analytics":
        show_analytics_page()
    elif page == "🔬 ML Model Insights":
        show_ml_insights_page()
    elif page == "📊 Clustering Analysis":
        show_clustering_page()


def show_home_page():
    """Home page with overview and key metrics"""
    st.markdown('<h2 class="sub-header">System Overview</h2>', unsafe_allow_html=True)
    
    features_df = st.session_state.features_df
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Suppliers", f"{len(features_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Countries", f"{features_df['Country'].nunique()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_score = features_df['Supplier_Score'].mean()
        st.metric("Avg Supplier Score", f"{avg_score:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        tier1_count = (features_df['Index_Tier'] == 'Tier 1').sum()
        st.metric("Tier 1 Suppliers", f"{tier1_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_supplier_distribution(features_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(plot_supplier_scores_distribution(features_df), use_container_width=True)
    
    # Top suppliers
    st.markdown('<h2 class="sub-header">Top 10 Suppliers</h2>', unsafe_allow_html=True)
    st.plotly_chart(plot_top_suppliers(features_df, n=10), use_container_width=True)
    
    # Performance matrix
    st.markdown('<h2 class="sub-header">Supplier Performance Matrix</h2>', unsafe_allow_html=True)
    st.plotly_chart(plot_performance_matrix(features_df), use_container_width=True)


def show_recommendation_page():
    """Supplier recommendation page with advanced filters"""
    st.markdown('<h2 class="sub-header">🎯 Get Supplier Recommendations</h2>', unsafe_allow_html=True)
    
    features_df = st.session_state.features_df
    loader = st.session_state.loader
    models = st.session_state.models
    
    # Filter section
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Filter Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=3,
            help="How many suppliers to recommend"
        )
        
        supplier_cutoff = st.slider(
            "Minimum Supplier Score",
            min_value=0,
            max_value=100,
            value=50,
            help="Minimum score threshold for recommendations"
        )
    
    with col2:
        countries = ['All'] + loader.get_countries()
        selected_country = st.selectbox(
            "Country",
            countries,
            help="Filter by supplier country"
        )
        
        categories, subcategories = loader.get_material_categories()
        selected_category = st.selectbox(
            "Material Category",
            ['All'] + categories,
            help="Select material category"
        )
    
    with col3:
        if selected_category != 'All':
            # Filter subcategories based on selected category
            materials_df = st.session_state.data['materials']
            materials_df.columns = ['Category', 'Subcategory']
            filtered_subs = materials_df[materials_df['Category'] == selected_category]['Subcategory'].unique().tolist()
            selected_subcategory = st.selectbox(
                "Material Subcategory",
                ['All'] + filtered_subs,
                help="Select material subcategory"
            )
        else:
            selected_subcategory = st.selectbox(
                "Material Subcategory",
                ['All'] + subcategories,
                help="Select material subcategory"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get recommendations button
    if st.button("🚀 Get Recommendations", use_container_width=True):
        with st.spinner("Analyzing suppliers..."):
            # Apply filters
            filtered_df = features_df.copy()
            
            if selected_country != 'All':
                filtered_df = filtered_df[filtered_df['Country'] == selected_country]
            
            filtered_df = filtered_df[filtered_df['Supplier_Score'] >= supplier_cutoff]
            
            # Sort by score and get top N
            recommendations = filtered_df.nlargest(n_recommendations, 'Supplier_Score')
            
            if len(recommendations) == 0:
                st.warning("⚠️ No suppliers found matching your criteria. Try adjusting the filters.")
            else:
                st.success(f"✅ Found {len(recommendations)} recommended suppliers!")
                
                # Display recommendations
                st.markdown('<h3 class="sub-header">Recommended Suppliers</h3>', unsafe_allow_html=True)
                
                for idx, (_, supplier) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"#{idx} - {supplier['Supplier']} (Score: {supplier['Supplier_Score']:.1f})"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**📋 Basic Information**")
                            st.write(f"**SAP ID:** {supplier['SAP ID']}")
                            st.write(f"**Country:** {supplier['Country']}")
                            st.write(f"**Index Tier:** {supplier['Index_Tier']}")
                        
                        with col2:
                            st.markdown("**⭐ Performance Scores**")
                            st.write(f"**Quality:** {supplier.get('Quality score', 0):.1f}/100")
                            st.write(f"**Delivery:** {supplier.get('Delivery score', 0):.1f}/100")
                            st.write(f"**Communication:** {supplier.get('Communication score', 0):.1f}/100")
                        
                        with col3:
                            st.markdown("**💰 Financial & Risk**")
                            st.write(f"**D&B Rating:** {supplier.get('D&B Rating', 'N/A')}")
                            st.write(f"**Financial Risk:** {supplier.get('Financial Risk', 0):.1f}")
                            st.write(f"**Total Spend:** ${supplier.get('Total_Spend', 0):,.0f}")
                
                # Export recommendations
                st.markdown("---")
                st.download_button(
                    label="📥 Download Recommendations (CSV)",
                    data=recommendations.to_csv(index=False),
                    file_name=f"supplier_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def show_analytics_page():
    """Supplier analytics and insights page"""
    st.markdown('<h2 class="sub-header">📈 Supplier Analytics</h2>', unsafe_allow_html=True)
    
    features_df = st.session_state.features_df
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "⚠️ Risk Assessment", "🌱 Sustainability"])
    
    with tab1:
        st.markdown("### Performance Analysis")
        st.plotly_chart(plot_performance_matrix(features_df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_top_suppliers(features_df, n=15, score_col='Quality score'), use_container_width=True)
        with col2:
            st.plotly_chart(plot_top_suppliers(features_df, n=15, score_col='Delivery score'), use_container_width=True)
    
    with tab2:
        st.markdown("### Risk Assessment")
        risk_fig = plot_risk_assessment(features_df)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)
        
        # High risk suppliers
        st.markdown("#### ⚠️ High Risk Suppliers")
        high_risk = features_df.nlargest(10, 'Financial Risk')[['Supplier', 'Country', 'Financial Risk', 'Compliance risk', 'Supply Chain Risk']]
        st.dataframe(high_risk, use_container_width=True)
    
    with tab3:
        st.markdown("### ESG Sustainability Scores")
        esg_fig = plot_sustainability_scores(features_df)
        if esg_fig:
            st.plotly_chart(esg_fig, use_container_width=True)
        
        # Top sustainable suppliers
        st.markdown("#### 🌱 Top Sustainable Suppliers")
        if 'Environmental score' in features_df.columns:
            top_sustainable = features_df.nlargest(10, 'Environmental score')[['Supplier', 'Country', 'Environmental score', 'Social Score', 'Governance score']]
            st.dataframe(top_sustainable, use_container_width=True)


def show_ml_insights_page():
    """ML model insights and feature importance"""
    st.markdown('<h2 class="sub-header">🔬 Machine Learning Model Insights</h2>', unsafe_allow_html=True)
    
    models = st.session_state.models
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost"]
    )
    
    if model_type == "Random Forest":
        model = models['random_forest']
    else:
        model = models['xgboost']
    
    # Feature importance
    st.markdown("### 📊 Feature Importance")
    importance_df = model.get_feature_importance()
    
    if importance_df is not None:
        st.plotly_chart(plot_feature_importance(importance_df, top_n=20), use_container_width=True)
        
        # Show table
        with st.expander("📋 View Full Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True)
    
    # Model performance
    st.markdown("### 🎯 Model Performance")
    col1, col2 = st.columns(2)
    
    # You would need to store these metrics during training
    with col1:
        st.metric("Model Type", model_type)
        st.metric("Number of Features", len(model.feature_cols))
    
    with col2:
        st.metric("Training Status", "✅ Trained")
        st.metric("Model Complexity", "Medium")


def show_clustering_page():
    """Clustering analysis page"""
    st.markdown('<h2 class="sub-header">📊 Supplier Clustering Analysis</h2>', unsafe_allow_html=True)
    
    models = st.session_state.models
    features_df = st.session_state.features_df
    
    # PCA Scatter plot
    st.markdown("### 🎨 Supplier Segmentation Visualization")
    cluster_fig = plot_cluster_scatter(
        models['pca_data'],
        models['clusters'],
        models['cluster_profiles']
    )
    st.plotly_chart(cluster_fig, use_container_width=True)
    
    # Cluster profiles
    st.markdown("### 📋 Cluster Profiles")
    cluster_profile_fig = plot_cluster_profiles(models['cluster_profiles'])
    st.plotly_chart(cluster_profile_fig, use_container_width=True)
    
    # Detailed cluster information
    st.markdown("### 📊 Cluster Details")
    
    for profile in models['cluster_profiles']:
        with st.expander(f"Cluster {profile['cluster_id'] + 1} - {profile['size']} suppliers"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Suppliers", profile['size'])
                st.metric("Avg Spend", f"${profile['avg_spend']:,.0f}")
            
            with col2:
                st.metric("Avg Quality", f"{profile['avg_quality']:.1f}")
                st.metric("Avg Delivery", f"{profile['avg_delivery']:.1f}")
            
            with col3:
                st.metric("Avg Index", f"{profile['avg_index']:.1f}")
                
                # Top countries in cluster
                if profile['top_countries']:
                    st.markdown("**Top Countries:**")
                    for country, count in list(profile['top_countries'].items())[:3]:
                        st.write(f"• {country}: {count}")


if __name__ == "__main__":
    main()
