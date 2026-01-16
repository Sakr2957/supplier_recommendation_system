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

# Load PPx Ocean Theme CSS
with open('assets/ppx_theme.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# PPx Ocean Theme is now fully loaded from ppx_theme.css


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
    # Header with PPx Ocean Theme
    st.markdown('''
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(11, 122, 184, 0.1); border: 1px solid rgba(11, 122, 184, 0.2); border-radius: 9999px; padding: 0.375rem 1rem; font-size: 0.875rem; font-weight: 600; color: #0B7AB8; margin-bottom: 1rem;">
            <span style="height: 6px; width: 6px; border-radius: 50%; background: #24A896; animation: pulse 2s ease-in-out infinite;"></span>
            Trusted by 500+ Water Technology Companies
        </div>
        <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 3rem; font-weight: 700; background: linear-gradient(135deg, #096285 0%, #20B2E6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0.5rem 0;">
            🏭 Smart Supplier Recommendation System
        </h1>
        <p style="font-size: 1.125rem; color: #5D6B7F; max-width: 800px; margin: 0 auto;">
            AI-Powered Procurement Intelligence Platform for Water Technology Suppliers
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
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
        ["🏠 Home & Overview", "🎯 Supplier Recommendations", "🌟 PPx SVD Recommender", "📈 Supplier Analytics", "🔬 ML Model Insights", "📊 Clustering Analysis"]
    )
    
    # Route to selected page
    if page == "🏠 Home & Overview":
        show_home_page()
    elif page == "🎯 Supplier Recommendations":
        show_recommendation_page()
    elif page == "🌟 PPx SVD Recommender":
        show_ppx_recommender_page()
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


def show_ppx_recommender_page():
    """PPx SVD-based Recommender System Page"""
    st.markdown('<h2 class="sub-header">🌟 PPx SVD Recommender System</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(11, 122, 184, 0.1); border-left: 4px solid #0B7AB8; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;">
        <p style="margin: 0; color: #132340;">
            <strong>Enterprise-grade recommendation</strong> using <strong>SVD matrix factorization</strong> with governance thresholds and supplier index re-ranking.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize PPx recommender if not already done
    if 'ppx_recommender' not in st.session_state:
        with st.spinner("🔧 Initializing PPx SVD Recommender..."):
            from utils.ppx_recommender import PPxRecommenderSystem
            
            ppx = PPxRecommenderSystem()
            
            # Get data
            data = st.session_state.data
            
            # Prepare interactions
            pdt_df = data['purchasing_delivery']
            ppx.prepare_interactions(pdt_df)
            
            # Train SVD
            ppx.train_svd(n_factors=50, n_epochs=30)
            
            # Calculate supplier index
            ppx.calculate_supplier_index(
                data['performance'],
                data['risk'],
                data['financial'],
                data['sustainability']
            )
            
            # Filter eligible suppliers
            ppx.filter_eligible_suppliers(
                data['performance'],
                data['risk'],
                data['financial'],
                data['sustainability']
            )
            
            st.session_state.ppx_recommender = ppx
        
        st.success("✅ PPx SVD Recommender initialized successfully!")
    
    ppx = st.session_state.ppx_recommender
    
    # Configuration section
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Material selection
        materials = ppx.get_available_materials()
        if materials:
            selected_material = st.selectbox(
                "📦 Select Material Category",
                materials,
                help="Choose the material category for which you want supplier recommendations"
            )
        else:
            st.error("No materials available")
            return
    
    with col2:
        # Alpha parameter
        alpha = st.slider(
            "🎯 SVD Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Weight of SVD score vs Supplier Index. Higher = more emphasis on historical interactions"
        )
        
        # Number of recommendations
        top_n = st.slider(
            "🔢 Number of Recommendations",
            min_value=3,
            max_value=20,
            value=10,
            step=1
        )
    
    # Governance thresholds expander
    with st.expander("🛡️ Governance Thresholds (Click to customize)"):
        st.markdown("**Risk Thresholds (Maximum allowed)**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_fin_risk = st.number_input("Max Financial Risk", value=70, min_value=0, max_value=100)
        with col2:
            max_comp_risk = st.number_input("Max Compliance Risk", value=70, min_value=0, max_value=100)
        with col3:
            max_sc_risk = st.number_input("Max Supply Chain Risk", value=70, min_value=0, max_value=100)
        
        st.markdown("**Performance Thresholds (Minimum required)**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_env = st.number_input("Min Environmental", value=50, min_value=0, max_value=100)
        with col2:
            min_social = st.number_input("Min Social", value=50, min_value=0, max_value=100)
        with col3:
            min_gov = st.number_input("Min Governance", value=50, min_value=0, max_value=100)
        with col4:
            min_qual = st.number_input("Min Quality", value=60, min_value=0, max_value=100)
        
        if st.button("✅ Update Thresholds"):
            ppx.update_thresholds(
                max_financial_risk=max_fin_risk,
                max_compliance_risk=max_comp_risk,
                max_supply_chain_risk=max_sc_risk,
                min_environmental=min_env,
                min_social=min_social,
                min_governance=min_gov,
                min_quality=min_qual
            )
            # Re-filter eligible suppliers
            ppx.filter_eligible_suppliers(
                st.session_state.data['performance'],
                st.session_state.data['risk'],
                st.session_state.data['financial'],
                st.session_state.data['sustainability']
            )
            st.success("✅ Thresholds updated!")
    
    # Get recommendations button
    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("🔍 Finding best suppliers..."):
            recommendations = ppx.recommend_suppliers(
                material_name=selected_material,
                top_n=top_n,
                alpha=alpha
            )
        
        if len(recommendations) == 0:
            st.warning("⚠️ No eligible suppliers found for this material with current thresholds. Try relaxing the governance thresholds.")
        else:
            st.markdown(f"### 🏆 Top {len(recommendations)} Recommended Suppliers for: **{selected_material}**")
            
            # Display recommendations
            for idx, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="background: white; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 24px -4px rgba(19, 35, 64, 0.08); border-left: 4px solid #0B7AB8;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #132340;">#{idx + 1} - {row['Supplier']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🎯 Final Score", f"{row['Final_Score']:.1f}/100")
                    with col2:
                        st.metric("🧠 SVD Score", f"{row['SVD_Score']:.2f}/5")
                    with col3:
                        st.metric("📊 Supplier Index", f"{row['Supplier_Index']:.1f}/100")
            
            # Download button
            st.markdown("---")
            st.download_button(
                label="📥 Download Recommendations (CSV)",
                data=recommendations.to_csv(index=False),
                file_name=f"ppx_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Explanation
            st.markdown("""
            <div style="background: rgba(36, 168, 150, 0.1); border-radius: 0.5rem; padding: 1rem; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.875rem; color: #132340;">
                    <strong>💡 How it works:</strong> The PPx recommender uses <strong>SVD (Singular Value Decomposition)</strong> to learn patterns from historical purchase data, 
                    then re-ranks results using the <strong>Supplier Index</strong> (quality, delivery, risk, ESG). Only suppliers meeting governance thresholds are shown.
                </p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
