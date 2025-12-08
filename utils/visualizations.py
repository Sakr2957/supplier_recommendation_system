"""
Modern Visualization Utilities using Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_cluster_scatter(pca_data, clusters, cluster_profiles):
    """
    Create beautiful PCA scatter plot with cluster annotations
    Similar to the image provided
    """
    df = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        'Cluster': [f'Cluster {c+1}' for c in clusters]
    })
    
    # Define colors for clusters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',
        color_discrete_sequence=colors,
        title='Supplier Segmentation - PCA Visualization',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        template='plotly_white',
        width=1000,
        height=600
    )
    
    # Add cluster annotations
    for i, profile in enumerate(cluster_profiles):
        cluster_points = df[df['Cluster'] == f'Cluster {i+1}']
        center_x = cluster_points['PC1'].mean()
        center_y = cluster_points['PC2'].mean()
        
        # Create annotation text
        annotation_text = f"<b>Cluster {i+1}</b><br>"
        annotation_text += f"Size: {profile['size']} suppliers<br>"
        annotation_text += f"Avg Score: {profile['avg_index']:.1f}"
        
        fig.add_annotation(
            x=center_x,
            y=center_y,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=colors[i],
            bgcolor='white',
            bordercolor=colors[i],
            borderwidth=2,
            font=dict(size=10)
        )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        font=dict(size=12),
        title_font=dict(size=18, family='Arial Black'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def plot_cluster_profiles(cluster_profiles):
    """Create radar chart showing cluster characteristics"""
    categories = ['Avg Spend', 'Quality', 'Delivery', 'Supplier Index']
    
    fig = go.Figure()
    
    for profile in cluster_profiles:
        # Normalize values to 0-100 scale for radar chart
        values = [
            min(profile['avg_spend'] / 1000000 * 10, 100),  # Spend in millions
            profile['avg_quality'],
            profile['avg_delivery'],
            profile['avg_index']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f"Cluster {profile['cluster_id'] + 1}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='Cluster Profiles Comparison',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_feature_importance(importance_df, top_n=15):
    """Plot feature importance from ML models"""
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importance',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        template='plotly_white',
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=False
    )
    
    return fig


def plot_supplier_distribution(features_df):
    """Plot supplier distribution by country and tier"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Suppliers by Country (Top 10)', 'Suppliers by Index Tier'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Country distribution
    country_counts = features_df['Country'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=country_counts.index, y=country_counts.values, name='Country', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    # Tier distribution
    if 'Index_Tier' in features_df.columns:
        tier_counts = features_df['Index_Tier'].value_counts()
        fig.add_trace(
            go.Pie(labels=tier_counts.index, values=tier_counts.values, name='Tier'),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text='Supplier Distribution Analysis',
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_supplier_scores_distribution(features_df):
    """Plot distribution of supplier scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=features_df['Supplier_Score'],
        nbinsx=30,
        name='Supplier Score',
        marker_color='#2ca02c',
        opacity=0.7
    ))
    
    if 'Supplier_Index' in features_df.columns:
        fig.add_trace(go.Histogram(
            x=features_df['Supplier_Index'],
            nbinsx=30,
            name='Supplier Index',
            marker_color='#ff7f0e',
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Distribution of Supplier Scores',
        xaxis_title='Score',
        yaxis_title='Number of Suppliers',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_top_suppliers(features_df, n=10, score_col='Supplier_Score'):
    """Plot top N suppliers by score"""
    top_suppliers = features_df.nlargest(n, score_col)[['Supplier', score_col, 'Country']].copy()
    
    fig = px.bar(
        top_suppliers,
        x=score_col,
        y='Supplier',
        orientation='h',
        title=f'Top {n} Suppliers by Score',
        labels={score_col: 'Score', 'Supplier': 'Supplier Name'},
        color='Country',
        template='plotly_white',
        height=500
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True
    )
    
    return fig


def plot_performance_matrix(features_df):
    """Create performance matrix (Quality vs Delivery)"""
    fig = px.scatter(
        features_df,
        x='Quality score',
        y='Delivery score',
        size='Total_Spend',
        color='Supplier_Score',
        hover_data=['Supplier', 'Country'],
        title='Supplier Performance Matrix',
        labels={
            'Quality score': 'Quality Score',
            'Delivery score': 'Delivery Score',
            'Supplier_Score': 'Overall Score'
        },
        template='plotly_white',
        color_continuous_scale='RdYlGn',
        height=600
    )
    
    # Add quadrant lines
    median_quality = features_df['Quality score'].median()
    median_delivery = features_df['Delivery score'].median()
    
    fig.add_hline(y=median_delivery, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_quality, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=90, y=90, text="Strategic", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=90, y=70, text="Leverage", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=70, y=90, text="Critical", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=70, y=70, text="Tail", showarrow=False, font=dict(size=14, color="red"))
    
    return fig


def plot_risk_assessment(features_df):
    """Plot risk assessment across dimensions"""
    risk_cols = ['Financial Risk', 'Compliance risk', 'Supply Chain Risk', 'Geopolitical Risk']
    available_risks = [col for col in risk_cols if col in features_df.columns]
    
    if not available_risks:
        return None
    
    risk_data = features_df[available_risks].mean().reset_index()
    risk_data.columns = ['Risk Type', 'Average Score']
    
    fig = px.bar(
        risk_data,
        x='Risk Type',
        y='Average Score',
        title='Average Risk Scores Across Suppliers',
        labels={'Average Score': 'Average Risk Score (Lower is Better)'},
        template='plotly_white',
        color='Average Score',
        color_continuous_scale='Reds_r',
        height=400
    )
    
    return fig


def plot_sustainability_scores(features_df):
    """Plot ESG sustainability scores"""
    esg_cols = ['Environmental score', 'Social Score', 'Governance score']
    available_esg = [col for col in esg_cols if col in features_df.columns]
    
    if not available_esg:
        return None
    
    esg_data = features_df[available_esg].mean().reset_index()
    esg_data.columns = ['ESG Dimension', 'Average Score']
    
    fig = px.bar(
        esg_data,
        x='ESG Dimension',
        y='Average Score',
        title='Average ESG Scores Across Suppliers',
        labels={'Average Score': 'Average ESG Score'},
        template='plotly_white',
        color='Average Score',
        color_continuous_scale='Greens',
        height=400
    )
    
    return fig


def create_supplier_card(supplier_data):
    """Create a detailed supplier information card"""
    card_html = f"""
    <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f8f9fa;">
        <h3 style="color: #1f77b4; margin-top: 0;">{supplier_data['Supplier']}</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div><strong>SAP ID:</strong> {supplier_data.get('SAP ID', 'N/A')}</div>
            <div><strong>Country:</strong> {supplier_data.get('Country', 'N/A')}</div>
            <div><strong>Supplier Score:</strong> {supplier_data.get('Supplier_Score', 0):.1f}</div>
            <div><strong>Index Tier:</strong> {supplier_data.get('Index_Tier', 'N/A')}</div>
            <div><strong>Quality Score:</strong> {supplier_data.get('Quality score', 0):.1f}</div>
            <div><strong>Delivery Score:</strong> {supplier_data.get('Delivery score', 0):.1f}</div>
        </div>
    </div>
    """
    return card_html
