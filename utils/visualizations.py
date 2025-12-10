"""
Modern Visualization Utilities using Plotly with Horizon UI Design
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Horizon UI Color Palette
COLORS = {
    'primary': '#4318FF',
    'purple': '#7551FF',
    'cyan': '#6AD2FF',
    'green': '#01B574',
    'orange': '#FFB547',
    'red': '#EE5D50',
    'text_primary': '#2B3674',
    'text_secondary': '#A3AED0',
    'background': '#F4F7FE'
}

# Chart color sequences
CHART_COLORS = ['#4318FF', '#7551FF', '#6AD2FF', '#01B574', '#FFB547', '#EE5D50']
GRADIENT_COLORS = [[0, '#868CFF'], [1, '#4318FF']]


def get_base_layout():
    """Get base layout configuration for all charts"""
    return dict(
        font=dict(family='"DM Sans", sans-serif', size=14, color=COLORS['text_primary']),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family='"DM Sans", sans-serif'
        )
    )


def plot_cluster_scatter(pca_data, clusters, cluster_profiles):
    """Create beautiful PCA scatter plot with Horizon UI colors"""
    df = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        'Cluster': [f'Cluster {c+1}' for c in clusters]
    })
    
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',
        color_discrete_sequence=CHART_COLORS,
        title='<b>Supplier Segmentation - PCA Visualization</b>',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        template='plotly_white'
    )
    
    # Add cluster annotations
    for i, profile in enumerate(cluster_profiles):
        cluster_points = df[df['Cluster'] == f'Cluster {i+1}']
        center_x = cluster_points['PC1'].mean()
        center_y = cluster_points['PC2'].mean()
        
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
            arrowcolor=CHART_COLORS[i],
            bgcolor='white',
            bordercolor=CHART_COLORS[i],
            borderwidth=2,
            font=dict(size=11, color=COLORS['text_primary'])
        )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0)))
    fig.update_layout(**get_base_layout())
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def plot_cluster_profiles(cluster_profiles):
    """Create radar chart showing cluster characteristics with Horizon UI colors"""
    categories = ['Avg Spend', 'Quality', 'Delivery', 'Supplier Index']
    
    fig = go.Figure()
    
    for i, profile in enumerate(cluster_profiles):
        values = [
            min(profile['avg_spend'] / 1000000 * 10, 100),
            profile['avg_quality'],
            profile['avg_delivery'],
            profile['avg_index']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f"Cluster {profile['cluster_id'] + 1}",
            line_color=CHART_COLORS[i],
            fillcolor=CHART_COLORS[i],
            opacity=0.6
        ))
    
    fig.update_layout(
        **get_base_layout(),
        title='<b>Cluster Characteristics Profile</b>',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#E0E5F2'
            ),
            angularaxis=dict(
                gridcolor='#E0E5F2'
            )
        )
    )
    
    return fig


def plot_supplier_distribution(features_df):
    """Create bar chart of supplier distribution by country"""
    country_counts = features_df['Country'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=country_counts.index,
            y=country_counts.values,
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['purple'], width=0)
            ),
            text=country_counts.values,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Suppliers: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title='<b>Top 10 Countries by Supplier Count</b>',
        xaxis_title='Country',
        yaxis_title='Number of Suppliers',
        showlegend=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def plot_supplier_scores_distribution(features_df):
    """Create histogram of supplier scores with Horizon UI colors"""
    fig = go.Figure(data=[
        go.Histogram(
            x=features_df['Supplier_Score'],
            nbinsx=30,
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['purple'], width=0)
            ),
            hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title='<b>Supplier Score Distribution</b>',
        xaxis_title='Supplier Score',
        yaxis_title='Frequency',
        showlegend=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def plot_top_suppliers(features_df, n=10, score_col='Supplier_Score'):
    """Create horizontal bar chart of top suppliers"""
    top_suppliers = features_df.nlargest(n, score_col)[['Supplier', score_col]]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_suppliers['Supplier'],
            x=top_suppliers[score_col],
            orientation='h',
            marker=dict(
                color=COLORS['green'],
                line=dict(color=COLORS['green'], width=0)
            ),
            text=top_suppliers[score_col].round(1),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title=f'<b>Top {n} Suppliers by {score_col.replace("_", " ")}</b>',
        xaxis_title='Score',
        yaxis_title='',
        showlegend=False,
        height=400
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    fig.update_yaxes(showgrid=False)
    
    return fig


def plot_performance_matrix(features_df):
    """Create scatter plot showing quality vs delivery performance"""
    fig = go.Figure(data=[
        go.Scatter(
            x=features_df.get('Quality score', features_df.get('Quality_Score', [0])),
            y=features_df.get('Delivery score', features_df.get('Delivery_Score', [0])),
            mode='markers',
            marker=dict(
                size=8,
                color=features_df['Supplier_Score'],
                colorscale=[[0, COLORS['red']], [0.5, COLORS['orange']], [1, COLORS['green']]],
                showscale=True,
                colorbar=dict(title="Supplier<br>Score"),
                line=dict(width=0)
            ),
            text=features_df['Supplier'],
            hovertemplate='<b>%{text}</b><br>Quality: %{x:.1f}<br>Delivery: %{y:.1f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title='<b>Quality vs Delivery Performance Matrix</b>',
        xaxis_title='Quality Score',
        yaxis_title='Delivery Score'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def plot_risk_assessment(features_df):
    """Create risk assessment visualization"""
    if 'Financial Risk' not in features_df.columns:
        return None
    
    risk_cols = ['Financial Risk', 'Compliance risk', 'Supply Chain Risk']
    available_cols = [col for col in risk_cols if col in features_df.columns]
    
    if not available_cols:
        return None
    
    avg_risks = features_df[available_cols].mean()
    
    fig = go.Figure(data=[
        go.Bar(
            x=available_cols,
            y=avg_risks.values,
            marker=dict(
                color=[COLORS['red'], COLORS['orange'], COLORS['orange']],
                line=dict(width=0)
            ),
            text=avg_risks.values.round(1),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Average Risk: %{y:.1f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title='<b>Average Risk Scores by Category</b>',
        xaxis_title='Risk Category',
        yaxis_title='Risk Score',
        showlegend=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def plot_sustainability_scores(features_df):
    """Create sustainability scores visualization"""
    esg_cols = ['Environmental score', 'Social Score', 'Governance score']
    available_cols = [col for col in esg_cols if col in features_df.columns]
    
    if not available_cols:
        return None
    
    avg_scores = features_df[available_cols].mean()
    
    fig = go.Figure(data=[
        go.Bar(
            x=available_cols,
            y=avg_scores.values,
            marker=dict(
                color=[COLORS['green'], COLORS['cyan'], COLORS['primary']],
                line=dict(width=0)
            ),
            text=avg_scores.values.round(1),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Average Score: %{y:.1f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title='<b>Average ESG Scores</b>',
        xaxis_title='ESG Category',
        yaxis_title='Score',
        showlegend=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def plot_feature_importance(importance_df, top_n=20):
    """Create feature importance chart"""
    top_features = importance_df.head(top_n).sort_values('importance')
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker=dict(
                color=COLORS['purple'],
                line=dict(width=0)
            ),
            text=top_features['importance'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title=f'<b>Top {top_n} Feature Importance</b>',
        xaxis_title='Importance',
        yaxis_title='',
        showlegend=False,
        height=max(400, top_n * 25)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    fig.update_yaxes(showgrid=False)
    
    return fig


def create_pie_chart(labels, values, title):
    """Create pie chart with Horizon UI colors"""
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=CHART_COLORS),
            hole=0.4,
            textinfo='label+percent',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title=f'<b>{title}</b>',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_line_chart(x_data, y_data, x_title, y_title, title, y_data2=None, name1="Series 1", name2="Series 2"):
    """Create line chart with Horizon UI colors and gradients"""
    fig = go.Figure()
    
    # First line with gradient fill
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name=name1,
        line=dict(color=COLORS['purple'], width=3),
        fill='tozeroy',
        fillcolor='rgba(117, 81, 255, 0.2)',
        hovertemplate='<b>%{x}</b><br>' + name1 + ': %{y:.1f}<extra></extra>'
    ))
    
    # Second line if provided
    if y_data2 is not None:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data2,
            mode='lines',
            name=name2,
            line=dict(color=COLORS['cyan'], width=3),
            fill='tozeroy',
            fillcolor='rgba(106, 210, 255, 0.2)',
            hovertemplate='<b>%{x}</b><br>' + name2 + ': %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        **get_base_layout(),
        title=f'<b>{title}</b>',
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def create_bar_chart(x_data, y_data, x_title, y_title, title):
    """Create bar chart with Horizon UI gradient colors"""
    fig = go.Figure(data=[
        go.Bar(
            x=x_data,
            y=y_data,
            marker=dict(
                color=COLORS['primary'],
                line=dict(width=0)
            ),
            text=y_data,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' + y_title + ': %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title=f'<b>{title}</b>',
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig


def create_histogram(data, x_title, y_title, title):
    """Create histogram with Horizon UI colors"""
    fig = go.Figure(data=[
        go.Histogram(
            x=data,
            nbinsx=30,
            marker=dict(
                color=COLORS['cyan'],
                line=dict(width=0)
            ),
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        **get_base_layout(),
        title=f'<b>{title}</b>',
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E5F2')
    
    return fig
