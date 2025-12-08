# Smart Supplier Recommendation System - Project Summary

## Project Overview

A professional, AI-powered procurement intelligence platform built with Streamlit, featuring advanced machine learning algorithms for supplier recommendations, clustering analysis, and comprehensive analytics.

## Key Features

### Multi-Algorithm ML Pipeline
- Random Forest Regressor (R² = 0.952)
- XGBoost Regressor (R² = 0.971)
- K-Means Clustering (4 clusters)
- Collaborative Filtering (539 suppliers × 73 materials)
- 5-Step Supplier Indexing Methodology

### Advanced Filtering System
- Number of recommendations (1-20)
- Minimum supplier score cutoff (0-100)
- Country filter (31 countries)
- Material category filter (4 categories)
- Material subcategory filter (62 subcategories)

### Modern Interactive Visualizations
- PCA-based cluster scatter plots
- Performance matrix (Quality vs Delivery)
- Feature importance charts
- Risk assessment dashboards
- ESG sustainability scores

## Test Results

All 8 tests passed (100%):
- Data loading
- Feature engineering
- Supplier indexing
- K-Means clustering
- Random Forest training
- XGBoost training
- Collaborative filtering
- End-to-end recommendations

## Quick Start

Linux/Mac: `./run_app.sh`
Windows: `run_app.bat`

Or manually:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Documentation

- README.md - Comprehensive user guide
- DEPLOYMENT_GUIDE.md - Deployment instructions
- GITHUB_SETUP.md - GitHub setup guide

Ready for deployment!
