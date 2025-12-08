# 🏭 Smart Supplier Recommendation System

An AI-powered procurement intelligence platform that leverages machine learning to provide intelligent supplier recommendations, clustering analysis, and comprehensive supplier analytics.

## 🌟 Features

### Phase 1: Procurement Dashboard
- Real-time supplier performance metrics
- Interactive visualizations with Plotly
- Comprehensive analytics across multiple dimensions

### Phase 2: Supplier Indexing
- **5-Step ML Indexing Process:**
  1. Obtain Raw Metrics (Spend, Performance, Network)
  2. Metric Normalization (0-100 scale)
  3. Node-level Aggregation (weighted average)
  4. Non-linear Transformation (square root)
  5. Index Normalization (final 0-100 scale)
- Supplier tier classification (Tier 1-4)

### Phase 3: Supplier Recommendation
- **Multiple ML Algorithms:**
  - Random Forest Regressor
  - XGBoost Regressor
  - K-Means Clustering (4 clusters)
  - Collaborative Filtering
- **Advanced Filtering:**
  - Number of recommendations (1-20)
  - Supplier score cutoff (0-100)
  - Country filter
  - Material category filter
  - Material subcategory filter

### Visualizations
- PCA-based cluster scatter plots
- Performance matrix (Quality vs Delivery)
- Feature importance charts
- Risk assessment dashboards
- ESG sustainability scores
- Interactive supplier cards

## 📊 Data Sources

The system integrates data from 12 Excel files:
- `SupplierList.xlsx` - Master supplier data (1,254 suppliers)
- `SupplierFinancialHealth.xlsx` - Financial metrics and D&B ratings
- `SupplierPerformance.xlsx` - Quality, delivery, communication scores
- `SupplierRisk.xlsx` - Financial, compliance, supply chain risks
- `SupplierSustainbility.xlsx` - ESG scores
- `PurchasingDeliveryTool.xlsx` - Procurement transactions (2,345 records)
- `SupplierDeliveryStatus.xlsx` - Delivery performance (5,774 records)
- `SupplierCommercialData.xlsx` - Payment terms
- `MaterialCategory.xlsx` - Material hierarchy (62 categories)
- `CountryRisk.xlsx` - Geopolitical risk data
- `AgreementList.xlsx` - Supplier agreements
- `EnitityInformation.xlsx` - Plant locations

## 🚀 Quick Start

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/supplier-recommendation-system.git
cd supplier-recommendation-system
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Place data files:**
- Copy all Excel files to the `data/` directory

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Open in browser:**
- The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
supplier-recommendation-system/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore file
├── data/                           # Data directory (Excel files)
│   ├── SupplierList.xlsx
│   ├── SupplierFinancialHealth.xlsx
│   ├── SupplierPerformance.xlsx
│   ├── SupplierRisk.xlsx
│   ├── SupplierSustainbility.xlsx
│   ├── PurchasingDeliveryTool.xlsx
│   ├── SupplierDeliveryStatus.xlsx
│   ├── SupplierCommercialData.xlsx
│   ├── MaterialCategory.xlsx
│   ├── CountryRisk.xlsx
│   ├── AgreementList.xlsx
│   └── EnitityInformation.xlsx
├── utils/                          # Utility modules
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── ml_models.py               # ML models (RF, XGBoost, K-Means, CF)
│   └── visualizations.py          # Plotly visualization functions
├── models/                         # Saved ML models (generated)
└── assets/                         # Static assets (images, etc.)
```

## 🔧 Configuration

### Adjusting ML Parameters

Edit `utils/ml_models.py` to customize:

**Supplier Indexing Weights:**
```python
self.weights = {
    'spend': 0.25,
    'quality': 0.20,
    'delivery': 0.20,
    'risk': 0.15,
    'sustainability': 0.10,
    'network': 0.10
}
```

**Clustering Parameters:**
```python
clustering_model = SupplierClusteringModel(n_clusters=4)  # Change number of clusters
```

**Random Forest Parameters:**
```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

## 🌐 GitHub Deployment Guide

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon → "New repository"
3. Name: `supplier-recommendation-system`
4. Description: "AI-powered supplier recommendation system with ML clustering and analytics"
5. Choose "Public" or "Private"
6. Click "Create repository"

### Step 2: Initialize Git and Push

```bash
# Navigate to your project directory
cd /path/to/supplier-recommendation-system

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Smart Supplier Recommendation System"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/supplier-recommendation-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Select your repository:** `YOUR_USERNAME/supplier-recommendation-system`
5. **Set main file path:** `app.py`
6. **Click "Deploy"**

The app will be live at: `https://YOUR_USERNAME-supplier-recommendation-system.streamlit.app`

### Step 4: Handle Data Files (Important!)

⚠️ **Do NOT commit large Excel files to GitHub!**

**Option A: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track Excel files
git lfs track "data/*.xlsx"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS for data files"
git push
```

**Option B: Use Streamlit Secrets (Recommended for sensitive data)**
1. Upload data files to a cloud storage (Google Drive, Dropbox, AWS S3)
2. In Streamlit Cloud, go to App Settings → Secrets
3. Add file URLs:
```toml
[data_urls]
supplier_list = "https://your-cloud-storage/SupplierList.xlsx"
financial_health = "https://your-cloud-storage/SupplierFinancialHealth.xlsx"
# ... add all file URLs
```

4. Modify `data_loader.py` to download from URLs:
```python
import requests

def download_data_from_url(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
```

**Option C: Use .gitignore (Local development only)**
```bash
# Add to .gitignore
echo "data/*.xlsx" >> .gitignore
git add .gitignore
git commit -m "Ignore data files"
```

### Step 5: Environment Variables

If you need API keys or secrets:

1. In Streamlit Cloud → App Settings → Secrets
2. Add in TOML format:
```toml
[api_keys]
dnb_api_key = "your_api_key_here"

[database]
connection_string = "your_connection_string"
```

3. Access in code:
```python
import streamlit as st
api_key = st.secrets["api_keys"]["dnb_api_key"]
```

## 📚 Usage Guide

### 1. Home & Overview
- View system-wide metrics
- Explore supplier distribution by country and tier
- Analyze top-performing suppliers
- Review performance matrix

### 2. Supplier Recommendations
- Set number of recommendations (1-20)
- Apply minimum score threshold
- Filter by country
- Filter by material category/subcategory
- Get instant recommendations with detailed profiles
- Export results to CSV

### 3. Supplier Analytics
- **Performance Tab:** Quality and delivery analysis
- **Risk Assessment Tab:** Financial, compliance, and supply chain risks
- **Sustainability Tab:** ESG scores and rankings

### 4. ML Model Insights
- View feature importance from Random Forest/XGBoost
- Understand which factors drive supplier scores
- Compare model performance

### 5. Clustering Analysis
- Visualize supplier segmentation (PCA plot)
- Explore cluster profiles
- Identify supplier groups by characteristics

## 🧪 Testing

Run the data loader test:
```bash
cd utils
python data_loader.py
```

Run the ML models test:
```bash
cd utils
python ml_models.py
```

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Ensure virtual environment is activated and dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: data/SupplierList.xlsx"
**Solution:** Ensure all Excel files are in the `data/` directory

### Issue: "Memory Error during model training"
**Solution:** Reduce dataset size or use sampling:
```python
features_df = features_df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

### Issue: Slow performance
**Solution:** Enable Streamlit caching (already implemented) and reduce data refresh frequency

## 📈 Performance Optimization

- **Caching:** All data loading and model training is cached with `@st.cache_data` and `@st.cache_resource`
- **Lazy Loading:** Models are trained only once on first load
- **Efficient Filtering:** Pandas vectorized operations for fast filtering
- **Plotly:** Hardware-accelerated interactive charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- Veolia Water Technology Solutions for the use case
- Accenture for project sponsorship
- Streamlit for the amazing framework
- Scikit-learn, XGBoost, and Plotly communities

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Built with ❤️ using Streamlit, Python, and Machine Learning**
