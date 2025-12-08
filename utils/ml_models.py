"""
Machine Learning Models for Supplier Recommendation
Includes: Random Forest, XGBoost, K-Means Clustering, and Collaborative Filtering
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SupplierIndexingModel:
    """
    Phase 2: Supplier Indexing Using Machine Learning
    Implements the 5-step indexing process
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.weights = {
            'spend': 0.25,
            'quality': 0.20,
            'delivery': 0.20,
            'risk': 0.15,
            'sustainability': 0.10,
            'network': 0.10
        }
    
    def calculate_supplier_index(self, features_df):
        """
        5-Step Indexing Process:
        1. Obtain Raw Metrics
        2. Metric Normalization
        3. Node-level Aggregation
        4. Non-linear Transformation
        5. Index Normalization
        """
        print("Calculating Supplier Index Scores...")
        
        df = features_df.copy()
        
        # Step 1: Obtain Raw Metrics (already in features_df)
        metrics = {
            'spend': ['Total_Spend', 'Avg_PO_Value', 'PO_Count'],
            'quality': ['Quality score'],
            'delivery': ['Delivery score', 'OnTime_Delivery_Rate'],
            'risk': ['Financial Risk', 'Compliance risk', 'Supply Chain Risk'],
            'sustainability': ['Environmental score', 'Social Score', 'Governance score'],
            'network': []  # Will calculate uniqueness and centrality
        }
        
        # Step 2: Metric Normalization (0-100 scale)
        normalized_scores = {}
        
        for category, cols in metrics.items():
            if not cols:
                continue
            
            category_score = pd.Series(0, index=df.index)
            
            for col in cols:
                if col in df.columns:
                    col_data = df[col].fillna(df[col].median())
                    
                    # Normalize to 0-100
                    if category == 'risk':  # Lower is better for risk
                        normalized = 100 - ((col_data - col_data.min()) / (col_data.max() - col_data.min() + 0.001) * 100)
                    else:
                        normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 0.001) * 100
                    
                    category_score += normalized / len(cols)
            
            normalized_scores[category] = category_score
        
        # Step 3: Node-level Aggregation (weighted average)
        index_score = pd.Series(0, index=df.index)
        
        for category, weight in self.weights.items():
            if category in normalized_scores:
                index_score += weight * normalized_scores[category]
        
        # Step 4: Non-linear Transformation (square root to reduce extremes)
        transformed_score = np.sqrt(index_score) * 10
        
        # Step 5: Index Normalization (final 0-100 scale)
        final_index = (transformed_score - transformed_score.min()) / (transformed_score.max() - transformed_score.min() + 0.001) * 100
        
        # Assign tiers
        tiers = pd.cut(final_index, bins=[0, 25, 50, 75, 100], labels=['Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'])
        
        df['Supplier_Index'] = final_index
        df['Index_Tier'] = tiers
        
        print(f"✓ Calculated index for {len(df)} suppliers")
        print(f"  Tier 1 (75-100): {(tiers == 'Tier 1').sum()} suppliers")
        print(f"  Tier 2 (50-75): {(tiers == 'Tier 2').sum()} suppliers")
        print(f"  Tier 3 (25-50): {(tiers == 'Tier 3').sum()} suppliers")
        print(f"  Tier 4 (0-25): {(tiers == 'Tier 4').sum()} suppliers")
        
        return df


class SupplierClusteringModel:
    """
    K-Means Clustering for Supplier Segmentation
    """
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.feature_cols = None
    
    def fit(self, features_df):
        """Train K-Means clustering model"""
        print(f"Training K-Means clustering (k={self.n_clusters})...")
        
        # Select numeric features for clustering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col not in ['SAP ID']]
        
        X = features_df[self.feature_cols].fillna(features_df[self.feature_cols].median())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Calculate PCA for visualization
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate cluster quality metrics
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        
        print(f"✓ Clustering complete")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        return clusters, X_pca
    
    def predict(self, features_df):
        """Predict cluster for new suppliers"""
        X = features_df[self.feature_cols].fillna(features_df[self.feature_cols].median())
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_profiles(self, features_df, clusters):
        """Generate cluster profiles with characteristics"""
        df = features_df.copy()
        df['Cluster'] = clusters
        
        profiles = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_spend': cluster_data['Total_Spend'].mean() if 'Total_Spend' in cluster_data.columns else 0,
                'avg_quality': cluster_data['Quality score'].mean() if 'Quality score' in cluster_data.columns else 0,
                'avg_delivery': cluster_data['Delivery score'].mean() if 'Delivery score' in cluster_data.columns else 0,
                'top_countries': cluster_data['Country'].value_counts().head(3).to_dict() if 'Country' in cluster_data.columns else {},
                'avg_index': cluster_data['Supplier_Index'].mean() if 'Supplier_Index' in cluster_data.columns else 0
            }
            
            profiles.append(profile)
        
        return profiles


class SupplierRankingModel:
    """
    Random Forest and XGBoost models for supplier ranking
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_cols = None
        self.scaler = StandardScaler()
    
    def train(self, features_df, target_col='Supplier_Score'):
        """Train ranking model"""
        print(f"Training {self.model_type} model...")
        
        # Prepare features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col not in ['SAP ID', target_col, 'Supplier_Index']]
        
        X = features_df[self.feature_cols].fillna(features_df[self.feature_cols].median())
        y = features_df[target_col].fillna(features_df[target_col].median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✓ Model trained")
        print(f"  Train R²: {train_score:.3f}")
        print(f"  Test R²: {test_score:.3f}")
        
        return train_score, test_score
    
    def predict(self, features_df):
        """Predict supplier scores"""
        X = features_df[self.feature_cols].fillna(features_df[self.feature_cols].median())
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance rankings"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None


class CollaborativeFilteringModel:
    """
    Simplified collaborative filtering for supplier recommendations
    (Alternative to SVD++ which requires surprise library)
    """
    
    def __init__(self):
        self.supplier_similarity = None
        self.material_similarity = None
        self.interaction_matrix = None
    
    def fit(self, interactions_df):
        """
        Train collaborative filtering model
        interactions_df should have: Supplier, Material_Category, Rating
        """
        print("Training Collaborative Filtering model...")
        
        # Create user-item matrix
        self.interaction_matrix = interactions_df.pivot_table(
            index='Supplier',
            columns='Material_Category',
            values='Rating',
            fill_value=0
        )
        
        # Calculate supplier similarity (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        
        self.supplier_similarity = pd.DataFrame(
            cosine_similarity(self.interaction_matrix),
            index=self.interaction_matrix.index,
            columns=self.interaction_matrix.index
        )
        
        # Calculate material similarity
        self.material_similarity = pd.DataFrame(
            cosine_similarity(self.interaction_matrix.T),
            index=self.interaction_matrix.columns,
            columns=self.interaction_matrix.columns
        )
        
        print(f"✓ Model trained on {len(self.interaction_matrix)} suppliers and {len(self.interaction_matrix.columns)} materials")
        
        return self
    
    def recommend_suppliers(self, material_category, n_recommendations=3, min_score=0):
        """Recommend top suppliers for a given material category"""
        if material_category not in self.interaction_matrix.columns:
            # Return suppliers with highest overall ratings
            avg_ratings = self.interaction_matrix.mean(axis=1).sort_values(ascending=False)
            return avg_ratings.head(n_recommendations).index.tolist()
        
        # Get suppliers who have worked with this material
        material_ratings = self.interaction_matrix[material_category]
        
        # Sort by rating
        top_suppliers = material_ratings[material_ratings > min_score].sort_values(ascending=False)
        
        return top_suppliers.head(n_recommendations).index.tolist()
    
    def recommend_similar_suppliers(self, supplier_name, n_recommendations=3):
        """Find similar suppliers based on their material portfolio"""
        if supplier_name not in self.supplier_similarity.index:
            return []
        
        similar = self.supplier_similarity[supplier_name].sort_values(ascending=False)[1:n_recommendations+1]
        return similar.index.tolist()


def save_models(models_dict, save_dir='models'):
    """Save trained models to disk"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    for name, model in models_dict.items():
        with open(save_path / f'{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    print(f"✓ Saved {len(models_dict)} models to {save_dir}/")


def load_models(model_names, load_dir='models'):
    """Load trained models from disk"""
    load_path = Path(load_dir)
    models = {}
    
    for name in model_names:
        model_file = load_path / f'{name}.pkl'
        if model_file.exists():
            with open(model_file, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models
