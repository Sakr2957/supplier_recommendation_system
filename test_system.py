"""
Comprehensive Test Script for Supplier Recommendation System
Run this before deploying to ensure everything works
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.ml_models import (
    SupplierIndexingModel,
    SupplierClusteringModel,
    SupplierRankingModel,
    CollaborativeFilteringModel
)

def test_data_loading():
    """Test 1: Data Loading"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        loader = DataLoader('data')
        data = loader.load_all_data()
        
        assert len(data) == 12, f"Expected 12 datasets, got {len(data)}"
        print("✓ All 12 datasets loaded successfully")
        
        categories, subcategories = loader.get_material_categories()
        print(f"✓ Found {len(categories)} categories, {len(subcategories)} subcategories")
        
        countries = loader.get_countries()
        print(f"✓ Found {len(countries)} countries")
        
        return loader, data
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None, None


def test_feature_engineering(loader):
    """Test 2: Feature Engineering"""
    print("\n" + "="*60)
    print("TEST 2: Feature Engineering")
    print("="*60)
    
    try:
        features = loader.create_ml_features()
        
        assert features.shape[0] > 0, "No features created"
        assert features.shape[1] > 10, f"Too few features: {features.shape[1]}"
        assert 'Supplier_Score' in features.columns, "Supplier_Score not found"
        
        print(f"✓ Created feature set: {features.shape[0]} suppliers, {features.shape[1]} features")
        print(f"✓ Supplier_Score range: {features['Supplier_Score'].min():.2f} - {features['Supplier_Score'].max():.2f}")
        print(f"✓ Missing values: {features.isnull().sum().sum()}")
        
        return features
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return None


def test_supplier_indexing(features):
    """Test 3: Supplier Indexing Model"""
    print("\n" + "="*60)
    print("TEST 3: Supplier Indexing Model")
    print("="*60)
    
    try:
        indexing_model = SupplierIndexingModel()
        features_with_index = indexing_model.calculate_supplier_index(features)
        
        assert 'Supplier_Index' in features_with_index.columns, "Supplier_Index not created"
        assert 'Index_Tier' in features_with_index.columns, "Index_Tier not created"
        
        tier_counts = features_with_index['Index_Tier'].value_counts()
        print(f"✓ Supplier Index calculated")
        print(f"  Tier 1: {tier_counts.get('Tier 1', 0)} suppliers")
        print(f"  Tier 2: {tier_counts.get('Tier 2', 0)} suppliers")
        print(f"  Tier 3: {tier_counts.get('Tier 3', 0)} suppliers")
        print(f"  Tier 4: {tier_counts.get('Tier 4', 0)} suppliers")
        
        return features_with_index
    except Exception as e:
        print(f"✗ Supplier indexing failed: {e}")
        return features


def test_clustering(features):
    """Test 4: K-Means Clustering"""
    print("\n" + "="*60)
    print("TEST 4: K-Means Clustering")
    print("="*60)
    
    try:
        clustering_model = SupplierClusteringModel(n_clusters=4)
        clusters, pca_data = clustering_model.fit(features)
        
        assert len(clusters) == len(features), "Cluster count mismatch"
        assert pca_data.shape[1] == 2, "PCA should have 2 components"
        
        cluster_profiles = clustering_model.get_cluster_profiles(features, clusters)
        
        print(f"✓ K-Means clustering complete")
        print(f"  Clusters: {len(set(clusters))}")
        print(f"  PCA data shape: {pca_data.shape}")
        
        for profile in cluster_profiles:
            print(f"  Cluster {profile['cluster_id'] + 1}: {profile['size']} suppliers")
        
        return clustering_model, clusters, pca_data
    except Exception as e:
        print(f"✗ Clustering failed: {e}")
        return None, None, None


def test_random_forest(features):
    """Test 5: Random Forest Model"""
    print("\n" + "="*60)
    print("TEST 5: Random Forest Model")
    print("="*60)
    
    try:
        rf_model = SupplierRankingModel(model_type='random_forest')
        train_score, test_score = rf_model.train(features, target_col='Supplier_Score')
        
        print(f"✓ Random Forest trained")
        print(f"  Train R²: {train_score:.3f}")
        print(f"  Test R²: {test_score:.3f}")
        print(f"  Features used: {len(rf_model.feature_cols)}")
        
        return rf_model
    except Exception as e:
        print(f"✗ Random Forest training failed: {e}")
        return None


def test_xgboost(features):
    """Test 6: XGBoost Model"""
    print("\n" + "="*60)
    print("TEST 6: XGBoost Model")
    print("="*60)
    
    try:
        xgb_model = SupplierRankingModel(model_type='xgboost')
        train_score, test_score = xgb_model.train(features, target_col='Supplier_Score')
        
        print(f"✓ XGBoost trained")
        print(f"  Train R²: {train_score:.3f}")
        print(f"  Test R²: {test_score:.3f}")
        print(f"  Features used: {len(xgb_model.feature_cols)}")
        
        return xgb_model
    except Exception as e:
        print(f"✗ XGBoost training failed: {e}")
        return None


def test_collaborative_filtering(loader):
    """Test 7: Collaborative Filtering"""
    print("\n" + "="*60)
    print("TEST 7: Collaborative Filtering")
    print("="*60)
    
    try:
        interactions = loader.prepare_recommendation_data()
        
        assert len(interactions) > 0, "No interactions data"
        
        cf_model = CollaborativeFilteringModel()
        cf_model.fit(interactions)
        
        print(f"✓ Collaborative Filtering trained")
        print(f"  Interactions: {len(interactions)}")
        print(f"  Suppliers: {len(cf_model.interaction_matrix)}")
        print(f"  Materials: {len(cf_model.interaction_matrix.columns)}")
        
        # Test recommendation
        if len(cf_model.interaction_matrix.columns) > 0:
            test_material = cf_model.interaction_matrix.columns[0]
            recommendations = cf_model.recommend_suppliers(test_material, n_recommendations=3)
            print(f"  Sample recommendations for '{test_material}': {len(recommendations)} suppliers")
        
        return cf_model
    except Exception as e:
        print(f"✗ Collaborative Filtering failed: {e}")
        return None


def test_recommendations(features):
    """Test 8: End-to-End Recommendations"""
    print("\n" + "="*60)
    print("TEST 8: End-to-End Recommendations")
    print("="*60)
    
    try:
        # Filter by score
        top_suppliers = features.nlargest(5, 'Supplier_Score')
        
        print(f"✓ Top 5 Suppliers by Score:")
        for idx, (_, supplier) in enumerate(top_suppliers.iterrows(), 1):
            print(f"  {idx}. {supplier['Supplier']} - Score: {supplier['Supplier_Score']:.1f}")
        
        # Filter by country
        if 'Country' in features.columns:
            top_country = features['Country'].value_counts().index[0]
            country_suppliers = features[features['Country'] == top_country].nlargest(3, 'Supplier_Score')
            print(f"\n✓ Top 3 Suppliers in {top_country}:")
            for idx, (_, supplier) in enumerate(country_suppliers.iterrows(), 1):
                print(f"  {idx}. {supplier['Supplier']} - Score: {supplier['Supplier_Score']:.1f}")
        
        return True
    except Exception as e:
        print(f"✗ Recommendations test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SUPPLIER RECOMMENDATION SYSTEM - COMPREHENSIVE TEST")
    print("="*60)
    
    results = {
        'data_loading': False,
        'feature_engineering': False,
        'supplier_indexing': False,
        'clustering': False,
        'random_forest': False,
        'xgboost': False,
        'collaborative_filtering': False,
        'recommendations': False
    }
    
    # Test 1: Data Loading
    loader, data = test_data_loading()
    results['data_loading'] = loader is not None
    
    if not loader:
        print("\n❌ Data loading failed. Cannot proceed with other tests.")
        return results
    
    # Test 2: Feature Engineering
    features = test_feature_engineering(loader)
    results['feature_engineering'] = features is not None
    
    if features is None:
        print("\n❌ Feature engineering failed. Cannot proceed with ML tests.")
        return results
    
    # Test 3: Supplier Indexing
    features_with_index = test_supplier_indexing(features)
    results['supplier_indexing'] = 'Supplier_Index' in features_with_index.columns
    
    # Test 4: Clustering
    clustering_model, clusters, pca_data = test_clustering(features_with_index)
    results['clustering'] = clustering_model is not None
    
    # Test 5: Random Forest
    rf_model = test_random_forest(features_with_index)
    results['random_forest'] = rf_model is not None
    
    # Test 6: XGBoost
    xgb_model = test_xgboost(features_with_index)
    results['xgboost'] = xgb_model is not None
    
    # Test 7: Collaborative Filtering
    cf_model = test_collaborative_filtering(loader)
    results['collaborative_filtering'] = cf_model is not None
    
    # Test 8: Recommendations
    results['recommendations'] = test_recommendations(features_with_index)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print("\n" + "="*60)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for deployment.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
