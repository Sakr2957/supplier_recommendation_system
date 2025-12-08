"""
Data Loading and Preprocessing Utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and preprocess all supplier data files"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data = {}
        
    def load_all_data(self):
        """Load all Excel files into memory"""
        print("Loading data files...")
        
        # Load supplier master data
        self.data['suppliers'] = pd.read_excel(self.data_dir / 'SupplierList.xlsx')
        self.data['financial'] = pd.read_excel(self.data_dir / 'SupplierFinancialHealth.xlsx')
        self.data['performance'] = pd.read_excel(self.data_dir / 'SupplierPerformance.xlsx')
        self.data['risk'] = pd.read_excel(self.data_dir / 'SupplierRisk.xlsx')
        self.data['sustainability'] = pd.read_excel(self.data_dir / 'SupplierSustainbility.xlsx')
        
        # Load transactional data
        self.data['pdt'] = pd.read_excel(self.data_dir / 'PurchasingDeliveryTool.xlsx')
        self.data['delivery'] = pd.read_excel(self.data_dir / 'SupplierDeliveryStatus.xlsx')
        self.data['commercial'] = pd.read_excel(self.data_dir / 'SupplierCommercialData.xlsx')
        
        # Load reference data
        self.data['materials'] = pd.read_excel(self.data_dir / 'MaterialCategory.xlsx')
        self.data['country_risk'] = pd.read_excel(self.data_dir / 'CountryRisk.xlsx')
        self.data['agreements'] = pd.read_excel(self.data_dir / 'AgreementList.xlsx')
        self.data['entities'] = pd.read_excel(self.data_dir / 'EnitityInformation.xlsx')
        
        print(f"✓ Loaded {len(self.data)} datasets")
        return self.data
    
    def get_material_categories(self):
        """Get unique material categories and subcategories"""
        df = self.data['materials']
        # Handle different possible column structures
        if df.shape[1] == 2:
            df.columns = ['Category', 'Subcategory']
        
        categories = df['Category'].unique().tolist()
        subcategories = df['Subcategory'].unique().tolist()
        
        return categories, subcategories
    
    def get_countries(self):
        """Get unique countries from supplier list"""
        return sorted(self.data['suppliers']['Country'].unique().tolist())
    
    def get_suppliers_by_filters(self, country=None, category=None, subcategory=None):
        """Filter suppliers based on criteria"""
        df = self.data['suppliers'].copy()
        
        if country and country != 'All':
            df = df[df['Country'] == country]
        
        # Add material category filtering if needed
        # This requires joining with PDT data
        
        return df
    
    def create_ml_features(self):
        """Create comprehensive feature set for ML models"""
        print("Creating ML feature set...")
        
        # Start with supplier master
        features = self.data['suppliers'].copy()
        
        # Merge financial health
        features = features.merge(
            self.data['financial'],
            on=['SAP ID', 'Supplier', 'Country'],
            how='left'
        )
        
        # Aggregate performance scores (average across evaluation periods)
        perf_cols = {'Quality score': 'mean', 'Delivery score': 'mean'}
        if 'Communication Score' in self.data['performance'].columns:
            perf_cols['Communication Score'] = 'mean'
        if 'Overall Score' in self.data['performance'].columns:
            perf_cols['Overall Score'] = 'mean'
        
        perf_agg = self.data['performance'].groupby(['SAP ID', 'Supplier']).agg(perf_cols).reset_index()
        
        features = features.merge(perf_agg, on=['SAP ID', 'Supplier'], how='left')
        
        # Merge risk scores
        features = features.merge(
            self.data['risk'],
            on=['SAP ID', 'Supplier'],
            how='left'
        )
        
        # Merge sustainability scores
        features = features.merge(
            self.data['sustainability'],
            on=['SAP ID', 'Supplier'],
            how='left'
        )
        
        # Calculate spend metrics from PDT
        spend_metrics = self.data['pdt'].groupby('Supplier').agg({
            'PO Amount': ['sum', 'mean', 'count']
        }).reset_index()
        spend_metrics.columns = ['Supplier', 'Total_Spend', 'Avg_PO_Value', 'PO_Count']
        
        features = features.merge(spend_metrics, on='Supplier', how='left')
        
        # Calculate delivery performance
        delivery_df = self.data['delivery'].copy()
        
        # Calculate on-time delivery rate from status
        if 'Delivery Status_Text' in delivery_df.columns:
            delivery_metrics = delivery_df.groupby('Supplier Name').agg({
                'Delivery Status_Text': lambda x: (x == 'On Time').sum() / len(x) * 100 if len(x) > 0 else 0,
                'PO ID': 'count'
            }).reset_index()
            delivery_metrics.columns = ['Supplier', 'OnTime_Delivery_Rate', 'Delivery_Count']
        else:
            delivery_metrics = delivery_df.groupby('Supplier Name').size().reset_index(name='Delivery_Count')
            delivery_metrics['OnTime_Delivery_Rate'] = 75.0  # Default
            delivery_metrics.columns = ['Supplier', 'Delivery_Count', 'OnTime_Delivery_Rate']
        
        features = features.merge(delivery_metrics, on='Supplier', how='left')
        
        # Fill missing values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
        
        # Create composite supplier score
        features['Supplier_Score'] = self._calculate_composite_score(features)
        
        print(f"✓ Created feature set with {features.shape[0]} suppliers and {features.shape[1]} features")
        
        return features
    
    def _calculate_composite_score(self, df):
        """Calculate weighted composite supplier score"""
        weights = {
            'Quality score': 0.25,
            'Delivery score': 0.25,
            'Communication Score': 0.10,
            'Overall Score': 0.10,
            'Financial Risk': -0.10,  # Lower risk is better
            'Compliance risk': -0.05,
            'Supply Chain Risk': -0.05,
            'Environmental score': 0.10,
            'Social Score': 0.05,
            'Governance score': 0.05
        }
        
        score = pd.Series(0, index=df.index)
        
        for col, weight in weights.items():
            if col in df.columns:
                # Normalize to 0-100 scale
                col_data = df[col].fillna(df[col].median())
                if weight < 0:  # For risk metrics, invert
                    col_normalized = 100 - ((col_data - col_data.min()) / (col_data.max() - col_data.min() + 0.001) * 100)
                else:
                    col_normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 0.001) * 100
                
                score += weight * col_normalized
        
        # Normalize final score to 0-100
        score = (score - score.min()) / (score.max() - score.min() + 0.001) * 100
        
        return score
    
    def prepare_recommendation_data(self):
        """Prepare data for recommendation system (user-item matrix)"""
        print("Preparing recommendation data...")
        
        # Create supplier-material interaction matrix from PDT
        pdt = self.data['pdt'].copy()
        
        # Use Category Family and Sub Category from PDT
        if 'Category Family' in pdt.columns and 'Sub Category' in pdt.columns:
            interactions = pdt.groupby(['Supplier', 'Category Family', 'Sub Category']).agg({
                'PO Amount': 'sum',
                'Po Number': 'count'
            }).reset_index()
            interactions.columns = ['Supplier', 'Category', 'Subcategory', 'Total_Value', 'Interaction_Count']
            # Combine category and subcategory
            interactions['Material_Category'] = interactions['Category'] + ' - ' + interactions['Subcategory']
            interactions = interactions[['Supplier', 'Material_Category', 'Total_Value', 'Interaction_Count']]
        else:
            # Fallback: use PP Description as proxy
            interactions = pdt.groupby(['Supplier', 'PP Description']).agg({
                'PO Amount': 'sum',
                'Po Number': 'count'
            }).reset_index()
            interactions.columns = ['Supplier', 'Material_Category', 'Total_Value', 'Interaction_Count']
        
        # Create rating based on value and count
        interactions['Rating'] = (
            0.6 * (interactions['Total_Value'] / interactions['Total_Value'].max() * 5) +
            0.4 * (interactions['Interaction_Count'] / interactions['Interaction_Count'].max() * 5)
        )
        
        print(f"✓ Created {len(interactions)} supplier-material interactions")
        
        return interactions


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    data = loader.load_all_data()
    features = loader.create_ml_features()
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"\nSample data:\n{features.head()}")
