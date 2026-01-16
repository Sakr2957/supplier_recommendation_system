"""
PPx Recommender System - Lightweight Version
---------------------------------------------
Enterprise-grade supplier recommendation using collaborative filtering
with governance thresholds and supplier index re-ranking.
No external ML libraries required (pandas + numpy only).
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class PPxRecommenderSystem:
    """
    Collaborative filtering recommendation system with eligibility gates and index re-ranking
    """
    
    def __init__(self):
        self.svd_model = None
        self.supplier_index = None
        self.eligible_suppliers = None
        self.interaction_matrix = None
        self.supplier_to_idx = None
        self.material_to_idx = None
        self.idx_to_supplier = None
        self.idx_to_material = None
        self.thresholds = {
            "max_financial_risk": 70,
            "max_compliance_risk": 70,
            "max_supply_chain_risk": 70,
            "min_environmental": 50,
            "min_social": 50,
            "min_governance": 50,
            "min_quality": 60,
            "min_delivery": 60
        }
    
    def prepare_interactions(self, pdt_df):
        """
        Build supplier-material interaction matrix from purchase data
        
        Args:
            pdt_df: DataFrame with columns [Supplier, Category Family, Sub Category, PO Amount, Po Number]
        
        Returns:
            Interaction matrix
        """
        # Create material identifier
        pdt_df = pdt_df.copy()
        pdt_df["Material"] = pdt_df["Category Family"] + " - " + pdt_df["Sub Category"]
        
        # Aggregate interactions
        interactions = (
            pdt_df.groupby(["Supplier", "Material"])
            .agg(
                Total_Value=("PO Amount", "sum"),
                PO_Count=("Po Number", "count")
            )
            .reset_index()
        )
        
        # Normalize scores
        interactions["Value_Score"] = interactions["Total_Value"] / (interactions["Total_Value"].max() + 0.001)
        interactions["Freq_Score"] = interactions["PO_Count"] / (interactions["PO_Count"].max() + 0.001)
        
        # Calculate rating (0-5 scale)
        interactions["Rating"] = (
            0.6 * interactions["Value_Score"] +
            0.4 * interactions["Freq_Score"]
        ) * 5
        
        # Create mappings
        suppliers = sorted(interactions["Supplier"].unique())
        materials = sorted(interactions["Material"].unique())
        
        self.supplier_to_idx = {s: i for i, s in enumerate(suppliers)}
        self.material_to_idx = {m: i for i, m in enumerate(materials)}
        self.idx_to_supplier = {i: s for s, i in self.supplier_to_idx.items()}
        self.idx_to_material = {i: m for m, i in self.material_to_idx.items()}
        
        # Create interaction matrix
        n_suppliers = len(suppliers)
        n_materials = len(materials)
        self.interaction_matrix = np.zeros((n_suppliers, n_materials))
        
        for _, row in interactions.iterrows():
            s_idx = self.supplier_to_idx[row["Supplier"]]
            m_idx = self.material_to_idx[row["Material"]]
            self.interaction_matrix[s_idx, m_idx] = row["Rating"]
        
        return self.interaction_matrix
    
    def train_svd(self, n_components=50):
        """
        Train SVD model on supplier-material interactions using sklearn
        """
        if self.interaction_matrix is None:
            raise ValueError("Must call prepare_interactions() first")
        
        # Use TruncatedSVD from sklearn
        self.svd_model = TruncatedSVD(n_components=min(n_components, min(self.interaction_matrix.shape) - 1), random_state=42)
        self.svd_model.fit(self.interaction_matrix)
        
        return self.svd_model
    
    def calculate_supplier_index(self, performance_df, risk_df, financial_df, sustainability_df):
        """
        Calculate weighted Supplier Index (0-100)
        
        Args:
            performance_df: DataFrame with Quality score, Delivery score
            risk_df: DataFrame with Financial Risk, Compliance risk, Supply Chain Risk
            financial_df: DataFrame with financial metrics
            sustainability_df: DataFrame with Environmental score, Social Score, Governance score
        
        Returns:
            DataFrame with [Supplier, Supplier_Index]
        """
        # Merge all dataframes
        df = (
            performance_df.merge(risk_df, on="Supplier", how="left")
                         .merge(financial_df, on="Supplier", how="left")
                         .merge(sustainability_df, on="Supplier", how="left")
        )
        
        # Define weights
        weights = {
            "Quality score": 0.25,
            "Delivery score": 0.25,
            "Financial Risk": -0.15,
            "Compliance risk": -0.10,
            "Supply Chain Risk": -0.10,
            "Environmental score": 0.10,
            "Social Score": 0.03,
            "Governance score": 0.02
        }
        
        # Calculate weighted score
        score = pd.Series(0, index=df.index)
        
        for col, w in weights.items():
            if col in df.columns:
                c = df[col].fillna(df[col].median())
                norm = (c - c.min()) / (c.max() - c.min() + 0.001) * 100
                if w < 0:  # Invert risk scores
                    norm = 100 - norm
                score += w * norm
        
        # Normalize final score to 0-100
        df["Supplier_Index"] = (
            (score - score.min()) / (score.max() - score.min() + 0.001) * 100
        )
        
        self.supplier_index = df[["Supplier", "Supplier_Index"]]
        return self.supplier_index
    
    def filter_eligible_suppliers(self, performance_df, risk_df, financial_df, sustainability_df):
        """
        Apply governance thresholds to filter eligible suppliers
        
        Returns:
            DataFrame with eligible suppliers
        """
        df = (
            performance_df.merge(risk_df, on="Supplier", how="left")
                         .merge(financial_df, on="Supplier", how="left")
                         .merge(sustainability_df, on="Supplier", how="left")
        )
        
        eligible = df[
            (df["Financial Risk"].fillna(100) <= self.thresholds["max_financial_risk"]) &
            (df["Compliance risk"].fillna(100) <= self.thresholds["max_compliance_risk"]) &
            (df["Supply Chain Risk"].fillna(100) <= self.thresholds["max_supply_chain_risk"]) &
            (df["Environmental score"].fillna(0) >= self.thresholds["min_environmental"]) &
            (df["Social Score"].fillna(0) >= self.thresholds["min_social"]) &
            (df["Governance score"].fillna(0) >= self.thresholds["min_governance"]) &
            (df["Quality score"].fillna(0) >= self.thresholds["min_quality"]) &
            (df["Delivery score"].fillna(0) >= self.thresholds["min_delivery"])
        ]
        
        self.eligible_suppliers = eligible[["Supplier"]].drop_duplicates()
        return self.eligible_suppliers
    
    def recommend_suppliers(self, material_name, top_n=5, alpha=0.7):
        """
        Recommend suppliers for a given material using hybrid SVD + Index approach
        
        Args:
            material_name: Material category to recommend for
            top_n: Number of recommendations to return
            alpha: Weight of SVD score (0-1), (1-alpha) = weight of Supplier Index
        
        Returns:
            DataFrame with top N recommended suppliers and scores
        """
        if self.svd_model is None:
            raise ValueError("Must train SVD model first")
        if self.supplier_index is None:
            raise ValueError("Must calculate supplier index first")
        if self.eligible_suppliers is None:
            raise ValueError("Must filter eligible suppliers first")
        
        # Check if material exists
        if material_name not in self.material_to_idx:
            return pd.DataFrame(columns=["Supplier", "Final_Score", "SVD_Score", "Supplier_Index"])
        
        material_idx = self.material_to_idx[material_name]
        
        # Reconstruct the full matrix
        reconstructed = self.svd_model.inverse_transform(self.svd_model.transform(self.interaction_matrix))
        
        # Get predictions for this material
        material_scores = reconstructed[:, material_idx]
        
        # Create predictions dataframe
        preds = []
        for supplier_idx, score in enumerate(material_scores):
            supplier = self.idx_to_supplier[supplier_idx]
            preds.append((supplier, score))
        
        pred_df = pd.DataFrame(preds, columns=["Supplier", "SVD_Score"])
        
        # Apply eligibility filters
        pred_df = pred_df.merge(self.eligible_suppliers, on="Supplier", how="inner")
        
        if len(pred_df) == 0:
            return pd.DataFrame(columns=["Supplier", "Final_Score", "SVD_Score", "Supplier_Index"])
        
        # Add Supplier Index
        pred_df = pred_df.merge(self.supplier_index, on="Supplier", how="left")
        pred_df["Supplier_Index"] = pred_df["Supplier_Index"].fillna(0)
        
        # Normalize scores to 0-100
        if pred_df["SVD_Score"].max() > pred_df["SVD_Score"].min():
            pred_df["SVD_Norm"] = (
                (pred_df["SVD_Score"] - pred_df["SVD_Score"].min()) /
                (pred_df["SVD_Score"].max() - pred_df["SVD_Score"].min())
            ) * 100
        else:
            pred_df["SVD_Norm"] = 50
        
        pred_df["Index_Norm"] = pred_df["Supplier_Index"]
        
        # Calculate final hybrid score
        pred_df["Final_Score"] = (
            alpha * pred_df["SVD_Norm"] +
            (1 - alpha) * pred_df["Index_Norm"]
        )
        
        # Return top N
        result = (
            pred_df.sort_values("Final_Score", ascending=False)
                   .head(top_n)
                   [["Supplier", "Final_Score", "SVD_Score", "Supplier_Index"]]
        )
        
        return result
    
    def get_available_materials(self):
        """Get list of all available materials"""
        if self.material_to_idx is None:
            return []
        return sorted(self.material_to_idx.keys())
    
    def update_thresholds(self, **kwargs):
        """Update governance thresholds"""
        self.thresholds.update(kwargs)
