"""
PPx SVD-Based Recommender System
---------------------------------
Enterprise-grade supplier recommendation using matrix factorization
with governance thresholds and supplier index re-ranking.
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


class PPxRecommenderSystem:
    """
    SVD-based recommendation system with eligibility gates and index re-ranking
    """
    
    def __init__(self):
        self.svd_model = None
        self.supplier_index = None
        self.eligible_suppliers = None
        self.svd_df = None
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
        Build supplier-material interactions from purchase data
        
        Args:
            pdt_df: DataFrame with columns [Supplier, Category Family, Sub Category, PO Amount, Po Number]
        
        Returns:
            DataFrame with [Supplier, Material, Rating]
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
        
        # Calculate rating (1-5 scale)
        interactions["Rating"] = (
            0.6 * interactions["Value_Score"] +
            0.4 * interactions["Freq_Score"]
        ) * 5
        
        self.svd_df = interactions[["Supplier", "Material", "Rating"]]
        return self.svd_df
    
    def train_svd(self, n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02):
        """
        Train SVD model on supplier-material interactions
        """
        if self.svd_df is None:
            raise ValueError("Must call prepare_interactions() first")
        
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(self.svd_df, reader)
        trainset, _ = train_test_split(dataset, test_size=0.2, random_state=42)
        
        self.svd_model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.svd_model.fit(trainset)
        
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
        
        # Get all suppliers
        suppliers = self.svd_df["Supplier"].unique()
        
        # Predict SVD scores for all suppliers
        preds = []
        for s in suppliers:
            try:
                pred_score = self.svd_model.predict(s, material_name).est
                preds.append((s, pred_score))
            except:
                continue
        
        if not preds:
            return pd.DataFrame(columns=["Supplier", "Final_Score", "SVD_Score", "Supplier_Index"])
        
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
        if self.svd_df is None:
            return []
        return sorted(self.svd_df["Material"].unique().tolist())
    
    def update_thresholds(self, **kwargs):
        """Update governance thresholds"""
        self.thresholds.update(kwargs)
