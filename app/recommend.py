"""
recommend.py — Core recommendation logic module
Can be imported by streamlit_app.py or used standalone in scripts.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity


class ChurnPredictor:
    """Wrapper for the churn prediction model."""

    def __init__(self, model_path, scaler_path, feature_names_path):
        self.model         = joblib.load(model_path)
        self.scaler        = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)

    def predict(self, input_df: pd.DataFrame):
        """Returns (prediction, probability) for a single customer."""
        input_df = input_df[self.feature_names]
        prob = self.model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)
        return pred, round(prob, 4)


class RecommendationEngine:
    """Wrapper for the SVD + content-based recommendation models."""

    def __init__(self, svd_path, ui_matrix_path, content_sim_path, prod_feat_path, ratings_path):
        self.svd         = joblib.load(svd_path)
        self.ui_matrix   = joblib.load(ui_matrix_path)
        self.content_sim = joblib.load(content_sim_path)
        self.prod_feat   = joblib.load(prod_feat_path)
        self.ratings     = pd.read_csv(ratings_path)

    def svd_recommend(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """Get top N recommendations for a user using SVD."""
        all_products   = self.ratings['productId'].unique()
        rated_products = self.ratings[self.ratings['userId'] == user_id]['productId'].unique()
        unrated        = [p for p in all_products if p not in rated_products]

        preds = [(p, self.svd.predict(user_id, p).est) for p in unrated]
        preds.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(preds[:top_n], columns=['productId', 'predicted_rating'])

    def content_recommend(self, product_id: str, top_n: int = 10) -> pd.DataFrame:
        """Get top N similar products using content-based cosine similarity."""
        if product_id not in self.content_sim.index:
            return pd.DataFrame(columns=['productId', 'similarity'])

        sim_scores = (
            self.content_sim[product_id]
            .sort_values(ascending=False)
            .drop(product_id)
            .head(top_n)
            .reset_index()
        )
        sim_scores.columns = ['productId', 'similarity']
        return sim_scores

    def user_history(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Get a user's highest-rated products."""
        user_df = (
            self.ratings[self.ratings['userId'] == user_id]
            .sort_values('rating', ascending=False)
            .head(top_n)[['productId', 'rating']]
        )
        return user_df.reset_index(drop=True)

    def hybrid_recommend(self, user_id: str, top_n: int = 10,
                          svd_weight: float = 0.7) -> pd.DataFrame:
        """
        Hybrid recommendation combining SVD and content-based scores.
        svd_weight: how much weight to give SVD vs content-based (0-1)
        """
        # SVD predictions for all unrated items
        all_products   = self.ratings['productId'].unique()
        rated_products = self.ratings[self.ratings['userId'] == user_id]['productId'].unique()
        unrated        = [p for p in all_products if p not in rated_products]

        svd_scores = {p: self.svd.predict(user_id, p).est for p in unrated}

        # Content-based: average similarity to user's top 5 rated products
        top_rated = self.user_history(user_id, top_n=5)['productId'].tolist()
        content_scores = {}

        for prod in unrated:
            sims = []
            for rated_prod in top_rated:
                if rated_prod in self.content_sim.index and prod in self.content_sim.columns:
                    sims.append(self.content_sim.loc[rated_prod, prod])
            content_scores[prod] = np.mean(sims) if sims else 0

        # Normalize
        max_svd     = max(svd_scores.values())     if svd_scores     else 1
        max_content = max(content_scores.values()) if content_scores else 1

        hybrid = []
        for prod in unrated:
            svd_norm     = svd_scores.get(prod, 0)     / max_svd
            content_norm = content_scores.get(prod, 0) / max_content
            combined     = svd_weight * svd_norm + (1 - svd_weight) * content_norm
            hybrid.append((prod, combined))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(hybrid[:top_n], columns=['productId', 'hybrid_score'])


def precision_at_k(predictions, k=10, threshold=4.0):
    """Compute Precision@K from Surprise predictions."""
    from collections import defaultdict
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k       = user_ratings[:k]
        n_rel_rec_k = sum(1 for (_, true_r) in top_k if true_r >= threshold)
        precisions.append(n_rel_rec_k / k if k > 0 else 0)

    return round(np.mean(precisions), 4)


def ndcg_at_k(predictions, k=10, threshold=4.0):
    """Compute NDCG@K from Surprise predictions."""
    from collections import defaultdict
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        dcg  = sum((1 if true_r >= threshold else 0) / np.log2(i + 2)
                   for i, (_, true_r) in enumerate(top_k))
        n_rel = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)
        idcg  = sum(1 / np.log2(i + 2) for i in range(min(n_rel, k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return round(np.mean(ndcgs), 4)
