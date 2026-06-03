"""
E-Commerce Product Recommendation System — Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛒",
    layout="wide"
)

# ── Load artifacts ────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    import scipy.sparse as sp
    svd         = joblib.load(os.path.join(APP_DIR, 'svd_model.pkl'))
    sparse_mat  = sp.load_npz(os.path.join(APP_DIR, 'sparse_matrix.npz'))
    idx_maps    = joblib.load(os.path.join(APP_DIR, 'index_maps.pkl'))
    content_sim = joblib.load(os.path.join(APP_DIR, 'content_sim_matrix.pkl'))
    prod_feat   = joblib.load(os.path.join(APP_DIR, 'product_features.pkl'))
    ratings     = pd.read_csv(os.path.join(APP_DIR, 'ratings_clean.csv'))
    return svd, sparse_mat, idx_maps, content_sim, prod_feat, ratings

try:
    svd_model, sparse_matrix, idx_maps, content_sim_df, product_features, ratings_df = load_artifacts()
    user_to_idx = idx_maps['user_to_idx']
    idx_to_prod = idx_maps['idx_to_prod']
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 E-Commerce Product Recommendation System")
st.markdown("""
> **Business Goal:** Personalise product discovery to increase revenue, engagement, and retention.
> Powered by Collaborative Filtering (SVD) and Content-Based Filtering.
""")
st.divider()

if not artifacts_loaded:
    st.warning("⚠️ Models not found. Please run `model_training.ipynb` first.")
    st.stop()

# ── Helper Functions ──────────────────────────────────────────────────────────
def get_user_recommendations(user_id, top_n=10):
    all_products   = ratings_df['productId'].unique()
    rated_products = ratings_df[ratings_df['userId'] == user_id]['productId'].unique()
    unrated        = [p for p in all_products if p not in rated_products]
    preds = [(p, svd_model.predict(user_id, p).est) for p in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(preds[:top_n], columns=['Product ID', 'Predicted Rating'])

def get_content_recommendations(product_id, top_n=10):
    if product_id not in content_sim_df.index:
        return pd.DataFrame()
    sim_scores = content_sim_df[product_id].sort_values(ascending=False).drop(product_id)
    top_prods  = sim_scores.head(top_n).reset_index()
    top_prods.columns = ['Product ID', 'Similarity Score']
    # Merge with product features for extra context
    top_prods = top_prods.merge(
        product_features[['productId', 'avg_rating', 'num_ratings']],
        left_on='Product ID', right_on='productId', how='left'
    ).drop('productId', axis=1)
    top_prods.columns = ['Product ID', 'Similarity Score', 'Avg Rating', '# Ratings']
    return top_prods.round(3)

def get_user_history(user_id, top_n=5):
    user_df = ratings_df[ratings_df['userId'] == user_id].sort_values('rating', ascending=False)
    return user_df[['productId', 'rating']].head(top_n).rename(
        columns={'productId': 'Product ID', 'rating': 'Rating'})

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "👤 User-Based Recommendations (SVD)",
    "📦 Product Similarity (Content-Based)",
    "📊 Dataset Overview"
])

# ── Tab 1: SVD Recommendations ────────────────────────────────────────────────
with tab1:
    st.subheader("Personalised Recommendations — Matrix Factorization (SVD)")
    st.markdown("Enter a User ID to get their top personalised product recommendations.")

    all_users = ratings_df['userId'].unique().tolist()
    user_id_input = st.selectbox(
        "Select or type a User ID",
        options=all_users,
        index=0,
        help="These are users from the training dataset"
    )
    top_n_svd = st.slider("Number of recommendations", 5, 20, 10, key="svd_slider")

    if st.button("🔮 Get Recommendations", key="svd_btn"):
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### 🎯 Top Recommendations")
            recs = get_user_recommendations(user_id_input, top_n=top_n_svd)
            recs.index = range(1, len(recs) + 1)
            recs['Predicted Rating'] = recs['Predicted Rating'].round(2)

            def color_rating(val):
                color = '#2ecc71' if val >= 4 else '#f39c12' if val >= 3 else '#e74c3c'
                return f'background-color: {color}22; color: {color}; font-weight: bold'

            styled = recs.style.map(color_rating, subset=['Predicted Rating'])
            st.dataframe(styled, use_container_width=True)

        with col2:
            st.markdown("#### 📋 User's Rating History (Top 5)")
            history = get_user_history(user_id_input)
            if not history.empty:
                st.dataframe(history.reset_index(drop=True), use_container_width=True)
                avg = history['Rating'].mean()
                st.metric("User's Average Rating", f"{avg:.2f} ⭐")
                st.caption(f"Total products rated: {len(ratings_df[ratings_df['userId'] == user_id_input])}")
            else:
                st.info("No history found for this user.")

# ── Tab 2: Content-Based ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Similar Products — Content-Based Filtering (Cosine Similarity)")
    st.markdown("Find products similar to a given product based on rating patterns.")

    all_products = product_features['productId'].tolist()
    product_id_input = st.selectbox(
        "Select a Product ID",
        options=all_products,
        index=0,
        help="Similarity is computed using rating statistics"
    )
    top_n_content = st.slider("Number of similar products", 5, 20, 10, key="content_slider")

    if st.button("🔍 Find Similar Products", key="content_btn"):
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown("#### 📦 Similar Products")
            sim_recs = get_content_recommendations(product_id_input, top_n=top_n_content)
            if not sim_recs.empty:
                sim_recs.index = range(1, len(sim_recs) + 1)
                st.dataframe(sim_recs, use_container_width=True)
            else:
                st.warning("Product not found in similarity matrix.")

        with col2:
            st.markdown("#### 📊 Selected Product Stats")
            prod_stats = product_features[product_features['productId'] == product_id_input]
            if not prod_stats.empty:
                st.metric("Average Rating", f"{prod_stats['avg_rating'].values[0]:.2f} ⭐")
                st.metric("Number of Ratings", int(prod_stats['num_ratings'].values[0]))
                std_val = prod_stats['rating_std'].values[0]
                st.metric("Rating Std Dev", f"{std_val:.2f}")
                if std_val > 1.5:
                    st.warning("⚡ High rating variance — polarising product")
                elif std_val < 0.5:
                    st.success("✅ Consistent ratings — reliable product")

# ── Tab 3: Dataset Overview ───────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Dataset Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Ratings",  f"{len(ratings_df):,}")
    m2.metric("Unique Users",   f"{ratings_df['userId'].nunique():,}")
    m3.metric("Unique Products",f"{ratings_df['productId'].nunique():,}")
    m4.metric("Avg Rating",     f"{ratings_df['rating'].mean():.2f} ⭐")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rating Distribution")
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)

    with col2:
        st.markdown("#### Top 10 Most-Rated Products")
        top_prods = ratings_df['productId'].value_counts().head(10).reset_index()
        top_prods.columns = ['Product ID', 'Number of Ratings']
        top_prods['Product ID'] = top_prods['Product ID'].str[:20] + '...'
        st.dataframe(top_prods, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Sample Raw Data")
    st.dataframe(ratings_df.sample(min(20, len(ratings_df))).reset_index(drop=True),
                 use_container_width=True)