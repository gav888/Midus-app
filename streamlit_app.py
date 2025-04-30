import streamlit as st
import io
import pandas as pd
import importlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.colors

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("MIDUS Codes Clustering & Semantic Networks")

# --- Caching with new APIs ---
@st.cache_resource
def load_model():
    """Lazy-load SentenceTransformer to avoid front-end watcher issues."""
    try:
        s2 = importlib.import_module('sentence_transformers')
        SentenceTransformer = getattr(s2, 'SentenceTransformer')
    except ImportError as e:
        st.error(f"Could not import sentence_transformers: {e}")
        raise
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def compute_cooccurrence(df_codes: pd.DataFrame):
    """Compute co-occurrence matrix from code DataFrame."""
    mat = np.dot(df_codes.T, df_codes)
    np.fill_diagonal(mat, 0)
    return mat

@st.cache_data
def compute_semantics_and_clusters(df_codes: pd.DataFrame, n_clusters: int):
    """
    Compute semantic embeddings, semantic similarity matrix,
    and clustering assignments (both co-occurrence & semantic).
    Returns:
      - co_occurrence matrix
      - sim_mat (or None if failed)
      - cluster_df with two cluster labels (semantic may be -1)
      - sim_df for top similarities (empty if failed)
      - semantic_available flag
    """
    codes = df_codes.columns.tolist()
    # Co-occurrence
    co_mat = np.dot(df_codes.T, df_codes)
    np.fill_diagonal(co_mat, 0)
    # Semantic embeddings
    semantic_ok = True
    try:
        model = load_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}\nSkipping semantic analysis.")
        semantic_ok = False
        sim_mat = None
    # Clustering on co-occurrence
    clust_cooc = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    # Clustering on semantic if available
    if semantic_ok:
        clust_sem = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    else:
        clust_sem = np.array([-1] * len(codes))
    # Cluster DataFrame
    cluster_df = pd.DataFrame({
        'Code': codes,
        'Cluster_Cooccurrence': clust_cooc,
        'Cluster_Semantic': clust_sem
    })
    # Top semantic similarities
    if semantic_ok:
        pairs = [(codes[i], codes[j], sim_mat[i, j])
                 for i in range(len(codes)) for j in range(i+1, len(codes))]
        sim_df = (pd.DataFrame(pairs, columns=['Code1', 'Code2', 'CosineSimilarity'])
                  .sort_values('CosineSimilarity', ascending=False)
                  .reset_index(drop=True))
    else:
        sim_df = pd.DataFrame(columns=['Code1', 'Code2', 'CosineSimilarity'])
    return co_mat, sim_mat, cluster_df, sim_df, semantic_ok

@st.cache_data
def build_network(matrix: np.ndarray, labels: list, clusters: np.ndarray, threshold: float):
    """Build a NetworkX graph colored by cluster assignments."""
    G = nx.Graph()
    unique_clusters = np.unique(clusters)
    palette = sns.color_palette("hls", max(len(unique_clusters), 2))
    hex_colors = [matplotlib.colors.to_hex(c) for c in palette]
    for idx, label in enumerate(labels):
        color = hex_colors[0] if clusters[idx] < 0 else hex_colors[clusters[idx] % len(hex_colors)]
        G.add_node(label, color=color)
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            w = matrix[i, j]
            if w > threshold:
                G.add_edge(labels[i], labels[j], weight=w)
    return G

# --- Sidebar Inputs ---
st.sidebar.header("Data & Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file (.xlsx/.xls)", type=['xlsx', 'xls']
)
n_clusters = st.sidebar.slider("# Clusters", 2, 10, 5)
threshold_cooc = st.sidebar.slider("Co-occurrence threshold", 1, 20, 5)

# Semantic threshold slider only shown when semantic available preview
def placeholder(): return None
threshold_sem = placeholder()

# --- Main App Logic ---
if not uploaded_file:
    st.info("Please upload the MIDUS coding Excel file to proceed.")
else:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    # Filter _M2 codes
    codes = [col for col in df.columns if col.endswith('_M2')]
    if not codes:
        st.warning("No '_M2' columns found.")
        st.stop()
    df_codes = df[codes].replace({'.': np.nan, ' ': np.nan})
    df_codes = df_codes.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    # Data summary
    st.subheader("Data Snapshot & Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_codes.shape[0])
    c2.metric("_M2 Codes", df_codes.shape[1])
    c3.metric("Total Instances", int(df_codes.values.sum()))
    with st.expander("Show raw code matrix"):
        st.dataframe(df_codes)
    # Compute everything
    with st.spinner("Calculating co-occurrence and semantic clusters..."):
        co_mat, sim_mat, cluster_df, sim_df, sem_ok = compute_semantics_and_clusters(df_codes, n_clusters)
    # If semantic ok, show semantic threshold slider
    if sem_ok:
        threshold_sem = st.sidebar.slider("Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05)
    # Show cluster assignments
    st.subheader("Cluster Assignments")
    st.dataframe(cluster_df)
    st.download_button("Download cluster assignments", cluster_df.to_csv(index=False), "clusters.csv")
    # Show top similarities if available
    if sem_ok:
        st.subheader("Top Semantic Similarities")
        st.dataframe(sim_df.head(15))
        st.download_button("Download semantic similarities", sim_df.to_csv(index=False), "similarities.csv")
    # Networks
    st.subheader("Networks Visualization")
    colA, colB = st.columns(2)
    # Co-occurrence network
    with colA:
        st.markdown("**Co-occurrence Network**")
        Gc = build_network(co_mat, codes, cluster_df['Cluster_Cooccurrence'].values, threshold_cooc)
        pos = nx.spring_layout(Gc, seed=42)
        fig, ax = plt.subplots(figsize=(5,5))
        nx.draw(Gc, pos, with_labels=True,
                node_color=[Gc.nodes[n]['color'] for n in Gc.nodes()],
                edge_color='lightblue', width=2, alpha=0.7, ax=ax)
        ax.axis('off')
        st.pyplot(fig)
    # Semantic network if available
    if sem_ok:
        with colB:
            st.markdown("**Semantic Similarity Network**")
            Gs = build_network(sim_mat, codes, cluster_df['Cluster_Semantic'].values, threshold_sem)
            pos2 = nx.kamada_kawai_layout(Gs)
            fig2, ax2 = plt.subplots(figsize=(5,5))
            edges = list(Gs.edges(data=True))
            weights = [attr['weight'] for *_, attr in edges]
            if weights:
                wmin, wmax = min(weights), max(weights)
                for u,v,attr in edges:
                    w = attr['weight']; norm=(w-wmin)/(wmax-wmin) if wmax>wmin else 1
                    nx.draw_networkx_edges(Gs, pos2, edgelist=[(u,v)],
                        width=0.5+3*norm, alpha=0.3+0.7*norm, edge_color='#888888', ax=ax2)
            nx.draw_networkx_nodes(Gs, pos2,
                node_size=[300+20*Gs.degree(n) for n in Gs.nodes()],
                node_color=[Gs.nodes[n]['color'] for n in Gs.nodes()], ax=ax2)
            nx.draw_networkx_labels(Gs, pos2, font_size=8, ax=ax2)
            ax2.axis('off')
            st.pyplot(fig2)
    st.success("Analysis complete!")
