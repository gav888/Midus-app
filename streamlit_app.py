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
from pyvis.network import Network as PyvisNetwork
import streamlit.components.v1 as components

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

    # --- Process M3 codes at time 2 ---
    codes_M3 = [col for col in df.columns if col.endswith('_M3')]
    df_codes_M3 = df[codes_M3].replace({'.': np.nan, ' ': np.nan})
    df_codes_M3 = df_codes_M3.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Data summary
    st.subheader("Data Snapshot & Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_codes.shape[0])
    c2.metric("_M2 Codes", df_codes.shape[1])
    c3.metric("Total Instances", int(df_codes.values.sum()))
    with st.expander("Show raw code matrix"):
        st.dataframe(df_codes)

    # M3 Data Snapshot & Summary
    st.subheader("M3 Data Snapshot & Summary")
    d1, d2, d3 = st.columns(3)
    d1.metric("Rows", df_codes_M3.shape[0])
    d2.metric("_M3 Codes", df_codes_M3.shape[1])
    d3.metric("Total Instances", int(df_codes_M3.values.sum()))
    with st.expander("Show raw M3 code matrix"):
        st.dataframe(df_codes_M3)

    # Compute everything
    with st.spinner("Calculating co-occurrence and semantic clusters..."):
        co_mat_M2, sim_mat_M2, cluster_df_M2, sim_df_M2, sem_ok_M2 = \
            compute_semantics_and_clusters(df_codes, n_clusters)
        co_mat_M3, sim_mat_M3, cluster_df_M3, sim_df_M3, sem_ok_M3 = \
            compute_semantics_and_clusters(df_codes_M3, n_clusters)

    # If semantic ok, show semantic threshold slider
    if sem_ok_M2 or sem_ok_M3:
        threshold_sem = st.sidebar.slider("Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05)

    # --- Build network graphs for interactive rendering ---
    Gc = build_network(co_mat_M2, codes,
                       cluster_df_M2['Cluster_Cooccurrence'].values,
                       threshold_cooc)
    if sem_ok_M2:
        Gs = build_network(sim_mat_M2, codes,
                           cluster_df_M2['Cluster_Semantic'].values,
                           threshold_sem)
    Gc_M3 = build_network(co_mat_M3, codes_M3,
                          cluster_df_M3['Cluster_Cooccurrence'].values,
                          threshold_cooc)
    if sem_ok_M3:
        Gs_M3 = build_network(sim_mat_M3, codes_M3,
                              cluster_df_M3['Cluster_Semantic'].values,
                              threshold_sem)

    # --- Hybrid clustering & similarity for M2 ---
    alpha = 0.5
    norm_co_mat_M2 = co_mat_M2 / co_mat_M2.max()
    hybrid_sim_M2 = alpha * norm_co_mat_M2 + (1 - alpha) * sim_mat_M2
    hybrid_dist_M2 = 1 - hybrid_sim_M2
    clust_hybrid_M2 = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    ).fit_predict(hybrid_dist_M2)
    # Color palette for hybrid
    palette_hyb2 = sns.color_palette("hls", max(len(np.unique(clust_hybrid_M2)),2))
    hex_colors_hyb2 = [matplotlib.colors.to_hex(c) for c in palette_hyb2]
    # Build hybrid network for M2
    G_hybrid_M2 = nx.Graph()
    for i, label in enumerate(codes):
        G_hybrid_M2.add_node(label, color=hex_colors_hyb2[clust_hybrid_M2[i]])
    flat_vals2 = hybrid_sim_M2[np.triu_indices_from(hybrid_sim_M2, k=1)]
    threshold_hyb2 = np.percentile(flat_vals2, 75)
    for i in range(len(codes)):
        for j in range(i+1, len(codes)):
            if hybrid_sim_M2[i, j] > threshold_hyb2:
                G_hybrid_M2.add_edge(codes[i], codes[j], weight=hybrid_sim_M2[i, j])

    # --- Hybrid clustering & similarity for M3 ---
    norm_co_mat_M3 = co_mat_M3 / co_mat_M3.max()
    hybrid_sim_M3 = alpha * norm_co_mat_M3 + (1 - alpha) * sim_mat_M3
    hybrid_dist_M3 = 1 - hybrid_sim_M3
    clust_hybrid_M3 = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    ).fit_predict(hybrid_dist_M3)
    palette_hyb3 = sns.color_palette("hls", max(len(np.unique(clust_hybrid_M3)),2))
    hex_colors_hyb3 = [matplotlib.colors.to_hex(c) for c in palette_hyb3]
    G_hybrid_M3 = nx.Graph()
    for i, label in enumerate(codes_M3):
        G_hybrid_M3.add_node(label, color=hex_colors_hyb3[clust_hybrid_M3[i]])
    flat_vals3 = hybrid_sim_M3[np.triu_indices_from(hybrid_sim_M3, k=1)]
    threshold_hyb3 = np.percentile(flat_vals3, 75)
    for i in range(len(codes_M3)):
        for j in range(i+1, len(codes_M3)):
            if hybrid_sim_M3[i, j] > threshold_hyb3:
                G_hybrid_M3.add_edge(codes_M3[i], codes_M3[j], weight=hybrid_sim_M3[i, j])

    # --- Interactive Network Visualizations ---
    def render_pyvis(G: nx.Graph, height=800, width=1200):
        net = PyvisNetwork(height=f"{height}px", width=f"{width}px", directed=False)
        for node, data in G.nodes(data=True):
            nid = str(node)
            net.add_node(nid, label=nid, color=data.get("color"))
        for u, v, data in G.edges(data=True):
            uid, vid = str(u), str(v)
            w = data.get("weight", 1)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            net.add_edge(uid, vid, value=w)
        net.repulsion(node_distance=100, central_gravity=0.2)
        html = net.generate_html()
        components.html(html, height=height, width=width, scrolling=True)

    st.subheader("M2 Networks")
    st.markdown("**Co-occurrence Network**")
    render_pyvis(Gc, height=800, width=1200)
    if sem_ok_M2:
        st.markdown("**Semantic Similarity Network**")
        render_pyvis(Gs, height=800, width=1200)
    st.markdown("**Hybrid Similarity Network**")
    render_pyvis(G_hybrid_M2, height=800, width=1200)

    st.subheader("M3 Networks")
    st.markdown("**Co-occurrence Network**")
    render_pyvis(Gc_M3, height=800, width=1200)
    if sem_ok_M3:
        st.markdown("**Semantic Similarity Network**")
        render_pyvis(Gs_M3, height=800, width=1200)
    st.markdown("**Hybrid Similarity Network**")
    render_pyvis(G_hybrid_M3, height=800, width=1200)

    st.success("Analysis complete!")    
