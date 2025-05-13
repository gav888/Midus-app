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

from sklearn.metrics import silhouette_score, calinski_harabasz_score

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

# M2 settings
st.sidebar.header("M2 Settings")
n_clusters_M2 = st.sidebar.slider("M2 # Clusters", 2, 10, 5)
threshold_cooc_M2 = st.sidebar.slider("M2 Co-occurrence threshold", 1, 20, 5)
threshold_sem_M2 = st.sidebar.slider(
    "M2 Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05
)

# M3 settings
st.sidebar.header("M3 Settings")
n_clusters_M3 = st.sidebar.slider("M3 # Clusters", 2, 10, 5)
threshold_cooc_M3 = st.sidebar.slider("M3 Co-occurrence threshold", 1, 20, 5)
threshold_sem_M3 = st.sidebar.slider(
    "M3 Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05
)

# --- Main App Logic ---
if not uploaded_file:
    st.info("Please upload the MIDUS coding Excel file to proceed.")
else:
    # Load workbook and let user pick sheet
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.sidebar.selectbox("Select sheet", xls.sheet_names)
    try:
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        st.error(f"Could not read sheet {sheet}: {e}")
        st.stop()

    # Discover code columns flexibly using regex
    codes_M2 = df.filter(regex=r'(?i)_M2$').columns.tolist()
    codes_M3 = df.filter(regex=r'(?i)_M3$').columns.tolist()

    if not codes_M2:
        st.warning("No `_M2` columns found. Check your sheet or column naming.")
        st.stop()
    if not codes_M3:
        st.warning("No `_M3` columns found. Check your sheet or column naming.")
        st.stop()

    # Convert to numeric DataFrames
    df_codes = (
        df[codes_M2]
        .replace({'.': np.nan, ' ': np.nan})
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .astype(int)
    )
    df_codes_M3 = (
        df[codes_M3]
        .replace({'.': np.nan, ' ': np.nan})
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .astype(int)
    )

# --- Compute everything ---
with st.spinner("Calculating co-occurrence and semantic clusters..."):
    co_mat_M2, sim_mat_M2, cluster_df_M2, sim_df_M2, sem_ok_M2 = \
        compute_semantics_and_clusters(df_codes, n_clusters_M2)
    co_mat_M3, sim_mat_M3, cluster_df_M3, sim_df_M3, sem_ok_M3 = \
        compute_semantics_and_clusters(df_codes_M3, n_clusters_M3)


# --- Cluster Validity Analysis for k=2 to 8 ---
@st.cache_data
def evaluate_k(co_mat, embeddings, ks):
    results = []
    for k in ks:
        # Co-occurrence clustering validity
        labels_co = AgglomerativeClustering(n_clusters=k).fit_predict(co_mat)
        sil_co = silhouette_score(co_mat, labels_co, metric='euclidean')
        ch_co  = calinski_harabasz_score(co_mat, labels_co)
        # Semantic clustering validity
        labels_sem = AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)
        sil_sem = silhouette_score(embeddings, labels_sem, metric='cosine')
        ch_sem  = calinski_harabasz_score(embeddings, labels_sem)
        results.append({
            'k': k,
            'silhouette_co': sil_co,
            'ch_co': ch_co,
            'silhouette_sem': sil_sem,
            'ch_sem': ch_sem
        })
    return pd.DataFrame(results)

# Compute embeddings once
embeddings_M2 = load_model().encode(codes_M2)
embeddings_M3 = load_model().encode(codes_M3)

# Evaluate cluster validity for M2 and M3
ks = list(range(2, 9))
eval_df_M2 = evaluate_k(co_mat_M2, embeddings_M2, ks)
eval_df_M3 = evaluate_k(co_mat_M3, embeddings_M3, ks)

# Display results
st.subheader("Cluster Validity Across k=2 to 8 (M2)")
st.dataframe(eval_df_M2)
st.subheader("Cluster Validity Across k=2 to 8 (M3)")
st.dataframe(eval_df_M3)



# --- Build network graphs for interactive rendering ---
Gc = build_network(co_mat_M2, codes_M2,
                   cluster_df_M2['Cluster_Cooccurrence'].values,
                   threshold_cooc_M2)
if sem_ok_M2:
    Gs = build_network(sim_mat_M2, codes_M2,
                       cluster_df_M2['Cluster_Semantic'].values,
                       threshold_sem_M2)
Gc_M3 = build_network(co_mat_M3, codes_M3,
                      cluster_df_M3['Cluster_Cooccurrence'].values,
                      threshold_cooc_M3)
if sem_ok_M3:
    Gs_M3 = build_network(sim_mat_M3, codes_M3,
                          cluster_df_M3['Cluster_Semantic'].values,
                          threshold_sem_M3)

# --- Hybrid clustering & similarity for M2 ---
alpha = 0.5
norm_co_mat_M2 = co_mat_M2 / co_mat_M2.max()
hybrid_sim_M2 = alpha * norm_co_mat_M2 + (1 - alpha) * sim_mat_M2
hybrid_dist_M2 = 1 - hybrid_sim_M2
clust_hybrid_M2 = AgglomerativeClustering(
    n_clusters=n_clusters_M2, metric='precomputed', linkage='average'
).fit_predict(hybrid_dist_M2)
# Color palette for hybrid
palette_hyb2 = sns.color_palette("hls", max(len(np.unique(clust_hybrid_M2)),2))
hex_colors_hyb2 = [matplotlib.colors.to_hex(c) for c in palette_hyb2]
# Build hybrid network for M2
G_hybrid_M2 = nx.Graph()
for i, label in enumerate(codes_M2):
    G_hybrid_M2.add_node(label, color=hex_colors_hyb2[clust_hybrid_M2[i]])
flat_vals2 = hybrid_sim_M2[np.triu_indices_from(hybrid_sim_M2, k=1)]
threshold_hyb2 = np.percentile(flat_vals2, 75)
for i in range(len(codes_M2)):
    for j in range(i+1, len(codes_M2)):
        if hybrid_sim_M2[i, j] > threshold_hyb2:
            G_hybrid_M2.add_edge(codes_M2[i], codes_M2[j], weight=hybrid_sim_M2[i, j])

# --- Hybrid clustering & similarity for M3 ---
norm_co_mat_M3 = co_mat_M3 / co_mat_M3.max()
hybrid_sim_M3 = alpha * norm_co_mat_M3 + (1 - alpha) * sim_mat_M3
hybrid_dist_M3 = 1 - hybrid_sim_M3
clust_hybrid_M3 = AgglomerativeClustering(
    n_clusters=n_clusters_M3, metric='precomputed', linkage='average'
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
