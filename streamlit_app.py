import streamlit as st
import pandas as pd
import numpy as np
import importlib
import networkx as nx
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import streamlit.components.v1 as components

# Authenticate with HuggingFace
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

# Page configuration
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("MIDUS Codes Clustering & Semantic Networks")

# Cache the model loader
@st.cache_resource
def load_model():
    module = importlib.import_module('sentence_transformers')
    SentenceTransformerCls = getattr(module, 'SentenceTransformer')
    return SentenceTransformerCls('all-MiniLM-L6-v2')

# Compute similarities and cluster labels
@st.cache_data
def compute_semantics_and_clusters(df_codes: pd.DataFrame, n_clusters: int):
    codes = df_codes.columns.tolist()
    # Co-occurrence matrix
    co_mat = df_codes.T.dot(df_codes).values
    np.fill_diagonal(co_mat, 0)

    # Semantic embeddings and similarity
    try:
        model = load_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
        sem_ok = True
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}. Skipping semantic analysis.")
        sim_mat = np.zeros((len(codes), len(codes)))
        sem_ok = False
        embeddings = None

    # Clustering: co-occurrence
    clust_cooc = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    # Clustering: semantic
    clust_sem = (
        AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        if sem_ok else np.full(len(codes), -1)
    )

    # Hybrid clustering
    if sem_ok:
        alpha = 0.5
        norm_cooc = co_mat / co_mat.max()
        hybrid_sim = alpha * norm_cooc + (1 - alpha) * sim_mat
        hybrid_dist = 1 - hybrid_sim
        clust_hybrid = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        ).fit_predict(hybrid_dist)
    else:
        hybrid_sim = np.zeros_like(sim_mat)
        clust_hybrid = np.full(len(codes), -1)

    # Assemble cluster DataFrame
    cluster_df = pd.DataFrame({
        'Code': codes,
        'Cluster_Cooccurrence': clust_cooc,
        'Cluster_Semantic': clust_sem,
        'Cluster_Hybrid': clust_hybrid
    })

    # Top semantic similarities
    pairs = [
        (codes[i], codes[j], float(sim_mat[i, j]))
        for i in range(len(codes)) for j in range(i+1, len(codes))
    ]
    sim_df = (
        pd.DataFrame(pairs, columns=['Code1', 'Code2', 'CosineSimilarity'])
          .sort_values('CosineSimilarity', ascending=False)
          .reset_index(drop=True)
    )

    return co_mat, sim_mat, hybrid_sim, cluster_df, sim_df, sem_ok

# Build NetworkX graph for PyVis
@st.cache_data
def build_network(sim_matrix: np.ndarray, labels: list, clusters: np.ndarray, threshold: float):
    G = nx.Graph()
    unique = np.unique(clusters)
    palette = sns.color_palette("hls", max(len(unique), 2))
    colors = [mcolors.to_hex(c) for c in palette]

    for i, label in enumerate(labels):
        color = colors[clusters[i] % len(colors)] if clusters[i] >= 0 else '#cccccc'
        G.add_node(label, color=color)
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            w = sim_matrix[i, j]
            if w > threshold:
                G.add_edge(labels[i], labels[j], weight=float(w))
    return G

# Render PyVis network as HTML
def render_pyvis(G: nx.Graph, height="700px", width="100%"):
    net = Network(height=height, width=width, notebook=False)
    net.from_nx(G)
    net.repulsion(node_distance=200, spring_length=200)
    return net.generate_html()

# Sidebar controls
st.sidebar.header("Settings")
threshold_cooc = st.sidebar.slider("Co-occurrence threshold", 0, 20, 5)
threshold_sem = st.sidebar.slider("Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05)
threshold_hybrid = st.sidebar.slider("Hybrid similarity threshold", 0.0, 1.0, 0.5, step=0.05)
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Excel file (.xlsx/.xls)", type=['xlsx','xls'])
if not uploaded_file:
    st.info("Upload your MIDUS coding Excel file to start.")
    st.stop()

# Load and preprocess codes
df = pd.read_excel(uploaded_file)
code_cols = [c for c in df.columns if c.endswith('_M2')]
if not code_cols:
    st.error("No columns ending with '_M2' found.")
    st.stop()

df_codes = (
    df[code_cols]
      .replace({'.': np.nan, ' ': np.nan})
      .apply(pd.to_numeric, errors='coerce')
      .fillna(0)
      .astype(int)
)

# Data snapshot
st.subheader("Data Snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df_codes.shape[0])
c2.metric("_M2 Codes", df_codes.shape[1])
c3.metric("Total Instances", int(df_codes.values.sum()))
with st.expander("View code matrix"):
    st.dataframe(df_codes)

# Compute semantics and clusters
with st.spinner("Computing..."):
    co_mat, sim_mat, hybrid_sim, cluster_df, sim_df, sem_ok = compute_semantics_and_clusters(df_codes, n_clusters)

# Display cluster assignments
st.subheader("Cluster Assignments")
st.dataframe(cluster_df)
st.download_button("Download clusters", cluster_df.to_csv(index=False), "clusters.csv")

# Display top semantic similarities
if sem_ok:
    st.subheader("Top 10 Semantic Similarities")
    st.dataframe(sim_df.head(10))
    st.download_button("Download similarities", sim_df.head(10).to_csv(index=False), "similarities.csv")

# Render co-occurrence network
st.subheader("Co-occurrence Network")
G_co = build_network(co_mat, code_cols, cluster_df['Cluster_Cooccurrence'].values, threshold_cooc)
components.html(render_pyvis(G_co), height=700)

# Render semantic network
if sem_ok:
    st.subheader("Semantic Similarity Network")
    G_se = build_network(sim_mat, code_cols, cluster_df['Cluster_Semantic'].values, threshold_sem)
    components.html(render_pyvis(G_se), height=700)

# Render hybrid network
st.subheader("Hybrid Similarity Network")
G_hy = build_network(hybrid_sim, code_cols, cluster_df['Cluster_Hybrid'].values, threshold_hybrid)
components.html(render_pyvis(G_hy), height=700)

# Compute and display network similarities
st.subheader("Network Similarities (Jaccard)")
def edge_set(G): return {frozenset(e) for e in G.edges()}
e_co = edge_set(G_co)
if sem_ok: e_se = edge_set(G_se)
e_hy = edge_set(G_hy)
st.write(f"Co-occurrence vs Semantic: {len(e_co & e_se)/len(e_co | e_se):.3f}" if sem_ok else "")
st.write(f"Co-occurrence vs Hybrid: {len(e_co & e_hy)/len(e_co | e_hy):.3f}")
st.write(f"Semantic vs Hybrid: {len(e_se & e_hy)/len(e_se | e_hy):.3f}" if sem_ok else "")
