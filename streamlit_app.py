import streamlit as st
import pandas as pd
import numpy as np
import importlib
import networkx as nx
import seaborn as sns
import matplotlib.colors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from pyvis.network import Network
import streamlit.components.v1 as components

# Authenticate and configure
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MIDUS Codes Clustering & Semantic Networks")

# Caching and model loader
def _import_sentence_transformers():
    module = importlib.import_module('sentence_transformers')
    return getattr(module, 'SentenceTransformer')

@st.cache_resource
def load_model():
    SentenceTransformer = _import_sentence_transformers()
    return SentenceTransformer('all-MiniLM-L6-v2')

# Compute clusters and similarities
@st.cache_data
def compute_semantics_and_clusters(df_codes: pd.DataFrame, n_clusters: int):
    codes = df_codes.columns.tolist()
    # Co-occurrence matrix
    co_mat = df_codes.T.dot(df_codes).values
    np.fill_diagonal(co_mat, 0)

    # Semantic embeddings & similarity
    semantic_ok = True
    try:
        model = load_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}. Skipping semantic analysis.")
        semantic_ok = False
        sim_mat = None

    # Clustering
    clust_cooc = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    clust_sem = (
        AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        if semantic_ok else np.full(len(codes), -1)
    )

    # DataFrames
    cluster_df = pd.DataFrame({
        'Code': codes,
        'Cluster_Cooccurrence': clust_cooc,
        'Cluster_Semantic': clust_sem
    })
    if semantic_ok:
        pairs = [(codes[i], codes[j], float(sim_mat[i,j]))
                 for i in range(len(codes)) for j in range(i+1, len(codes))]
        sim_df = pd.DataFrame(pairs, columns=['Code1','Code2','CosineSimilarity'])
        sim_df = sim_df.sort_values('CosineSimilarity', ascending=False).reset_index(drop=True)
    else:
        sim_df = pd.DataFrame(columns=['Code1','Code2','CosineSimilarity'])

    return co_mat, sim_mat, cluster_df, sim_df, semantic_ok

# Build NetworkX graph for PyVis
@st.cache_data
def build_network(matrix: np.ndarray, labels: list, clusters: np.ndarray, threshold: float):
    G = nx.Graph()
    unique = np.unique(clusters)
    palette = sns.color_palette("hls", max(len(unique), 2))
    colors = [matplotlib.colors.to_hex(c) for c in palette]

    # Add nodes
    for i, label in enumerate(labels):
        color = colors[0] if clusters[i] < 0 else colors[clusters[i] % len(colors)]
        G.add_node(label, color=color, degree=0)
    # Add edges
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            weight = matrix[i, j]
            if weight > threshold:
                G.add_edge(labels[i], labels[j], weight=float(weight))
                G.nodes[labels[i]]['degree'] += 1
                G.nodes[labels[j]]['degree'] += 1
    return G

# Render PyVis network
def render_pyvis(G, height="700px", width="100%"):
    net = Network(height=height, width=width, notebook=False)
    net.force_atlas_2based()
    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=node,
            color=data['color'],
            title=f"Degree: {int(data['degree'])}"
        )
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=int(data['weight']))
    # Save to HTML and return string
    path = f"network_{id(G)}.html"
    net.save_graph(path)
    return open(path, 'r', encoding='utf-8').read()

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
file = st.sidebar.file_uploader("Upload Excel file (.xlsx/.xls)", type=['xlsx','xls'])
n_clusters = st.sidebar.slider("# Clusters", 2, 10, 5)
threshold_cooc = st.sidebar.slider("Co-occurrence threshold", 1, 20, 5)

# Main flow
if not file:
    st.info("Upload your MIDUS coding Excel file to start.")
    st.stop()

df = pd.read_excel(file)
codes = [c for c in df.columns if c.endswith('_M2')]
if not codes:
    st.error("No columns ending with '_M2' found.")
    st.stop()

df_codes = df[codes].replace({'.':np.nan,' ':np.nan}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Summary metrics
st.subheader("Data Snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df_codes.shape[0])
c2.metric("_M2 Codes", df_codes.shape[1])
c3.metric("Total Instances", int(df_codes.values.sum()))
with st.expander("View code matrix"):
    st.dataframe(df_codes)

# Compute clustering & similarities
with st.spinner("Computing..."):
    co_mat, sim_mat, cluster_df, sim_df, sem_ok = compute_semantics_and_clusters(df_codes, n_clusters)

# Display tables
st.subheader("Cluster Assignments")
st.dataframe(cluster_df)
st.download_button("Download clusters", cluster_df.to_csv(index=False), "clusters.csv")

if sem_ok:
    st.subheader("Top Semantic Similarities")
    st.dataframe(sim_df.head(15))
    st.download_button("Download similarities", sim_df.to_csv(index=False), "similarities.csv")
    threshold_sem = st.sidebar.slider("Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05)
else:
    threshold_sem = 0.0

# Interactive networks stacked
st.subheader("Co-occurrence Network")
G_co = build_network(co_mat, codes, cluster_df['Cluster_Cooccurrence'].values, threshold_cooc)
html_co = render_pyvis(G_co)
components.html(html_co, height=700)

if sem_ok:
    st.subheader("Semantic Similarity Network")
    G_se = build_network(sim_mat, codes, cluster_df['Cluster_Semantic'].values, threshold_sem)
    html_se = render_pyvis(G_se)
    components.html(html_se, height=700)

st.success("Analysis complete!")
