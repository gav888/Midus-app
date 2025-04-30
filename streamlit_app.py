import streamlit as st
import io
import pandas as pd
import importlib
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.colors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from pyvis.network import Network
import streamlit.components.v1 as components

# 1) Log in to HuggingFace so SentenceTransformer can download
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

# --- Page config ---
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("MIDUS Codes Clustering & Semantic Networks")

# --- Helpers & Caching ---
@st.cache_resource
def load_model():
    """Lazy-load SentenceTransformer inside cache."""
    s2 = importlib.import_module('sentence_transformers')
    SentenceTransformer = getattr(s2, 'SentenceTransformer')
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def compute_semantics_and_clusters(df_codes: pd.DataFrame, n_clusters: int):
    codes = df_codes.columns.tolist()

    # Co-occurrence
    co_mat = np.dot(df_codes.T, df_codes)
    np.fill_diagonal(co_mat, 0)

    # Semantic embeddings + similarity
    semantic_ok = True
    try:
        model = load_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}\nSkipping semantic.")
        semantic_ok = False
        sim_mat = None

    # Clustering
    clust_cooc = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    if semantic_ok:
        clust_sem = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    else:
        clust_sem = np.array([-1]*len(codes))

    # DataFrames
    cluster_df = pd.DataFrame({
        'Code': codes,
        'Cluster_Cooccurrence': clust_cooc,
        'Cluster_Semantic':      clust_sem
    })
    if semantic_ok:
        pairs = [(codes[i], codes[j], sim_mat[i,j])
                 for i in range(len(codes)) for j in range(i+1, len(codes))]
        sim_df = (pd.DataFrame(pairs, columns=['Code1','Code2','CosineSimilarity'])
                  .sort_values('CosineSimilarity', ascending=False)
                  .reset_index(drop=True))
    else:
        sim_df = pd.DataFrame(columns=['Code1','Code2','CosineSimilarity'])

    return co_mat, sim_mat, cluster_df, sim_df, semantic_ok

def build_network(matrix, labels, clusters, threshold):
    G = nx.Graph()
    unique_clusters = np.unique(clusters)
    palette = sns.color_palette("hls", max(len(unique_clusters),2))
    hex_colors = [matplotlib.colors.to_hex(c) for c in palette]

    # Nodes
    for idx,label in enumerate(labels):
        color = hex_colors[0] if clusters[idx]<0 else hex_colors[clusters[idx]%len(hex_colors)]
        G.add_node(label, color=color, degree=0)

    # Edges
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            w = matrix[i,j]
            if w>threshold:
                G.add_edge(labels[i], labels[j], weight=w)
                G.nodes[labels[i]]['degree'] += 1
                G.nodes[labels[j]]['degree'] += 1

    return G

def draw_pyvis(G, height="600px", width="100%"):
    """Convert a networkx graph into a pyvis.Network for embedding."""
    net = Network(height=height, width=width, notebook=False)
    net.force_atlas_2based()
    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=node,
            color=data.get("color"),
            title=f"Degree: {data.get('degree',0)}"
        )
    for u,v,attr in G.edges(data=True):
        net.add_edge(u, v, value=attr.get("weight",1))
    return net

# --- Sidebar Inputs ---
st.sidebar.header("Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx/.xls)", type=['xlsx','xls'])
n_clusters     = st.sidebar.slider("# Clusters", 2, 10, 5)
threshold_cooc = st.sidebar.slider("Co-occurrence threshold", 1, 20, 5)

# --- Main ---
if not uploaded_file:
    st.info("Please upload your MIDUS coding Excel file.")
    st.stop()

df = pd.read_excel(uploaded_file)
codes = [c for c in df.columns if c.endswith('_M2')]
if not codes:
    st.error("No columns ending in '_M2' found.")
    st.stop()

df_codes = (df[codes]
             .replace({'.':np.nan,' ':np.nan})
             .apply(pd.to_numeric, errors='coerce')
             .fillna(0).astype(int)
)

# Show summary
st.subheader("Data Snapshot")
c1,c2,c3 = st.columns(3)
c1.metric("Rows", df_codes.shape[0])
c2.metric("_M2 Codes", df_codes.shape[1])
c3.metric("Total Codes", int(df_codes.values.sum()))
with st.expander("View raw code matrix"):
    st.dataframe(df_codes)

# Compute clusters & similarities
with st.spinner("Computing…"):
    co_mat, sim_mat, cluster_df, sim_df, sem_ok = compute_semantics_and_clusters(df_codes, n_clusters)

# Show tables
st.subheader("Cluster Assignments")
st.dataframe(cluster_df)
st.download_button("Download clusters", cluster_df.to_csv(index=False), "clusters.csv")

if sem_ok:
    st.subheader("Top Semantic Similarities")
    st.dataframe(sim_df.head(15))
    st.download_button("Download similarities", sim_df.to_csv(index=False), "similarities.csv")
    threshold_sem = st.sidebar.slider("Semantic similarity threshold", 0.0, 1.0, 0.4, step=0.05)
else:
    threshold_sem = None

# Draw interactive networks
st.subheader("Interactive Network Visualizations")
colA, colB = st.columns(2)

# Co-occurrence
with colA:
    st.markdown("**Co-occurrence Network**")
    Gc = build_network(co_mat, codes, cluster_df['Cluster_Cooccurrence'], threshold_cooc)
    net1 = draw_pyvis(Gc)
    net1.save_graph("cooc.html")
    html1 = open("cooc.html", "r", encoding="utf-8").read()
    components.html(html1, height=650, width=650)

# Semantic
if sem_ok:
    with colB:
        st.markdown("**Semantic Similarity Network**")
        Gs = build_network(sim_mat, codes, cluster_df['Cluster_Semantic'], threshold_sem)
        net2 = draw_pyvis(Gs)
        net2.save_graph("sem.html")
        html2 = open("sem.html", "r", encoding="utf-8").read()
        components.html(html2, height=650, width=650)

st.success("Analysis complete!")
