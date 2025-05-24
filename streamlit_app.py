import streamlit as st
import pandas as pd
import numpy as np
import importlib
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pyvis.network import Network as PyvisNetwork
import streamlit.components.v1 as components
import warnings
from scipy.cluster.hierarchy import ClusterWarning

# Suppress unneeded SciPy warnings
warnings.filterwarnings("ignore", category=ClusterWarning)

# --- Page Config ---
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("MIDUS Codes Clustering & Semantic Networks")

# --- Caching Helpers ---
@st.cache_resource
def load_embedding_model():
    try:
        st.session_state._model_module = importlib.import_module("sentence_transformers")
        return getattr(st.session_state._model_module, "SentenceTransformer")("all-MiniLM-L6-v2")
    except ImportError as e:
        st.error(f"Could not load embeddings model: {e}")
        st.stop()

@st.cache_data
def compute_cooccurrence(df_codes: pd.DataFrame) -> np.ndarray:
    mat = df_codes.T.dot(df_codes).values
    np.fill_diagonal(mat, 0)
    return mat

@st.cache_data
def compute_semantics_and_clusters(
    df_codes: pd.DataFrame, n_clusters: int
):
    codes = df_codes.columns.tolist()
    co_mat = compute_cooccurrence(df_codes)

    # Semantic embed + similarity
    try:
        model = load_embedding_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
        sem_ok = True
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}\nSkipping semantic analysis.")
        embeddings = None
        sim_mat = None
        sem_ok = False

    # Agglomerative clusters
    cl_co = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    cl_sem = (
        AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        if sem_ok else np.full(len(codes), -1)
    )

    cluster_df = pd.DataFrame({
        "Code": codes,
        "CoCluster": cl_co,
        "SemCluster": cl_sem,
    })
    return co_mat, sim_mat, embeddings, cluster_df, sem_ok

@st.cache_data
def evaluate_validity(
    matrix: np.ndarray, embeddings: np.ndarray, ks: list[int]
) -> pd.DataFrame:
    rows = []
    for k in ks:
        # Co-occurrence validity
        labels_co = AgglomerativeClustering(n_clusters=k).fit_predict(matrix)
        sil_co = silhouette_score(matrix, labels_co)
        ch_co = calinski_harabasz_score(matrix, labels_co)
        db_co = davies_bouldin_score(matrix, labels_co)
        # Semantic validity
        labels_sem = AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)
        sil_sem = silhouette_score(embeddings, labels_sem, metric="cosine")
        ch_sem = calinski_harabasz_score(embeddings, labels_sem)
        db_sem = davies_bouldin_score(embeddings, labels_sem)

        rows.append({
            "k": k,
            "sil_co": sil_co,
            "ch_co": ch_co,
            "db_co": db_co,
            "sil_sem": sil_sem,
            "ch_sem": ch_sem,
            "db_sem": db_sem,
        })
    return pd.DataFrame(rows)

@st.cache_data
def build_nx_graph(
    matrix: np.ndarray, labels: list[str], clusters: np.ndarray, threshold: float
) -> nx.Graph:
    G = nx.Graph()
    unique_labels = np.unique(clusters)
    palette = sns.color_palette("hls", len(unique_labels))
    hex_colors = [mcolors.to_hex(c) for c in palette]

    for idx, code in enumerate(labels):
        color = (
            hex_colors[clusters[idx] % len(hex_colors)]
            if clusters[idx] >= 0 else "#cccccc"
        )
        G.add_node(code, color=color)

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            w = matrix[i, j]
            if w > threshold:
                G.add_edge(labels[i], labels[j], weight=float(w))
    return G

# --- Sidebar: Upload and Parameters ---
st.sidebar.header("1) Upload Data")
uploaded = st.sidebar.file_uploader("Excel (.xls/.xlsx)", type=["xls","xlsx"])
if not uploaded:
    st.info("Please upload an Excel file to proceed.")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet = st.sidebar.selectbox("Select Sheet", xls.sheet_names)
df_raw = pd.read_excel(xls, sheet_name=sheet)

# Discover _M2 / _M3 columns
codes_M2 = df_raw.filter(regex=r'(?i)_M2$').columns.tolist()
codes_M3 = df_raw.filter(regex=r'(?i)_M3$').columns.tolist()
if not codes_M2 or not codes_M3:
    st.error("Could not find `_M2` or `_M3` columns. Check your sheet naming.")
    st.stop()

# Convert to numeric
df2 = (
    df_raw[codes_M2]
    .replace({".": np.nan, " ": np.nan})
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype(int)
)
df3 = (
    df_raw[codes_M3]
    .replace({".": np.nan, " ": np.nan})
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype(int)
)

st.sidebar.header("2) M2 Settings")
k2 = st.sidebar.slider("Clusters (M2)", 2, 10, 5)
thr2_co = st.sidebar.slider("Co-occ Thr (M2)", 1, int(df2.values.max()), 5)
thr2_sem = st.sidebar.slider("Sem Thr (M2)", 0.0, 1.0, 0.4, 0.05)

st.sidebar.header("3) M3 Settings")
k3 = st.sidebar.slider("Clusters (M3)", 2, 10, 5)
thr3_co = st.sidebar.slider("Co-occ Thr (M3)", 1, int(df3.values.max()), 5)
thr3_sem = st.sidebar.slider("Sem Thr (M3)", 0.0, 1.0, 0.4, 0.05)

# --- Main Computation ---
with st.spinner("Computing clusters…"):
    co2, sim2, emb2, cl2_df, ok2 = compute_semantics_and_clusters(df2, k2)
    co3, sim3, emb3, cl3_df, ok3 = compute_semantics_and_clusters(df3, k3)
    ks = list(range(2, 9))
    val2 = evaluate_validity(co2, emb2, ks)
    val3 = evaluate_validity(co3, emb3, ks)

# --- Display Results ---
st.subheader("Cluster Validity (k=2…8)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**M2 Validity**")
    st.dataframe(val2, use_container_width=True)
with c2:
    st.markdown("**M3 Validity**")
    st.dataframe(val3, use_container_width=True)

st.subheader("Cluster Assignments")
t1, t2 = st.tabs(["M2 Assignments", "M3 Assignments"])
with t1:
    st.dataframe(cl2_df, use_container_width=True)
with t2:
    st.dataframe(cl3_df, use_container_width=True)

# --- Interactive Networks ---
def render_pyvis(G: nx.Graph):
    net = PyvisNetwork(height="600px", width="100%", directed=False)
    for n, d in G.nodes(data=True):
        net.add_node(str(n), label=str(n), color=d["color"])
    for u, v, d in G.edges(data=True):
        net.add_edge(str(u), str(v), value=d["weight"])
    net.repulsion(node_distance=120)
    components.html(net.generate_html(), height=600)

st.subheader("Networks")
st.markdown("**M2 Co-occurrence**")
render_pyvis(build_nx_graph(co2, df2.columns.tolist(), cl2_df["CoCluster"].values, thr2_co))
if ok2:
    st.markdown("**M2 Semantic**")
    render_pyvis(build_nx_graph(sim2, df2.columns.tolist(), cl2_df["SemCluster"].values, thr2_sem))

st.markdown("**M3 Co-occurrence**")
render_pyvis(build_nx_graph(co3, df3.columns.tolist(), cl3_df["CoCluster"].values, thr3_co))
if ok3:
    st.markdown("**M3 Semantic**")
    render_pyvis(build_nx_graph(sim3, df3.columns.tolist(), cl3_df["SemCluster"].values, thr3_sem))

st.success("All done!")
