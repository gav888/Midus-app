```python
###############################################################################
#  MIDUS Codes Clustering & Semantic Networks — with Sub-group Comparison
#  ----------------------------------------------------------------------
#  • Upload an Excel file that contains binary MIDUS code columns ending
#    in '_M2' plus any grouping / covariate columns you like.
#  • Choose a grouping variable and two levels to compare.
#  • The app shows side-by-side networks, cluster tables, and an optional
#    edge-difference network (A – B).  Fisher’s exact p-values are provided
#    for quick statistical checks.
###############################################################################

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

# -------------------------------------------------------------------- #
# 0. Authentication & page config
# -------------------------------------------------------------------- #
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("MIDUS Codes Clustering & Semantic Networks (Sub-group mode)")

# -------------------------------------------------------------------- #
# 1. Helpers
# -------------------------------------------------------------------- #
def _import_sentence_transformers():
    module = importlib.import_module("sentence_transformers")
    return getattr(module, "SentenceTransformer")

@st.cache_resource(show_spinner=False)
def load_model():
    SentenceTransformer = _import_sentence_transformers()
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def compute_semantics_and_clusters(df_codes: pd.DataFrame, n_clusters: int):
    """Return co-occurrence, semantic-similarity matrices & cluster labels."""
    codes = df_codes.columns.tolist()

    # 1 Co-occurrence
    co_mat = df_codes.T.dot(df_codes).values
    np.fill_diagonal(co_mat, 0)

    # 2 Embeddings & cosine sim
    try:
        model = load_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
        semantic_ok = True
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}. Skipping semantic analysis.")
        embeddings = None
        sim_mat = None
        semantic_ok = False

    # 3 Hierarchical clustering
    clust_co = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    clust_sem = (
        AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        if semantic_ok else np.full(len(codes), -1)
    )

    cluster_df = pd.DataFrame(
        {"Code": codes, "Cluster_Cooccurrence": clust_co, "Cluster_Semantic": clust_sem}
    )

    if semantic_ok:
        tri = [(codes[i], codes[j], float(sim_mat[i, j]))
               for i in range(len(codes)) for j in range(i + 1, len(codes))]
        sim_df = (
            pd.DataFrame(tri, columns=["Code1", "Code2", "CosineSimilarity"])
            .sort_values("CosineSimilarity", ascending=False)
            .reset_index(drop=True)
        )
    else:
        sim_df = pd.DataFrame(columns=["Code1", "Code2", "CosineSimilarity"])

    return co_mat, sim_mat, cluster_df, sim_df, semantic_ok

@st.cache_data(show_spinner=False)
def build_network(matrix: np.ndarray, labels: list, clusters: np.ndarray,
                  threshold: float):
    """Undirected NetworkX graph thresholded by edge weight."""
    G = nx.Graph()
    unique = np.unique(clusters)
    palette = sns.color_palette("hls", max(len(unique), 2))
    colors = [matplotlib.colors.to_hex(c) for c in palette]

    for i, label in enumerate(labels):
        color = colors[0] if clusters[i] < 0 else colors[clusters[i] % len(colors)]
        G.add_node(label, color=color, degree=0)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            w = matrix[i, j]
            if w > threshold:
                G.add_edge(labels[i], labels[j], weight=float(w))
                G.nodes[labels[i]]["degree"] += 1
                G.nodes[labels[j]]["degree"] += 1
    return G

def render_pyvis(G: nx.Graph, height="700px", width="100%") -> str:
    """Return rendered PyVis network as HTML string."""
    net = Network(height=height, width=width, notebook=False)
    net.force_atlas_2based()
    for node, data in G.nodes(data=True):
        net.add_node(
            node, label=node, color=data["color"], title=f"Degree: {data['degree']}"
        )
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=int(d["weight"]), color=d.get("color"))
    path = f"network_{id(G)}.html"
    net.save_graph(path)
    return open(path, "r", encoding="utf-8").read()

def difference_network(mat_A, mat_B, labels, thresh):
    """Edge-difference network: positive Δ (green) stronger in A, negative Δ (red)."""
    diff = mat_A - mat_B
    G = nx.Graph()
    for lbl in labels:
        G.add_node(lbl, color="#dddddd")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            delta = diff[i, j]
            if abs(delta) > thresh:
                G.add_edge(
                    labels[i],
                    labels[j],
                    weight=abs(delta),
                    color="#2ECC71" if delta > 0 else "#E74C3C",
                    title=f"Δ = {delta:+.3f}",
                )
    return G

# -------------------------------------------------------------------- #
# 2. Load data
# -------------------------------------------------------------------- #
st.sidebar.header("Upload & basic settings")
file = st.sidebar.file_uploader("Excel file (.xlsx / .xls)", ["xlsx", "xls"])
if not file:
    st.info("Upload your MIDUS coding Excel file to start.")
    st.stop()

raw_df = pd.read_excel(file)
codes = [c for c in raw_df.columns if c.endswith("_M2")]
if not codes:
    st.error("No columns ending with '_M2' found.")
    st.stop()

# -------------------------------------------------------------------- #
# 3. Sidebar – analysis parameters
# -------------------------------------------------------------------- #
n_clusters = st.sidebar.slider("# Clusters", 2, 10, 5)
threshold_cooc = st.sidebar.slider("Co-occurrence threshold", 1, 20, 5)

# Sub-group controls
group_var = st.sidebar.selectbox(
    "Grouping variable (for sub-group comparison)",
    [None] + [c for c in raw_df.columns if c not in codes],
    index=0,
)

if group_var:
    levels = sorted(raw_df[group_var].dropna().unique())
    if len(levels) < 2:
        st.error(f"Grouping variable '{group_var}' has fewer than 2 levels.")
        st.stop()
    lvl_a = st.sidebar.selectbox("Sub-group A", levels, index=0)
    lvl_b = st.sidebar.selectbox(
        "Sub-group B", [l for l in levels if l != lvl_a], index=0
    )
else:
    lvl_a = lvl_b = None

# -------------------------------------------------------------------- #
# 4. Utility to convert raw rows into 0/1 code matrix
# -------------------------------------------------------------------- #
def make_code_matrix(_df):
    return (
        _df[codes]
        .replace({".": np.nan, " ": np.nan})
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

# -------------------------------------------------------------------- #
# 5. Prepare sub-sets
# -------------------------------------------------------------------- #
if group_var:
    df_A = raw_df[raw_df[group_var] == lvl_a]
    df_B = raw_df[raw_df[group_var] == lvl_b]
    if df_A.empty or df_B.empty:
        st.error("One of the chosen sub-groups is empty; pick different levels.")
        st.stop()
else:
    df_A = df_B = raw_df

df_codes_A = make_code_matrix(df_A)
df_codes_B = make_code_matrix(df_B)

# -------------------------------------------------------------------- #
# 6. Summary metrics for the uploaded data
# -------------------------------------------------------------------- #
st.subheader("Data snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows (full file)", raw_df.shape[0])
c2.metric("Code columns", len(codes))
c3.metric("Total 1s in codes", int(make_code_matrix(raw_df).values.sum()))

with st.expander("View code matrix (full sample)"):
    st.dataframe(make_code_matrix(raw_df))

# -------------------------------------------------------------------- #
# 7. Run clustering & similarity for each sub-group
# -------------------------------------------------------------------- #
with st.spinner("Computing networks..."):
    co_A, sim_A, clust_A, simdf_A, sem_ok_A = compute_semantics_and_clusters(
        df_codes_A, n_clusters
    )
    co_B, sim_B, clust_B, simdf_B, sem_ok_B = compute_semantics_and_clusters(
        df_codes_B, n_clusters
    )

# -------------------------------------------------------------------- #
# 8. Visual — side-by-side tabs
# -------------------------------------------------------------------- #
tabA, tabB = st.tabs(
    [f"{lvl_a or 'All'} (A)", f"{lvl_b or 'All'} (B)"]
)

# A ------------------------------------------------------------------ #
with tabA:
    st.subheader(f"Co-occurrence network — {lvl_a or 'all rows'}")
    G_A = build_network(co_A, codes, clust_A["Cluster_Cooccurrence"].values,
                        threshold_cooc)
    components.html(render_pyvis(G_A), height=700)
    st.subheader("Cluster assignments")
    st.dataframe(clust_A)

    if sem_ok_A:
        threshold_sem_A = st.slider(
            "Semantic similarity threshold (A)", 0.0, 1.0, 0.4, 0.05, key="th_sem_A"
        )
        st.subheader("Semantic network")
        G_A_sem = build_network(
            sim_A, codes, clust_A["Cluster_Semantic"].values, threshold_sem_A
        )
        components.html(render_pyvis(G_A_sem), height=700)

# B ------------------------------------------------------------------ #
with tabB:
    st.subheader(f"Co-occurrence network — {lvl_b or 'all rows'}")
    G_B = build_network(co_B, codes, clust_B["Cluster_Cooccurrence"].values,
                        threshold_cooc)
    components.html(render_pyvis(G_B), height=700)
    st.subheader("Cluster assignments")
    st.dataframe(clust_B)

    if sem_ok_B:
        threshold_sem_B = st.slider(
            "Semantic similarity threshold (B)", 0.0, 1.0, 0.4, 0.05, key="th_sem_B"
        )
        st.subheader("Semantic network")
        G_B_sem = build_network(
            sim_B, codes, clust_B["Cluster_Semantic"].values, threshold_sem_B
        )
        components.html(render_pyvis(G_B_sem), height=700)

# -------------------------------------------------------------------- #
# 9. Edge-difference network (A – B)
# -------------------------------------------------------------------- #
if group_var:
    st.subheader("Edge-difference network (A minus B)")
    delta_thresh = st.slider("Δ-edge threshold", 0.0, 1.0, 0.25, 0.05)
    G_delta = difference_network(co_A, co_B, codes, delta_thresh)
    components.html(render_pyvis(G_delta), height=700)

    # Optional Fisher exact tests (quick scan)
    from scipy.stats import fisher_exact

    def edge_pvals(dfA, dfB):
        rows = []
        for i, c1 in enumerate(codes):
            for j in range(i + 1, len(codes)):
                c2 = codes[j]
                # 2×2 table
                a11 = (dfA[c1] & dfA[c2]).sum()
                a1_ = dfA[c1].sum()
                a_1 = dfA[c2].sum()
                b11 = (dfB[c1] & dfB[c2]).sum()
                b1_ = dfB[c1].sum()
                b_1 = dfB[c2].sum()
                table = np.array([[a11, a1_ + a_1 - 2 * a11],
                                  [b11, b1_ + b_1 - 2 * b11]])
                try:
                    _, p = fisher_exact(table)
                except Exception:
                    p = np.nan
                rows.append((c1, c2, p))
        return (
            pd.DataFrame(rows, columns=["Code1", "Code2", "p_value"])
            .sort_values("p_value")
            .reset_index(drop=True)
        )

    with st.expander("Edge-wise Fisher exact p-values"):
        p_df = edge_pvals(df_codes_A, df_codes_B)
        st.dataframe(p_df.head(30))
        st.download_button("Download full p-value table",
                           p_df.to_csv(index=False), "edge_pvals.csv")

st.success("Analysis complete! 🎉")
```
