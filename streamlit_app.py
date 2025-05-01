###############################################################################
#  MIDUS Codes Clustering & Semantic Networks — Sub-group Edition (May 2025)
#  --------------------------------------------------------------------------
#  • Upload an Excel file that contains binary MIDUS code columns ending in
#    “_M2” plus any grouping/covariate columns you wish.
#  • Choose a grouping variable and two levels (A vs B) to compare.
#  • The app shows side-by-side co-occurrence & semantic networks, cluster
#    tables, and an optional edge-difference network (A – B).  Fisher p-values
#    are provided for quick checks.
#
#  *** 2025-05-01 PATCHES ***
#    (1) Suppress Streamlit’s file-watcher ↔ torch bug via ST_FILE_WATCHER_DISABLED
#    (2) Wrap Hugging Face login in a safe try/except, fall back to anonymous
#        model download if the token is missing / invalid / rate-limited
#    (3) Import sentence-transformers lazily inside the cached loader
###############################################################################

# -------------------------------------------------------------------- #
# 0. Environment patch **before** importing streamlit
# -------------------------------------------------------------------- #
import os
os.environ["ST_FILE_WATCHER_DISABLED"] = "true"

# -------------------------------------------------------------------- #
# 1. Imports
# -------------------------------------------------------------------- #
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
# 2. Safe, optional Hugging-Face authentication
# -------------------------------------------------------------------- #
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        st.sidebar.success("Hugging Face authentication ✓")
    except Exception as e:   # covers invalid token & 429 rate-limit
        st.sidebar.warning(f"HF login failed ({e}). Continuing anonymously.")
else:
    st.sidebar.info("No Hugging Face token supplied – anonymous mode.")

# -------------------------------------------------------------------- #
# 3. Streamlit page config
# -------------------------------------------------------------------- #
st.set_page_config(
    page_title="MIDUS Codes Clustering & Semantic Networks",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("MIDUS Codes – Clustering & Semantic Networks (Sub-group mode)")

# -------------------------------------------------------------------- #
# 4. Sentence-Transformer loader (lazy import prevents torch watcher bug)
# -------------------------------------------------------------------- #
def _import_sentence_transformers():
    return importlib.import_module("sentence_transformers").SentenceTransformer

@st.cache_resource(show_spinner=False)
def load_model():
    ST = _import_sentence_transformers()
    return ST("all-MiniLM-L6-v2")

# -------------------------------------------------------------------- #
# 5. Core analytic helpers
# -------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def compute_semantics_and_clusters(df_codes: pd.DataFrame, n_clusters: int):
    """Return co-occurrence, cosine-similarity matrices, cluster labels, etc."""
    codes = df_codes.columns.tolist()

    # Co-occurrence
    co_mat = df_codes.T.dot(df_codes).values
    np.fill_diagonal(co_mat, 0)

    # Semantic similarity
    try:
        model = load_model()
        embeddings = model.encode(codes)
        sim_mat = cosine_similarity(embeddings)
        sem_ok = True
    except Exception as e:
        st.warning(f"Semantic embedding failed: {e}. Skipping semantic analysis.")
        embeddings, sim_mat, sem_ok = None, None, False

    # Clustering
    clust_co = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(co_mat)
    clust_sem = (
        AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        if sem_ok else np.full(len(codes), -1)
    )

    cluster_df = pd.DataFrame({
        "Code": codes,
        "Cluster_Cooccurrence": clust_co,
        "Cluster_Semantic": clust_sem
    })

    if sem_ok:
        tri = [(codes[i], codes[j], float(sim_mat[i, j]))
               for i in range(len(codes)) for j in range(i + 1, len(codes))]
        sim_df = (
            pd.DataFrame(tri, columns=["Code1", "Code2", "CosineSimilarity"])
            .sort_values("CosineSimilarity", ascending=False)
            .reset_index(drop=True)
        )
    else:
        sim_df = pd.DataFrame(columns=["Code1", "Code2", "CosineSimilarity"])

    return co_mat, sim_mat, cluster_df, sim_df, sem_ok


@st.cache_data(show_spinner=False)
def build_network(matrix, labels, clusters, threshold):
    G = nx.Graph()
    unique = np.unique(clusters)
    palette = sns.color_palette("hls", max(len(unique), 2))
    colors = [matplotlib.colors.to_hex(c) for c in palette]

    for i, lab in enumerate(labels):
        col = colors[0] if clusters[i] < 0 else colors[clusters[i] % len(colors)]
        G.add_node(lab, color=col, degree=0)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            w = matrix[i, j]
            if w > threshold:
                G.add_edge(labels[i], labels[j], weight=float(w))
                G.nodes[labels[i]]["degree"] += 1
                G.nodes[labels[j]]["degree"] += 1
    return G


def render_pyvis(G, height="700px", width="100%"):
    net = Network(height=height, width=width, notebook=False)
    net.force_atlas_2based()
    for n, d in G.nodes(data=True):
        net.add_node(n, label=n, color=d["color"], title=f"Degree: {d['degree']}")
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=int(d["weight"]), color=d.get("color"))
    path = f"network_{id(G)}.html"
    net.save_graph(path)
    return open(path, "r", encoding="utf-8").read()


def difference_network(mat_A, mat_B, labels, thresh):
    diff = mat_A - mat_B
    G = nx.Graph()
    for lab in labels:
        G.add_node(lab, color="#dddddd")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            delta = diff[i, j]
            if abs(delta) > thresh:
                G.add_edge(
                    labels[i], labels[j],
                    weight=abs(delta),
                    color="#2ECC71" if delta > 0 else "#E74C3C",
                    title=f"Δ={delta:+.3f}"
                )
    return G

# -------------------------------------------------------------------- #
# 6. File upload and basic preprocessing
# -------------------------------------------------------------------- #
st.sidebar.header("Upload & settings")
file = st.sidebar.file_uploader("Excel file (.xlsx/.xls)", ["xlsx", "xls"])
if not file:
    st.info("Upload your MIDUS coding Excel file to start.")
    st.stop()

df_raw = pd.read_excel(file)
codes = [c for c in df_raw.columns if c.endswith("_M2")]
if not codes:
    st.error("No ‘_M2’ code columns found.")
    st.stop()

def make_code_matrix(df):
    return (
        df[codes]
        .replace({".": np.nan, " ": np.nan})
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0).astype(int)
    )

# -------------------------------------------------------------------- #
# 7. Sidebar – analysis parameters
# -------------------------------------------------------------------- #
n_clusters = st.sidebar.slider("# clusters", 2, 10, 5)
th_co = st.sidebar.slider("Co-occurrence threshold", 1, 20, 5)

grp_var = st.sidebar.selectbox(
    "Grouping variable (sub-group comparison)",
    [None] + [c for c in df_raw.columns if c not in codes],
    index=0
)
if grp_var:
    lvls = sorted(df_raw[grp_var].dropna().unique())
    if len(lvls) < 2:
        st.error(f"‘{grp_var}’ has fewer than two levels.")
        st.stop()
    lvl_A = st.sidebar.selectbox("Sub-group A", lvls, index=0)
    lvl_B = st.sidebar.selectbox("Sub-group B", [l for l in lvls if l != lvl_A], index=0)
else:
    lvl_A = lvl_B = None

# -------------------------------------------------------------------- #
# 8. Sub-group data frames
# -------------------------------------------------------------------- #
if grp_var:
    df_A = df_raw[df_raw[grp_var] == lvl_A]
    df_B = df_raw[df_raw[grp_var] == lvl_B]
    if df_A.empty or df_B.empty:
        st.error("One of the selected sub-groups is empty.")
        st.stop()
else:
    df_A = df_B = df_raw

df_codes_A = make_code_matrix(df_A)
df_codes_B = make_code_matrix(df_B)

# -------------------------------------------------------------------- #
# 9. Summary metrics
# -------------------------------------------------------------------- #
st.subheader("Dataset snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows (file)", len(df_raw))
c2.metric("Code columns", len(codes))
c3.metric("Total ‘1’s", int(make_code_matrix(df_raw).values.sum()))
with st.expander("View code matrix (full data)"):
    st.dataframe(make_code_matrix(df_raw))

# -------------------------------------------------------------------- #
# 10. Compute networks
# -------------------------------------------------------------------- #
with st.spinner("Computing networks…"):
    co_A, sim_A, clust_A, simdf_A, sem_A = compute_semantics_and_clusters(
        df_codes_A, n_clusters
    )
    co_B, sim_B, clust_B, simdf_B, sem_B = compute_semantics_and_clusters(
        df_codes_B, n_clusters
    )

# -------------------------------------------------------------------- #
# 11. Side-by-side tabs
# -------------------------------------------------------------------- #
tab_A, tab_B = st.tabs([f"{lvl_A or 'All'} (A)", f"{lvl_B or 'All'} (B)"])

with tab_A:
    st.subheader(f"Co-occurrence network — {lvl_A or 'all data'}")
    G_A = build_network(co_A, codes, clust_A["Cluster_Cooccurrence"].values, th_co)
    components.html(render_pyvis(G_A), height=700)
    st.subheader("Cluster assignments")
    st.dataframe(clust_A)
    if sem_A:
        th_sem_A = st.slider("Semantic similarity threshold (A)",
                             0.0, 1.0, 0.4, 0.05, key="semA")
        st.subheader("Semantic network")
        G_A_sem = build_network(sim_A, codes,
                                clust_A["Cluster_Semantic"].values, th_sem_A)
        components.html(render_pyvis(G_A_sem), height=700)

with tab_B:
    st.subheader(f"Co-occurrence network — {lvl_B or 'all data'}")
    G_B = build_network(co_B, codes, clust_B["Cluster_Cooccurrence"].values, th_co)
    components.html(render_pyvis(G_B), height=700)
    st.subheader("Cluster assignments")
    st.dataframe(clust_B)
    if sem_B:
        th_sem_B = st.slider("Semantic similarity threshold (B)",
                             0.0, 1.0, 0.4, 0.05, key="semB")
        st.subheader("Semantic network")
        G_B_sem = build_network(sim_B, codes,
                                clust_B["Cluster_Semantic"].values, th_sem_B)
        components.html(render_pyvis(G_B_sem), height=700)

# -------------------------------------------------------------------- #
# 12. Edge-difference network (A – B)
# -------------------------------------------------------------------- #
if grp_var:
    st.subheader("Edge-difference network (A – B)")
    delta_th = st.slider("Δ-edge threshold", 0.0, 1.0, 0.25, 0.05)
    G_d = difference_network(co_A, co_B, codes, delta_th)
    components.html(render_pyvis(G_d), height=700)

    from scipy.stats import fisher_exact

    def edge_pvals(df1, df2):
        rows = []
        for i, c1 in enumerate(codes):
            for j in range(i + 1, len(codes)):
                c2 = codes[j]
                a11 = (df1[c1] & df1[c2]).sum()
                a1_ = df1[c1].sum()
                a_1 = df1[c2].sum()
                b11 = (df2[c1] & df2[c2]).sum()
                b1_ = df2[c1].sum()
                b_1 = df2[c2].sum()
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
                           p_df.to_csv(index=False),
                           "edge_pvals.csv")

st.success("Analysis complete ✓")
