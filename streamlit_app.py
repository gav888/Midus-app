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

    # Show cluster assignments
    st.subheader("M2 Cluster Assignments")
    st.dataframe(cluster_df_M2)
    st.download_button("Download M2 cluster assignments",
                       cluster_df_M2.to_csv(index=False), "clusters_M2.csv")
    st.subheader("M3 Cluster Assignments")
    st.dataframe(cluster_df_M3)
    st.download_button("Download M3 cluster assignments",
                       cluster_df_M3.to_csv(index=False), "clusters_M3.csv")

    # Show top similarities if available
    if sem_ok_M2:
        st.subheader("M2 Top Semantic Similarities")
        st.dataframe(sim_df_M2.head(15))
        st.download_button("Download M2 semantic similarities",
                           sim_df_M2.to_csv(index=False), "similarities_M2.csv")
    if sem_ok_M3:
        st.subheader("M3 Top Semantic Similarities")
        st.dataframe(sim_df_M3.head(15))
        st.download_button("Download M3 semantic similarities",
                           sim_df_M3.to_csv(index=False), "similarities_M3.csv")

    # Networks
    st.subheader("Networks Visualization")
    colA, colB = st.columns(2)
    # Co-occurrence network
    with colA:
        st.markdown("**Co-occurrence Network**")
        Gc = build_network(co_mat_M2, codes, cluster_df_M2['Cluster_Cooccurrence'].values, threshold_cooc)
        pos = nx.spring_layout(Gc, seed=42)
        fig, ax = plt.subplots(figsize=(5,5))
        nx.draw(Gc, pos, with_labels=True,
                node_color=[Gc.nodes[n]['color'] for n in Gc.nodes()],
                edge_color='lightblue', width=2, alpha=0.7, ax=ax)
        ax.axis('off')
        st.pyplot(fig)
    # Semantic network if available
    if sem_ok_M2:
        with colB:
            st.markdown("**Semantic Similarity Network**")
            Gs = build_network(sim_mat_M2, codes, cluster_df_M2['Cluster_Semantic'].values, threshold_sem)
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

    st.markdown("**M3 Networks**")
    colC, colD = st.columns(2)
    with colC:
        st.markdown("**M3 Co-occurrence Network**")
        Gc_M3 = build_network(co_mat_M3, codes_M3,
                              cluster_df_M3['Cluster_Cooccurrence'].values,
                              threshold_cooc)
        pos3 = nx.spring_layout(Gc_M3, seed=42)
        fig3, ax3 = plt.subplots(figsize=(5,5))
        nx.draw(Gc_M3, pos3, with_labels=True,
                node_color=[Gc_M3.nodes[n]['color'] for n in Gc_M3.nodes()],
                edge_color='lightgreen', width=2, alpha=0.7, ax=ax3)
        ax3.axis('off')
        st.pyplot(fig3)
    if sem_ok_M3:
        with colD:
            st.markdown("**M3 Semantic Similarity Network**")
            Gs_M3 = build_network(sim_mat_M3, codes_M3,
                                  cluster_df_M3['Cluster_Semantic'].values,
                                  threshold_sem)
            pos4 = nx.kamada_kawai_layout(Gs_M3)
            fig4, ax4 = plt.subplots(figsize=(5,5))
            edges4 = list(Gs_M3.edges(data=True))
            weights4 = [attr['weight'] for *_, attr in edges4]
            if weights4:
                wmin4, wmax4 = min(weights4), max(weights4)
                for u,v,attr in edges4:
                    w = attr['weight']; norm=(w-wmin4)/(wmax4-wmin4) if wmax4>wmin4 else 1
                    nx.draw_networkx_edges(Gs_M3, pos4, edgelist=[(u,v)],
                        width=0.5+3*norm, alpha=0.3+0.7*norm,
                        edge_color='#888888', ax=ax4)
            nx.draw_networkx_nodes(Gs_M3, pos4,
                node_size=[300+20*Gs_M3.degree(n) for n in Gs_M3.nodes()],
                node_color=[Gs_M3.nodes[n]['color'] for n in Gs_M3.nodes()],
                ax=ax4)
            nx.draw_networkx_labels(Gs_M3, pos4, font_size=8, ax=ax4)
            ax4.axis('off')
            st.pyplot(fig4)

    # Inter-time network similarity
    def binary_adj_edges(G): return set(frozenset(e) for e in G.edges())
    def jaccard(edges1, edges2):
        if not edges1 and not edges2: return 1.0
        if not edges1 or not edges2: return 0.0
        return len(edges1 & edges2) / len(edges1 | edges2)
    edges_cooc_M2 = binary_adj_edges(Gc)
    edges_cooc_M3 = binary_adj_edges(Gc_M3)
    j_cooc = jaccard(edges_cooc_M2, edges_cooc_M3)
    st.subheader("Inter-time Network Similarity (Jaccard)")
    st.metric("Co-occurrence Network", f"{j_cooc:.3f}")
    if sem_ok_M2 and sem_ok_M3:
        edges_sem_M2 = binary_adj_edges(Gs)
        edges_sem_M3 = binary_adj_edges(Gs_M3)
        j_sem = jaccard(edges_sem_M2, edges_sem_M3)
        st.metric("Semantic Network", f"{j_sem:.3f}")

    st.success("Analysis complete!")    
