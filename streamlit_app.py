import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples

# Utility functions

def compute(df_codes, k):
    codes = df_codes.columns.tolist()
    co = df_codes.T.dot(df_codes).values
    np.fill_diagonal(co, 0)
    # semantic embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode(codes)
        sim = cosine_similarity(emb)
        ok = True
    except Exception:
        emb, sim, ok = None, None, False
    cco = AgglomerativeClustering(n_clusters=k).fit_predict(co)
    cse = (AgglomerativeClustering(n_clusters=k).fit_predict(emb)
           if ok else np.full(len(codes), -1))
    dfc = pd.DataFrame({'Code': codes, 'CoCluster': cco, 'SemCluster': cse})
    pal = sns.color_palette('hls', max(2, k))
    dfc['color'] = [mcolors.to_hex(pal[i % len(pal)]) for i in cco]
    return co, sim, emb, dfc, ok


def evaluate(matrix, emb, ks):
    rows = []
    for k in ks:
        lc = AgglomerativeClustering(n_clusters=k).fit_predict(matrix)
        sil_sem = db_sem = None
        if emb is not None:
            ls = AgglomerativeClustering(n_clusters=k).fit_predict(emb)
            sil_sem = silhouette_score(emb, ls)
            db_sem = davies_bouldin_score(emb, ls)
        rows.append({
            'k': k,
            'sil_co': silhouette_score(matrix, lc),
            'ch_co': calinski_harabasz_score(matrix, lc),
            'db_co': davies_bouldin_score(matrix, lc),
            'sil_sem': sil_sem,
            'db_sem': db_sem
        })
    return pd.DataFrame(rows)


def make_graph(matrix, labels, clusters, thr):
    G = nx.Graph()
    pal = sns.color_palette('hls', len(np.unique(clusters)))
    cols = [mcolors.to_hex(pal[i % len(pal)]) for i in clusters]
    for i, l in enumerate(labels):
        G.add_node(l, color=cols[i])
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if matrix[i, j] > thr:
                G.add_edge(labels[i], labels[j], weight=float(matrix[i, j]))
    return G

# Streamlit App
st.set_page_config(page_title="MIDUS Codes Clustering", layout="wide")
st.title("MIDUS Codes Clustering & Networks")

# File uploader
uploaded = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])
if uploaded:
    sheets = pd.ExcelFile(uploaded).sheet_names
    sheet = st.selectbox("Choose sheet", sheets)
    df = pd.read_excel(uploaded, sheet_name=sheet)
    
    # Select columns
    cols2 = df.filter(regex=r'(?i)_M2$').columns
    cols3 = df.filter(regex=r'(?i)_M3$').columns
    df2 = df[cols2].replace({'.': np.nan, ' ': np.nan}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df3 = df[cols3].replace({'.': np.nan, ' ': np.nan}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Parameters
    k2 = st.slider("# clusters (M2 & M3)", 2, 10, 5)
    thr2_co = st.slider("Co-occ threshold", 1, 100, 5)
    thr2_sem = st.slider("Sem threshold", 0.0, 1.0, 0.4, step=0.05)

    if st.button("Run Analysis"):
        co2, sim2, emb2, cl2, ok2 = compute(df2, k2)
        co3, sim3, emb3, cl3, ok3 = compute(df3, k2)

        tabs = st.tabs(["Validity", "Assignments", "Network", "M2 Internal", "Fit M2→M3"])

        # Validity Tab
        with tabs[0]:
            ks = list(range(2, 9))
            val2 = evaluate(co2, emb2, ks)
            st.subheader("Clustering Validity (M2)")
            st.dataframe(val2)

        # Assignments Tab
        with tabs[1]:
            st.subheader("Code Assignments")
            st.dataframe(cl2)

        # Network Tab
        with tabs[2]:
            st.subheader("Co-occurrence Network (M2)")
            fig, ax = plt.subplots()
            G = make_graph(co2, df2.columns.tolist(), cl2['CoCluster'], thr2_co)
            pos = nx.spring_layout(G, seed=1)
            colors = [d['color'] for _, d in G.nodes(data=True)]
            nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
            st.pyplot(fig)

        # M2 Internal Tab
        with tabs[3]:
            st.subheader("M2 Internal Analysis")
            val2 = evaluate(co2, emb2, ks)
            st.line_chart(val2.set_index('k')[['sil_co', 'db_co']])
            if emb2 is not None:
                st.line_chart(val2.set_index('k')[['sil_sem', 'db_sem']])

        # Fit M2→M3 Tab
        with tabs[4]:
            st.subheader("Fit M2 → M3 Analysis")
            metrics = {
                'sil_co': silhouette_score(co3, cl2['CoCluster']),
                'ch_co': calinski_harabasz_score(co3, cl2['CoCluster']),
                'db_co': davies_bouldin_score(co3, cl2['CoCluster'])
            }
            if ok3:
                metrics.update({
                    'sil_sem': silhouette_score(emb3, cl2['SemCluster'], metric='cosine'),
                    'ch_sem': calinski_harabasz_score(emb3, cl2['SemCluster']),
                    'db_sem': davies_bouldin_score(emb3, cl2['SemCluster'])
                })
            st.json(metrics)

            st.subheader("Contingency Heatmaps")
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(pd.crosstab(cl2['CoCluster'], cl3['CoCluster']), annot=True, fmt='d', ax=axs[0])
            sns.heatmap(pd.crosstab(cl2['SemCluster'], cl3['SemCluster']), annot=True, fmt='d', ax=axs[1])
            st.pyplot(fig)

            st.subheader("Silhouette Distributions")
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            sil_co_vals = silhouette_samples(co3, cl2['CoCluster'])
            for c in sorted(cl2['CoCluster'].unique()):
                axs[0].hist(sil_co_vals[cl2['CoCluster'] == c], bins=20, alpha=0.5, label=f'Co{c}')
            axs[0].set_title('Sil Co-dist')
            if ok3:
                sil_sem_vals = silhouette_samples(emb3, cl2['SemCluster'], metric='cosine')
                for c in sorted(cl2['SemCluster'].unique()):
                    axs[1].hist(sil_sem_vals[cl2['SemCluster'] == c], bins=20, alpha=0.5, label=f'Sem{c}')
                axs[1].set_title('Sil Sem-dist')
            axs[1].legend()
            st.pyplot(fig)
