import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)
from io import BytesIO

# ─── Utility Functions ─────────────────────────────────────────────────────────
def compute_hybrid(co_matrix, emb_matrix, k):
    hybrid_sim = 0.5 * (co_matrix / co_matrix.max()) + 0.5 * (cosine_similarity(emb_matrix))
    np.fill_diagonal(hybrid_sim, 0)
    clustering = AgglomerativeClustering(n_clusters=k).fit_predict(hybrid_sim)
    return hybrid_sim, clustering
def compute(df_codes, k):
    codes = df_codes.columns.tolist()
    co = df_codes.T.dot(df_codes).values
    np.fill_diagonal(co, 0)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode(codes)
        sim = cosine_similarity(emb)
        ok = True
    except Exception:
        emb, sim, ok = None, None, False
    # Co-occ remains internal
    cco = AgglomerativeClustering(n_clusters=k).fit_predict(co)
    # Semantic clustering
    cse = (AgglomerativeClustering(n_clusters=k).fit_predict(emb)
           if ok else np.full(len(codes), -1))
    dfc = pd.DataFrame({
        'Code': codes,
        'SemCluster': cse
    })
    pal = sns.color_palette('hls', max(2, k))
    dfc['color'] = [mcolors.to_hex(pal[i % len(pal)]) for i in cse]
    return co, sim, emb, dfc, ok


def evaluate(matrix, emb, ks):
    rows = []
    for k in ks:
        # Only semantic validity
        if emb is not None:
            ls = AgglomerativeClustering(n_clusters=k).fit_predict(emb)
            sil = silhouette_score(emb, ls)
            db = davies_bouldin_score(emb, ls)
            ch = calinski_harabasz_score(emb, ls)
        else:
            sil, db, ch = None, None, None
        rows.append({
            'k': k,
            'sil_sem': sil,
            'db_sem': db,
            'ch_sem': ch
        })
    return pd.DataFrame(rows)


def make_graph(matrix, labels, clusters, thr):
    G = nx.Graph()
    pal = sns.color_palette('hls', len(np.unique(clusters)))
    cols = [mcolors.to_hex(pal[i % len(pal)]) for i in clusters]
    for i, lbl in enumerate(labels):
        G.add_node(lbl, color=cols[i])
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if matrix[i, j] > thr:
                G.add_edge(labels[i], labels[j], weight=float(matrix[i, j]))
    return G
# ────────────────────────────────────────────────────────────────────────────────

# ─── Streamlit App ────────────────────────────────────────────────────────────
st.set_page_config(page_title="MIDUS Code Clustering & Network Analysis", layout="wide")
st.title("MIDUS Code Clustering & Network Analysis")

# Sidebar controls
st.sidebar.header("Settings")
st.sidebar.markdown("### Analysis Configuration")
uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xls", "xlsx"])
if uploaded:
    sheets = pd.ExcelFile(uploaded).sheet_names
    sheet = st.sidebar.selectbox("Select sheet", sheets)
    k = st.sidebar.slider("# clusters (Semantic)", 2, 10, 5)
    k_hybrid = st.sidebar.slider("# clusters (Hybrid)", 2, 10, 5)
    thr_sem = st.sidebar.slider("Semantic threshold", 0.0, 1.0, 0.4, step=0.05)
    thr_hybrid = st.sidebar.slider("Hybrid threshold", 0.0, 1.0, 0.4, step=0.05)
    run = st.sidebar.button("Run Analysis")

    if run:
        df = pd.read_excel(uploaded, sheet_name=sheet)
        cols2 = df.filter(regex=r'(?i)_M2$').columns
        cols3 = df.filter(regex=r'(?i)_M3$').columns
        df2 = (df[cols2]
               .replace({'.': np.nan, ' ': np.nan})
               .apply(pd.to_numeric, errors='coerce')
               .fillna(0)
               .astype(int))
        df3 = (df[cols3]
               .replace({'.': np.nan, ' ': np.nan})
               .apply(pd.to_numeric, errors='coerce')
               .fillna(0)
               .astype(int))

        # Compute with semantic focus
        co2, sim2, emb2, cl2, ok2 = compute(df2, k)
        co3, sim3, emb3, cl3, ok3 = compute(df3, k)

        # Hybrid computations for M2
        if ok2:
            sim_hybrid2, cl_hybrid2 = compute_hybrid(co2, emb2, k_hybrid)
        else:
            sim_hybrid2, cl_hybrid2 = None, np.full(len(df2.columns), -1)

        df_hybrid = pd.DataFrame({
            'Code': df2.columns,
            'HybridCluster': cl_hybrid2
        })
        pal_h = sns.color_palette('Set2', max(2, k_hybrid))
        df_hybrid['color'] = [mcolors.to_hex(pal_h[i % len(pal_h)]) for i in cl_hybrid2]

        ks = list(range(2, 9))
        sem_val = evaluate(sim2 if ok2 else np.zeros_like(sim2), emb2, ks)

        # Fit semantic metrics
        if ok3:
            fit_metrics = {
                'sil_sem': silhouette_score(emb3, cl2['SemCluster'], metric='cosine'),
                'ch_sem': calinski_harabasz_score(emb3, cl2['SemCluster']),
                'db_sem': davies_bouldin_score(emb3, cl2['SemCluster'])
            }
            df_fit = pd.DataFrame([fit_metrics])
        else:
            df_fit = pd.DataFrame([])

        # Semantic contingency and distributions
        cont_sem = pd.crosstab(cl2['SemCluster'], cl3['SemCluster'])
        sil_sem_vals = (silhouette_samples(emb3, cl2['SemCluster'], metric='cosine')
                        if ok3 else None)

        # Stability summary
        df_map = pd.DataFrame({
            'code': df2.columns,
            'm2_sem': cl2['SemCluster'],
            'm3_sem': cl3['SemCluster']
        })
        stability = []
        for lbl, grp in df_map.groupby('m2_sem'):
            vc = grp['m3_sem'].value_counts()
            dom = vc.idxmax(); cnt = vc.max(); total = len(grp)
            stability.append({
                'M2_cluster': lbl,
                'total_codes': total,
                'dominant_M3_cluster': dom,
                'count_in_dominant': cnt,
                'stability_ratio': f"{cnt/total:.1%}",
                'codes_M2': grp['code'].tolist(),
                'codes_in_dominant_M3': grp.loc[grp['m3_sem']==dom, 'code'].tolist()
            })
        df_stab = pd.DataFrame(stability).sort_values('M2_cluster')

        # Tabs: semantic-only + hybrid
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "Semantic Validity", "Semantic Assignments", "Semantic Network",
            "M2 Internal Semantic", "Fit M2→M3 Semantic",
            "Hybrid Network", "Hybrid Clustering"
        ])
        with t6:
            st.subheader("Hybrid Network (M2)")
            if sim_hybrid2 is not None:
                fig, ax = plt.subplots()
                G_hybrid = make_graph(sim_hybrid2, df2.columns.tolist(), cl_hybrid2, thr_hybrid)
                pos = nx.spring_layout(G_hybrid, seed=1)
                nx.draw(G_hybrid, pos,
                        node_color=[mcolors.to_hex(pal_h[i % len(pal_h)]) for i in cl_hybrid2],
                        with_labels=True, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Hybrid network computation unavailable.")

        with t7:
            st.subheader("Hybrid Clustering Assignments (M2)")
            st.dataframe(df_hybrid[['Code', 'HybridCluster']])

        with t1:
            st.subheader("Semantic Clustering Validity")
            st.dataframe(sem_val.set_index('k'))

        with t2:
            st.subheader("Semantic Code Assignments")
            st.dataframe(cl2[['Code', 'SemCluster']])

        with t3:
            st.subheader("Semantic Network (M2)")
            fig, ax = plt.subplots()
            G_sem = make_graph(sim2, df2.columns.tolist(), cl2['SemCluster'], thr_sem)
            pos = nx.spring_layout(G_sem, seed=1)
            nx.draw(G_sem, pos,
                    node_color=[d['color'] for _, d in G_sem.nodes(data=True)],
                    with_labels=True, ax=ax)
            st.pyplot(fig)

        with t4:
            st.subheader("M2 Internal Semantic Analysis")
            df_plot = sem_val.set_index('k')
            st.line_chart(df_plot)

        with t5:
            st.subheader("Fit M2 → M3 Semantic Metrics")
            if not df_fit.empty:
                st.dataframe(df_fit)
                st.subheader("Semantic Contingency")
                fig2, ax2 = plt.subplots()
                sns.heatmap(cont_sem, annot=True, fmt='d', ax=ax2)
                st.pyplot(fig2)
                st.subheader("Silhouette Distribution (Semantic)")
                fig3, ax3 = plt.subplots()
                for c in sorted(cl2['SemCluster'].unique()):
                    ax3.hist(sil_sem_vals[cl2['SemCluster']==c], bins=20, alpha=0.5, label=f'Sem{c}')
                ax3.legend()
                st.pyplot(fig3)
                st.subheader("Enhanced Semantic Cluster Stability Summary")
                st.dataframe(df_stab)
            else:
                st.warning("Semantic Fit analysis unavailable: embeddings missing.")

        # Export semantic results
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            sem_val.to_excel(writer, sheet_name='SemanticValidity', index=False)
            cl2[['Code','SemCluster']].to_excel(writer, sheet_name='Assignments', index=False)
            sem_val.to_excel(writer, sheet_name='M2_Internal_Semantic', index=False)
            df_fit.to_excel(writer, sheet_name='Fit_Semantic_Metrics', index=False)
            df_stab.to_excel(writer, sheet_name='Semantic_Stability', index=False)
            df_hybrid[['Code','HybridCluster']].to_excel(writer, sheet_name='Hybrid_Assignments', index=False)
        buf.seek(0)
        st.sidebar.markdown("### Export Results")
        st.sidebar.download_button(
            "Download Analysis Results", buf, "semantic_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
