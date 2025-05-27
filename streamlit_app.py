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

    cco = AgglomerativeClustering(n_clusters=k).fit_predict(co)
    cse = (AgglomerativeClustering(n_clusters=k).fit_predict(emb)
           if ok else np.full(len(codes), -1))

    dfc = pd.DataFrame({
        'Code': codes,
        'CoCluster': cco,
        'SemCluster': cse
    })
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
    for i, lbl in enumerate(labels):
        G.add_node(lbl, color=cols[i])
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if matrix[i, j] > thr:
                G.add_edge(labels[i], labels[j], weight=float(matrix[i, j]))
    return G
# ────────────────────────────────────────────────────────────────────────────────

# ─── Streamlit App ────────────────────────────────────────────────────────────
st.set_page_config(page_title="MIDUS Codes Clustering & Networks", layout="wide")
st.title("MIDUS Codes Clustering & Networks")

# Sidebar for settings & upload
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xls", "xlsx"])
if uploaded:
    sheets = pd.ExcelFile(uploaded).sheet_names
    sheet = st.sidebar.selectbox("Select sheet", sheets)
    k2 = st.sidebar.slider("# clusters (M2 & M3)", 2, 10, 5)
    thr2_co = st.sidebar.slider("Co-occ threshold", 1, 100, 5)
    thr2_sem = st.sidebar.slider("Sem threshold", 0.0, 1.0, 0.4, step=0.05)
    run = st.sidebar.button("Run Analysis")

    if run:
        # Load & clean data
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

        # Compute clusters & embeddings
        co2, sim2, emb2, cl2, ok2 = compute(df2, k2)
        co3, sim3, emb3, cl3, ok3 = compute(df3, k2)

        ks = list(range(2, 9))
        val2 = evaluate(co2, emb2, ks)

        # Prepare Fit M2→M3 data
        fit_metrics = {
            'sil_co': silhouette_score(co3, cl2['CoCluster']),
            'ch_co': calinski_harabasz_score(co3, cl2['CoCluster']),
            'db_co': davies_bouldin_score(co3, cl2['CoCluster'])
        }
        if ok3:
            fit_metrics.update({
                'sil_sem': silhouette_score(emb3, cl2['SemCluster'], metric='cosine'),
                'ch_sem': calinski_harabasz_score(emb3, cl2['SemCluster']),
                'db_sem': davies_bouldin_score(emb3, cl2['SemCluster'])
            })
        df_fit_metrics = pd.DataFrame([fit_metrics])
        cont_co = pd.crosstab(cl2['CoCluster'], cl3['CoCluster'])
        cont_sem = pd.crosstab(cl2['SemCluster'], cl3['SemCluster'])
        sil_co_vals = silhouette_samples(co3, cl2['CoCluster'])
        sil_sem_vals = (silhouette_samples(emb3, cl2['SemCluster'], metric='cosine')
                        if ok3 else None)

        # Build stability summary
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
                'total': total,
                'dominant_M3': dom,
                'count': cnt,
                'stability': f"{cnt/total:.1%}",
                'codes_all': ", ".join(grp['code']),
                'codes_dom': ", ".join(grp.loc[grp['m3_sem']==dom, 'code'])
            })
        df_stability = pd.DataFrame(stability).sort_values('M2_cluster')

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Validity", "Assignments", "Network", "M2 Internal", "Fit M2→M3"
        ])

        # Validity tab
        with tab1:
            st.subheader("M2 Clustering Validity")
            st.dataframe(val2)

        # Assignments tab
        with tab2:
            st.subheader("Code Assignments")
            st.dataframe(cl2)

        # Network tab
        with tab3:
            st.subheader("Co-occurrence Network (M2)")
            fig1, ax1 = plt.subplots()
            G_co = make_graph(co2, df2.columns.tolist(), cl2['CoCluster'], thr2_co)
            pos_co = nx.spring_layout(G_co, seed=1)
            nx.draw(G_co, pos_co,
                    node_color=[d['color'] for _, d in G_co.nodes(data=True)],
                    with_labels=True, ax=ax1)
            st.pyplot(fig1)
            if ok2:
                st.subheader("Semantic Network (M2)")
                fig2, ax2 = plt.subplots()
                G_se = make_graph(sim2, df2.columns.tolist(), cl2['SemCluster'], thr2_sem)
                pos_se = nx.spring_layout(G_se, seed=1)
                nx.draw(G_se, pos_se,
                        node_color=[d['color'] for _, d in G_se.nodes(data=True)],
                        with_labels=True, ax=ax2)
                st.pyplot(fig2)

        # M2 Internal tab
        with tab4:
            st.subheader("M2 Internal Analysis")
            df_plot = val2.set_index('k')
            st.line_chart(df_plot[['sil_co', 'db_co']])
            if emb2 is not None:
                st.line_chart(df_plot[['sil_sem', 'db_sem']])

        # Fit M2→M3 tab
        with tab5:
            st.subheader("Fit M2 → M3 Metrics")
            st.dataframe(df_fit_metrics)
            st.subheader("Contingency Heatmaps")
            fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(cont_co, annot=True, fmt='d', ax=ax3); ax3.set_title('Co-occ')
            sns.heatmap(cont_sem, annot=True, fmt='d', ax=ax4); ax4.set_title('Semantic')
            st.pyplot(fig3)
            st.subheader("Silhouette Distributions")
            fig4, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 4))
            for c in sorted(cl2['CoCluster'].unique()):
                ax5.hist(sil_co_vals[cl2['CoCluster'] == c], bins=20, alpha=0.5, label=f'Co{c}')
            ax5.set_title('Sil Co-dist'); ax5.legend()
            if sil_sem_vals is not None:
                for c in sorted(cl2['SemCluster'].unique()):
                    ax6.hist(sil_sem_vals[cl2['SemCluster'] == c], bins=20, alpha=0.5, label=f'Sem{c}')
                ax6.set_title('Sil Sem-dist'); ax6.legend()
            st.pyplot(fig4)

        # Export results button
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            val2.to_excel(writer, sheet_name='M2_validity', index=False)
            cl2.to_excel(writer, sheet_name='Assignments', index=False)
            val2.to_excel(writer, sheet_name='M2_internal', index=False)
            df_fit_metrics.to_excel(writer, sheet_name='Fit_metrics', index=False)
            df_stability.to_excel(writer, sheet_name='Fit_stability', index=False)
        towrite.seek(0)
        st.sidebar.download_button(
            label="Download Results as Excel",
            data=towrite.getvalue(),
            file_name="midus_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
