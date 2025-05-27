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
from io import BytesIO

st.set_page_config(page_title="MIDUS Codes Clustering & Networks", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xls", "xlsx"])
if uploaded:
    sheets = pd.ExcelFile(uploaded).sheet_names
    sheet = st.sidebar.selectbox("Select sheet", sheets)
    # Threshold and cluster parameters
    k2 = st.sidebar.slider("# clusters (M2 & M3)", 2, 10, 5)
    thr2_co = st.sidebar.slider("Co-occ threshold", 1, 100, 5)
    thr2_sem = st.sidebar.slider("Sem threshold", 0.0, 1.0, 0.4, step=0.05)
    run = st.sidebar.button("Run Analysis")

    if run:
        # Load and clean data
        df = pd.read_excel(uploaded, sheet_name=sheet)
        cols2 = df.filter(regex=r'(?i)_M2$').columns
        cols3 = df.filter(regex=r'(?i)_M3$').columns
        df2 = df[cols2].replace({'.': np.nan, ' ': np.nan}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        df3 = df[cols3].replace({'.': np.nan, ' ': np.nan}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        # Compute clusters and similarities
        co2, sim2, emb2, cl2, ok2 = compute(df2, k2)
        co3, sim3, emb3, cl3, ok3 = compute(df3, k2)

        # Evaluate validity
        ks = list(range(2, 9))
        val2 = evaluate(co2, emb2, ks)
        eval_m2 = evaluate(co2, emb2, ks)

        # Prepare Fit metrics and stability
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
        sil_sem_vals = silhouette_samples(emb3, cl2['SemCluster'], metric='cosine') if ok3 else None
        # Stability summary
        df_map = pd.DataFrame({'code': df2.columns, 'm2_sem': cl2['SemCluster'], 'm3_sem': cl3['SemCluster']})
        stability = []
        for lbl, grp in df_map.groupby('m2_sem'):
            vc = grp['m3_sem'].value_counts()
            dom = vc.idxmax(); cnt = vc.max(); total = len(grp)
            stability.append({
                'M2_cluster': lbl, 'total': total,
                'dominant_M3': dom, 'count': cnt,
                'stability': f"{cnt/total:.1%}",
                'codes_all': ", ".join(grp['code']),
                'codes_dom': ", ".join(grp.loc[grp['m3_sem']==dom, 'code'])
            })
        df_stability = pd.DataFrame(stability).sort_values('M2_cluster')

        # Tabs for results
        tabs = st.tabs(["Validity", "Assignments", "Network", "M2 Internal", "Fit M2→M3"])

        # Validity
        with tabs[0]:
            st.subheader("Clustering Validity (M2)")
            st.dataframe(val2)

        # Assignments
        with tabs[1]:
            st.subheader("Code Assignments")
            st.dataframe(cl2)

        # Network
        with tabs[2]:
            st.subheader("Co-occurrence Network (M2)")
            fig1, ax1 = plt.subplots()
            G_co = make_graph(co2, df2.columns.tolist(), cl2['CoCluster'], thr2_co)
            pos = nx.spring_layout(G_co, seed=1)
            nx.draw(G_co, pos, node_color=[d['color'] for _,d in G_co.nodes(data=True)], with_labels=True, ax=ax1)
            st.pyplot(fig1)
            if ok2:
                st.subheader("Semantic Similarity Network (M2)")
                fig2, ax2 = plt.subplots()
                G_sem = make_graph(sim2, df2.columns.tolist(), cl2['SemCluster'], thr2_sem)
                nx.draw(G_sem, nx.spring_layout(G_sem, seed=1), node_color=[d['color'] for _,d in G_sem.nodes(data=True)], with_labels=True, ax=ax2)
                st.pyplot(fig2)

        # M2 Internal
        with tabs[3]:
            st.subheader("M2 Internal Analysis")
            st.line_chart(eval_m2.set_index('k')[['sil_co','db_co']])
            if emb2 is not None:
                st.line_chart(eval_m2.set_index('k')[['sil_sem','db_sem']])

        # Fit M2→M3
        with tabs[4]:
            st.subheader("Fit M2 → M3 Metrics")
            st.dataframe(df_fit_metrics)
            st.subheader("Contingency Heatmaps")
            fig3, (ax3, ax4) = plt.subplots(1,2,figsize=(10,4))
            sns.heatmap(cont_co, annot=True, fmt='d', ax=ax3); ax3.set_title('Co-occ')
            sns.heatmap(cont_sem, annot=True, fmt='d', ax=ax4); ax4.set_title('Semantic')
            st.pyplot(fig3)
            st.subheader("Silhouette Distributions")
            fig4, (ax5, ax6) = plt.subplots(1,2,figsize=(10,4))
            for c in sorted(cl2['CoCluster'].unique()): ax5.hist(sil_co_vals[cl2['CoCluster']==c], bins=20, alpha=0.5, label=f'Co{c}')
            ax5.set_title('Sil Co-dist'); ax5.legend()
            if sil_sem_vals is not None:
                for c in sorted(cl2['SemCluster'].unique()): ax6.hist(sil_sem_vals[cl2['SemCluster']==c], bins=20, alpha=0.5, label=f'Sem{c}')
                ax6.set_title('Sil Sem-dist'); ax6.legend()
            st.pyplot(fig4)

        # Export results
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            val2.to_excel(writer, sheet_name='M2_validity', index=False)
            cl2.to_excel(writer, sheet_name='Assignments', index=False)
            eval_m2.to_excel(writer, sheet_name='M2_internal', index=False)
            df_fit_metrics.to_excel(writer, sheet_name='Fit_metrics', index=False)
            df_stability.to_excel(writer, sheet_name='Fit_stability', index=False)
        towrite.seek(0)
        st.sidebar.download_button(
            label='Download Results as Excel',
            data=towrite.getvalue(),
            file_name='midus_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
