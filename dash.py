import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# Load the data
data = pd.read_csv('segmented_customers.csv')

# Streamlit Title and Description
st.title("Customer Segmentation Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of customer segmentation using K-Means clustering. 
Explore demographics, spending patterns, and other insights for strategic decision-making.
""")

# Data Overview
st.subheader("Dataset Overview")
if st.checkbox("Show Dataset"):
    st.write(data)

st.write(f"Total Records: {len(data)}")
st.write(f"Total Clusters: {data['cluster_kmeans'].nunique()}")

# Numeric Columns for Analysis
numeric_columns = data.select_dtypes(include=['number']).columns

# 1. PCA Visualization
st.subheader("PCA Visualization of Clusters (2D)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[numeric_columns])
pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = data['cluster_kmeans']

fig_pca = plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=50)
plt.title("PCA Visualization of Clusters (2D)")
st.pyplot(fig_pca)

# 2. t-SNE Visualization
st.subheader("t-SNE Visualization of Clusters (3D)")
tsne = TSNE(n_components=3, random_state=42)
tsne_result = tsne.fit_transform(data[numeric_columns])
tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2', 'TSNE3'])
tsne_df['Cluster'] = data['cluster_kmeans']

fig_tsne = px.scatter_3d(tsne_df, x='TSNE1', y='TSNE2', z='TSNE3', color=tsne_df['Cluster'].astype(str),
                         title="t-SNE Visualization of Clusters (3D)")
st.plotly_chart(fig_tsne)

# 3. Cluster Summary
st.subheader("Cluster Summary Statistics")
cluster_summary = data.groupby('cluster_kmeans', as_index=False)[numeric_columns].mean()
st.dataframe(cluster_summary)

# 4. Demographic Analysis
st.subheader("Demographic Analysis")
st.write("Analyze demographic distribution across clusters.")

# Gender Distribution
gender_dist = data.groupby(['cluster_kmeans', 'gender']).size().reset_index(name='count')
fig_gender = px.bar(gender_dist, x='cluster_kmeans', y='count', color='gender',
                    title='Gender Distribution by Cluster', labels={'count': 'Count'})
st.plotly_chart(fig_gender)

# Marital Status Distribution
if 'marital_status' in data.columns:
    marital_dist = data.groupby(['cluster_kmeans', 'marital_status']).size().reset_index(name='count')
    fig_marital = px.bar(marital_dist, x='cluster_kmeans', y='count', color='marital_status',
                         title='Marital Status Distribution by Cluster', labels={'count': 'Count'})
    st.plotly_chart(fig_marital)

# 5. Spending Patterns
st.subheader("Spending Patterns")
spending_data = data.groupby('cluster_kmeans')['avg_purchase_value'].mean().reset_index()
fig_spending = px.bar(spending_data, x='cluster_kmeans', y='avg_purchase_value',
                      title='Average Purchase Value by Cluster', labels={'avg_purchase_value': 'Average Purchase Value'})
st.plotly_chart(fig_spending)

# 6. Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
correlation = data[numeric_columns].corr()
fig_corr = plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Features")
st.pyplot(fig_corr)

# 7. Filter and Drill Down
st.subheader("Filter and Drill Down by Cluster")
selected_cluster = st.selectbox("Select a Cluster", data['cluster_kmeans'].unique())
filtered_data = data[data['cluster_kmeans'] == selected_cluster]
st.write(f"Details for Cluster {selected_cluster}")
st.dataframe(filtered_data)

# 8. Top Products (Optional Feature)
if 'product_category' in data.columns:
    st.subheader("Top Products by Cluster")
    product_data = data.groupby(['cluster_kmeans', 'product_category']).size().reset_index(name='count')
    fig_product = px.bar(product_data, x='cluster_kmeans', y='count', color='product_category', barmode='group',
                         title="Top Products by Cluster", labels={'count': 'Count'})
    st.plotly_chart(fig_product)

# Conclusion
st.markdown("""
### Conclusion
This dashboard offers a detailed breakdown of customer clusters, enabling businesses to make data-driven decisions.
By tailoring marketing efforts and understanding customer behaviors, businesses can optimize their strategies and improve customer retention.
""")
