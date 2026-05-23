import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------------
# 1. Page Configuration & Setup
# --------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Engine",
    layout="wide"
)

st.title("Real-Time Customer Profiling Engine")
st.write("This application models customer segmentation using K-Means clustering over baseline benchmarks.")
st.markdown("---")

# --------------------------------------------------------
# 2. Automated Safe Data Loading
# --------------------------------------------------------
try:
    df = pd.read_csv('Mall_Customers.csv')
except Exception:
    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerID': range(1, 201),
        'Annual Income (k$)': np.random.randint(15, 135, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    })

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------
# 3. K-Means Pipeline & Mapping Execution
# --------------------------------------------------------
num_clusters = 5

# Fit K-Means algorithm using 5 strategic clusters
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster_ID'] = y_kmeans

# Map Cluster IDs to Real Business Profiles dynamically based on centroids
cluster_mapping = {}
centroids = df.groupby('Cluster_ID')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()

for idx, row in centroids.iterrows():
    inc = row['Annual Income (k$)']
    spnd = row['Spending Score (1-100)']
    
    if inc > 65 and spnd > 65:
        cluster_mapping[idx] = "Elite Spenders (High Income, High Spend)"
    elif inc > 65 and spnd <= 65:
        cluster_mapping[idx] = "Careful Buyers (High Income, Low Spend)"
    elif inc <= 65 and spnd > 65:
        cluster_mapping[idx] = "Impulsive Shoppers (Low Income, High Spend)"
    elif inc <= 40 and spnd <= 40:
        cluster_mapping[idx] = "Bargain Hunters (Low Income, Low Spend)"
    else:
        cluster_mapping[idx] = "Standard Buyers (Middle Class)"

df['Customer Segment'] = df['Cluster_ID'].map(cluster_mapping)
sil_score = silhouette_score(X_scaled, y_kmeans)

# --------------------------------------------------------
# 4. Main Dashboard Output Layout (KPI Metrics)
# --------------------------------------------------------
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric(label="Active Algorithm Cluster Count", value=f"{num_clusters} Strategy Groups")
metric_col2.metric(label="Model Accuracy Silhouette Score", value=f"{sil_score:.4f}")
metric_col3.metric(label="Total Modeled Customer Rows", value=f"{len(df)} Records")

st.markdown("---")

# --------------------------------------------------------
# 5. Live Customer Input Section (Moved into the Middle)
# --------------------------------------------------------
st.subheader("Live Customer Profiling Input")
st.write("Adjust the sliders below to see which segment a new client matches in real time:")

col_input1, col_input2 = st.columns(2)

with col_input1:
    input_income = st.slider("Annual Income (k$):", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 60)

with col_input2:
    input_spending = st.slider("Spending Score (1-100):", int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max()), 50)

# Process individual live input instance safely with array extraction [0]
simulated_vector = np.array([[input_income, input_spending]])
simulated_scaled = scaler.transform(simulated_vector)
predicted_cluster_id = kmeans.predict(simulated_scaled)[0]  # FIXED: Extracted integer to avoid index mapping error
predicted_profile = cluster_mapping[predicted_cluster_id]

# Show the clean text result right under the sliders
st.info(f"Prediction Result Analysis: The user belongs to the **{predicted_profile}** segment.")
st.markdown("---")

# --------------------------------------------------------
# 6. Visualization & Dual Plot Window Configurations
# --------------------------------------------------------
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("Structural Variance Evaluation (Elbow Graph)")
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4.2))
    ax_elbow.plot(range(1, 11), wcss, marker='o', color='#2563eb', linewidth=2.5, markersize=5)
    ax_elbow.axvline(x=num_clusters, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Current Target Cut-off (K={num_clusters})')
    ax_elbow.set_title('Inertia (WCSS) Value Trend Path', fontsize=10, fontweight='bold', color='#1e293b')
    ax_elbow.set_xlabel('Cluster Count (K)', fontsize=8)
    ax_elbow.set_ylabel('Within-Cluster Variance', fontsize=8)
    ax_elbow.grid(True, linestyle=':', alpha=0.5)
    ax_elbow.legend(fontsize=8)
    st.pyplot(fig_elbow)

with plot_col2:
    st.subheader("Spatial Segment Clustering Visualization Plot")
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    
    unique_clusters = df['Customer Segment'].unique()
    
    for segment in unique_clusters:
        segmented_data = df[df['Customer Segment'] == segment]
        ax_scatter.scatter(
            segmented_data['Annual Income (k$)'], segmented_data['Spending Score (1-100)'],
            label=segment, s=60, alpha=0.8
        )
    
    # Overlay the simulated user vector visually on map
    ax_scatter.scatter(
        input_income, input_spending, 
        c='#0f172a', marker='X', s=350, 
        edgecolor='#ffffff', linewidth=2.5, 
        label='Simulated Client Vector'
    )
    
    ax_scatter.set_title(f'K-Means Multi-Segment Map Output (K={num_clusters})', fontsize=10, fontweight='bold', color='#1e293b')
    ax_scatter.set_xlabel('Annual Income (k$)', fontsize=8)
    ax_scatter.set_ylabel('Spending Score (1-100)', fontsize=8)
    ax_scatter.grid(True, linestyle=':', alpha=0.4)
    ax_scatter.legend(loc='upper right', fontsize=7)
    st.pyplot(fig_scatter)

st.markdown("---")

# Lower Row: Datatable Profile Summary Matrix Breakdown
st.subheader("Segment Cluster Centroids & Operational Profiling Descriptions")
cluster_profiles = df.groupby('Customer Segment')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(cluster_profiles, use_container_width=True)
