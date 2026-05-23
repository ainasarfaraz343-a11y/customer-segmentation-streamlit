import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------------
# 1. Page Configuration & Layout Customization
# --------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Real-Time Customer Profiling Engine")
st.write("This application models customer segmentation using K-Means clustering over baseline benchmarks.")
st.markdown("---")

# --------------------------------------------------------
# 2. Automated Safe Data Loading Pipeline
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
# 3. K-Means Clustering Calculations
# --------------------------------------------------------
num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster_ID'] = y_kmeans

# Establish clear human-readable profiles based on calculated centroids
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
# 4. Upper Main Dashboard Row (KPI Summary Metrics)
# --------------------------------------------------------
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric(label="Active Algorithm Cluster Count", value=f"{num_clusters} Strategy Groups")
metric_col2.metric(label="Model Accuracy Silhouette Score", value=f"{sil_score:.4f}")
metric_col3.metric(label="Total Modeled Customer Rows", value=f"{len(df)} Records")

st.markdown("---")

# --------------------------------------------------------
# 5. Centered Live Interactive Simulator Control Panel
# --------------------------------------------------------
st.subheader("Live Customer Profiling Input")
st.write("Adjust the variables below to evaluate and route an incoming client profile in real time:")

left_spacer, center_content, right_spacer = st.columns()

with center_content:
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        input_income = st.slider("Annual Income (k$):", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 60)

    with col_input2:
        input_spending = st.slider("Spending Score (1-100):", int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max()), 50)

    # Process individual live input instance safely with array extraction
    simulated_vector = np.array([[input_income, input_spending]])
    simulated_scaled = scaler.transform(simulated_vector)
    predicted_cluster_id = kmeans.predict(simulated_scaled)[0]
    predicted_profile = cluster_mapping[predicted_cluster_id]

    # Dynamic UI status alert layout updates based on selected targets
    if "Elite" in predicted_profile:
        st.success(f"**Classification Output:** Profile metrics map directly to **{predicted_profile}**.")
    elif "Careful" in predicted_profile or "Standard" in predicted_profile:
        st.info(f"**Classification Output:** Profile metrics map directly to **{predicted_profile}**.")
    else:
        st.warning(f"**Classification Output:** Profile metrics map directly to **{predicted_profile}**.")

st.markdown("---")

# --------------------------------------------------------
# 6. Visualization Grid (Dual Plot Interfaces)
# --------------------------------------------------------
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("Structural Variance Evaluation")
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4.5))
    ax_elbow.plot(range(1, 11), wcss, marker='o', color='#2563eb', linewidth=2.5, markersize=5)
    ax_elbow.axvline(x=num_clusters, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Current Target Cut-off (K={num_clusters})')
    ax_elbow.set_title('Inertia (WCSS) Value Trend Path', fontsize=10, fontweight='bold', color='#1e293b')
    ax_elbow.set_xlabel('Cluster Count (K)', fontsize=8)
    ax_elbow.set_ylabel('Within-Cluster Variance', fontsize=8)
    ax_elbow.grid(True, linestyle=':', alpha=0.5)
    ax_elbow.legend(fontsize=8)
    st.pyplot(fig_elbow)

with plot_col2:
    st.subheader("Spatial Segment Clustering Analysis Map")
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4.5))  # Adjusted to match the layout perfectly
    
    unique_clusters = sorted(df['Customer Segment'].unique())
    
    for segment in unique_clusters:
        segmented_data = df[df['Customer Segment'] == segment]
        ax_scatter.scatter(
            segmented_data['Annual Income (k$)'], segmented_data['Spending Score (1-100)'],
            label=segment, s=65, alpha=0.75, edgecolor='none'
        )
    
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
    
    # Position adjusted cleanly outside the bounding box frame
    ax_scatter.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    st.pyplot(fig_scatter, bbox_inches='tight')

st.markdown("---")

# --------------------------------------------------------
# 7. Lower Section: Matrix Statistics & Strategic Playbook
# --------------------------------------------------------
st.subheader("Segment Cluster Centroids & Operational Profiling Descriptions")
cluster_profiles = df.groupby('Customer Segment')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(cluster_profiles, use_container_width=True)

st.markdown("### Strategic Application Guidelines")
st.write("- **Elite Spenders**: Target with premium tier brand advocacy memberships and high-end luxury item drops.")
st.write("- **Careful Buyers**: Offer cashback structures, value-focused bundles, and long-term security incentives.")
st.write("- **Impulsive Shoppers**: Focus on flash sales, real-time push notifications, and clear social proof elements.")
st.write("- **Bargain Hunters**: Allocate deep discounts, inventory clearance alerts, and strict loyalty reward tokens.")
st.write("- **Standard Buyers**: Engage with standard promotional schedules, consistent baseline options, and customer support consistency.")
