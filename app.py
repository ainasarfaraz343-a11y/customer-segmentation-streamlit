import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Set up the title of the Streamlit Web Application
st.title("Customer Segmentation")
st.write("Group your customers into different segments based on their purchasing behavior using K-Means Clustering.")

# Step 2: Implement the File Uploader feature so users can drag and drop their dataset
uploaded_file = st.file_uploader("Please upload your 'Mall_Customers.csv' file", type=["csv"])

# Step 3: Execute the data science pipeline once the file is uploaded successfully
if uploaded_file is not None:
    # Read the uploaded CSV data into a pandas dataframe
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset as a preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    if 'Gender' in df.columns:
        df['Gender_Mapped'] = df['Gender'].map({'Male': 0, 'Female': 1})
        
    # Extract features: Annual Income and Spending Score for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scale features using StandardScaler so they have equal variance weight
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 4: Implement the Elbow Method Section
    st.subheader("1. Elbow Method")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, 11), wcss, marker='o', color='b')
    ax_elbow.set_title('Elbow Method')
    ax_elbow.set_xlabel("Number of Clusters")
    ax_elbow.set_ylabel('WCSS')
    st.pyplot(fig_elbow)
    
    # Step 5: Add an interactive Streamlit slider for dynamic cluster (K) selection
    st.subheader("2. K-Means Clustering")
    num_clusters = st.slider("Select the number of clusters (K):", min_value=2, max_value=10, value=5)
    
    # Train the K-Means algorithm with the slider's user-selected cluster count
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    df['Cluster'] = y_kmeans
    
    # Calculate and display the overall cluster Silhouette Score metric
    score = silhouette_score(X_scaled, y_kmeans)
    st.metric(label="Silhouette Score", value=f"{score:.4f}")
    
    # NEW FEATURE: Sidebar Inputs for Live Single Customer Prediction
    st.sidebar.header("Single Customer Prediction")
    st.sidebar.write("Move the sliders to see which cluster a new customer belongs to:")
    
    # Creating individual inputs for dynamic user prediction
    input_income = st.sidebar.slider("Annual Income (k$):", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 50)
    input_spending = st.sidebar.slider("Spending Score (1-100):", int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max()), 50)
    
    # Transform inputs using the same scaler applied to training dataset
    user_data = np.array([[input_income, input_spending]])
    user_data_scaled = scaler.transform(user_data)
    
    # Predict the target cluster ID for the new customer profile
    predicted_cluster = kmeans.predict(user_data_scaled)[0]
    
    # Display the real-time prediction outcome on the sidebar panel
    st.sidebar.success(f"This customer belongs to **Cluster {predicted_cluster}**")
    
    # Step 6: Generate and render the final Customer Segmentation Scatter Plot
    st.subheader("3. Customer Segments Visualization")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    scatter = ax_scatter.scatter(
        df['Annual Income (k$)'], df['Spending Score (1-100)'],
        c=df['Cluster'], cmap='rainbow', s=100, edgecolor='black'
    )
    # Highlight the newly predicted user input position inside the global graph map
    ax_scatter.scatter(input_income, input_spending, c='black', marker='X', s=300, label='New Customer Input')
    
    ax_scatter.set_xlabel('Annual Income (k$)')
    ax_scatter.set_ylabel('Spending Score (1-100)')
    plt.colorbar(scatter, ax=ax_scatter, label='Cluster ID')
    plt.legend()
    st.pyplot(fig_scatter)
    
    # Step 7: Display mathematical summary profiles (mean values) for each group
    st.subheader("📈 4. Clusters Mean Profiles")
    st.dataframe(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
else:
    # Fallback message shown when the application is waiting for a data upload
    st.info("Please upload the 'Mall_Customers.csv' file to start the segmentation analysis.")
