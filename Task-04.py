import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def main():
    st.title("Customer Segmentation using K-means Clustering")

    # Step 1: Load the dataset
    st.header("1. Load Dataset")
    file_path = 'Mall_Customers.csv'

    try:
        data = pd.read_csv(file_path)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Step 2: Basic Information
        st.subheader("Dataset Information")
        st.write("Shape of the dataset:", data.shape)
        st.write("Data Types:")
        st.write(data.dtypes)

        # Step 3: EDA (Exploratory Data Analysis)
        st.header("2. Exploratory Data Analysis")

        # Distribution plots
        st.subheader("Feature Distributions")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        sns.histplot(data['Age'], kde=True, bins=20, color='skyblue', ax=axs[0])
        axs[0].set_title('Age Distribution')
        axs[0].set_xlabel('Age')

        sns.histplot(data['Annual Income (k$)'], kde=True, bins=20, color='orange', ax=axs[1])
        axs[1].set_title('Annual Income Distribution')
        axs[1].set_xlabel('Annual Income (k$)')

        sns.histplot(data['Spending Score (1-100)'], kde=True, bins=20, color='green', ax=axs[2])
        axs[2].set_title('Spending Score Distribution')
        axs[2].set_xlabel('Spending Score (1-100)')

        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        correlation = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Gender-based analysis
        st.subheader("Gender-Based Analysis")
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(x='Gender', y='Spending Score (1-100)', data=data, palette='pastel', ax=axs[0])
        axs[0].set_title('Spending Score by Gender')

        sns.boxplot(x='Gender', y='Annual Income (k$)', data=data, palette='pastel', ax=axs[1])
        axs[1].set_title('Annual Income by Gender')

        st.pyplot(fig)

        # Step 4: Data Preprocessing
        st.header("3. Data Preprocessing")
        selected_features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_features)
        st.write("Features have been standardized.")

        # Step 5: Determine Optimal K (Elbow Method)
        st.header("4. Optimal Number of Clusters (Elbow Method)")
        wcss = []
        K_range = range(1, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_range, wcss, marker='o', linestyle='--')
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        # Step 6: Apply K-means with K=4
        st.header("5. Apply K-means Clustering")
        optimal_k = st.slider("Select the optimal number of clusters (K):", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(scaled_features)
        data['Cluster'] = kmeans.labels_

        st.write("Cluster assignment completed. Preview of the dataset:")
        st.dataframe(data.head())

        # Cluster Centers
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=selected_features.columns
        )
        st.write("Cluster Centers (De-scaled):")
        st.dataframe(cluster_centers)

        # Step 7: Visualize Clusters
        st.header("6. Visualize Clusters")

        # 2D Visualization
        st.subheader("2D Visualization")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            x=data['Annual Income (k$)'],
            y=data['Spending Score (1-100)'],
            hue=data['Cluster'],
            palette='viridis',
            s=100,
            alpha=0.8,
            ax=ax
        )
        ax.set_title('Customer Clusters (2D Visualization)')
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score (1-100)')
        st.pyplot(fig)

        # 3D Visualization
        st.subheader("3D Visualization")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            data['Age'],
            data['Annual Income (k$)'],
            data['Spending Score (1-100)'],
            c=data['Cluster'],
            cmap='viridis',
            s=50,
            alpha=0.8
        )

        ax.set_title('Customer Clusters (3D Visualization)')
        ax.set_xlabel('Age')
        ax.set_ylabel('Annual Income (k$)')
        ax.set_zlabel('Spending Score (1-100)')

        st.pyplot(fig)

        # Step 8: Analyze and Save Results
        st.header("7. Analyze and Save Results")
        cluster_analysis = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
        cluster_sizes = data['Cluster'].value_counts()

        st.write("Cluster-wise Averages:")
        st.dataframe(cluster_analysis)

        st.write("Cluster Sizes:")
        st.dataframe(cluster_sizes)

        output_file_path = 'Mall_Customers_with_Clusters.csv'
        data.to_csv(output_file_path, index=False)

        st.success(f"Dataset with cluster labels saved successfully to: {output_file_path}")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
