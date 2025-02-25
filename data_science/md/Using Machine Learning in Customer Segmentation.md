
[Link](https://machinelearningmastery.com/using-machine-learning-in-customer-segmentation/)

In the past, businesses grouped customers based on simple things like age or gender. Now, machine learning has changed this process. Machine learning algorithms can analyze large amounts of data. In this article, we will explore how machine learning improves customer segmentation.
## Introduction to Customer Segmentation

Customer segmentation divides customers into different groups. These groups are based on similar traits or behaviors. The main goal is to understand each group better. This helps businesses create marketing strategies and products that fit each group’s specific needs.

The customers can be divided into groups based on several criteria:

1. **Demographic Segmentation**: Based on factors such as age, gender and occupation.
2. **Psychographic Segmentation**: Focusbrand loyalty and usage frequency.
4. **Geographic Segmentation**: Divides customers based on their geographical location.

Customer segmentation offers several advantages for businesses:

- **Personalized Marketing**: Businesses can send specific messages for each groups of customers.
- **Improved Customer Retention**: Organizations can identify the preferences of customers and make them loyal customers.
- **Enhanced Product Development**: Segmentation helps to understand what products customers want.

## Machine learning Algorithms for Customer Segmentation

Machine learning uses several algorithms to categorize customers based on their features. Some commonly used algorithms include:

1. **K-means Clustering**: Divides customers into clusters based on similar features.
2. **Hierarchical Clustering**: Organizes customers into a tree-like hierarchy of clusters.
3. **DBSCAN**: Identifies clusters based on density of points in data space.
4. **Principal Component Analysis (PCA)**: Reduces the dimensionality of data and preserves important information.
5. **Decision Trees**: Divides customers based on a series of hierarchical decisions.
6. **Neural Networks**: Learn complex patterns in data through interconnected layers of nodes.

We will use K-means algorithm to segment customers into various groups.

## Implementing K-means Clustering Algorithm

K-means clustering is an unsupervised algorithm. It operates without any predefined labels or training examples. This algorithm is used to group similar data points in a dataset. The goal is to divide the data into clusters. Each cluster contains similar data points. Let’s see how this algorithm works.

1. **Initialization**: Choose the number of clusters (k). Initialize k points randomly as centroids.
2. **Assignment**: Assign each data point to the nearest centroid and form the clusters.
3. **Update Centroids**: Calculate the mean of all data points assigned to each centroid. Move the centroid to this mean position.

Repeat steps 2 and 3 until convergence.

In the following sections, we are going to implement K-means clustering algorithm to group customers into clusters according to different features.

## Data Preparation

Let’s explore the customer dataset. Our dataset has around 5,00,000 data points.

![Customer dataset](https://www.kdnuggets.com/wp-content/uploads/Screenshot-84-1.png)

Customer dataset  

The missing values and duplicates are removed and three features (‘Quantity’, ‘UnitPrice’, ‘CustomerID’) are selected for clustering.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'data.csv' with your actual file path)
df = pd.read_csv('userdata.csv')

# Data cleaning: remove duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna()

# Feature selection: selecting relevant features for clustering
selected_features = ['Quantity', 'UnitPrice', 'CustomerID']
X = df[selected_features]

# Normalization (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

```

## Hyperparameter Tuning

One challenge in K-means clustering is to find out the optimal number of clusters. The elbow method help us in doing so. It plots the sum of squared distances from each point to its assigned cluster centroid (inertia) against K. T Look for the point where the inertia no longer decreases significantly with increasing K. This point is called the elbow of the clustering model. It suggests a suitable K value.

```Python
# Determine the optimal number of clusters using the Elbow Method
def determine_optimal_clusters(X_scaled, max_clusters=10):
    distances = []

    for n in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(X_scaled)
        distances.append(kmeans.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(range(2, max_clusters+1), distances, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.xticks(range(2, max_clusters+1))
    plt.grid(True)
    plt.show()

    return distances

distances = determine_optimal_clusters(X_scaled)
```

We can generate an inertia vs number of clusters plot using the above code.

![Elbow method](https://www.kdnuggets.com/wp-content/uploads/Elbow-method-1.png)

Elbow method  

At K=1, inertia is at the highest. From K=1 to K=5, the inertia decreases steeply. Between K=5 to K=7, the curve decreases gradually. Finally, at K=7, it becomes stable, so the optimal value of K is 7.

## Visualizing Segmentation Results

Let’s implement K-means clustering algorithm and visualize the results.

```Python
# Apply K-means clustering with the chosen number of clusters
chosen_clusters = 7  
kmeans = KMeans(n_clusters=chosen_clusters, random_state=42)
kmeans.fit(X_scaled)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add the cluster labels to the original dataframe
df['Cluster'] = cluster_labels

# Visualize the clusters in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
for cluster in range(chosen_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data['Quantity'], cluster_data['UnitPrice'], cluster_data['CustomerID'],
               label=f'Cluster {cluster}', s=50)

ax.set_xlabel('Quantity')
ax.set_ylabel('UnitPrice')
ax.set_zlabel('CustomerID')

# Add a legend
ax.legend()
plt.show()
```

![Scatter plot](https://www.kdnuggets.com/wp-content/uploads/Visualization-1.png)

The 3D scatter plot visualizes the clusters based on ‘Quantity’, ‘UnitPrice’, and ‘CustomerID’. Each cluster is differentiated by color and labeled accordingly.

## Conclusion

We have discussed customer segmentation using machine learning and its benefits. Furthermore, we showed how to implement the K-means algorithm to segment customers into different groups. First, we found a suitable number of clusters using the elbow method. Then, we implemented the K-means algorithm and visualized the results using a scatter plot. Through these steps, companies can segment customers into groups efficiently.

Ver tradução: [[Using Machine Learning in Customer Segmentation_translated]]



