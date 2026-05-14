import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, y = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    random_state=42
)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.show()

best_k = int(input("Enter best K from elbow graph: "))

final_model = KMeans(
    n_clusters=best_k,
    random_state=42
)

final_model.fit(X)

labels = final_model.labels_

df['Cluster'] = labels

print(df)

score = silhouette_score(X, labels)

print("\nSilhouette Score:", score)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.xlabel("Feature1")
plt.ylabel("Feature2")

plt.title("KMeans Clusters")

plt.show()