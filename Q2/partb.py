import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
RANDOM_STATE = 42

df = pd.read_csv("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\qam16_awgn_dataset.csv")
df_25 = df[df["SNR_dB"] == 25].reset_index(drop=True)

X_cart = df_25[["Rx_I", "Rx_Q"]].values
y_true = df_25["Symbol_Idx"].values   
k_values = list(range(2, 21))
inertias = []
silhouettes = []

for k in k_values:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X_cart)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cart, labels))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(k_values, inertias, marker="o", color="steelblue")
axes[0].axvline(x=16, color="red", linestyle="--", alpha=0.6, label="K=16 (true # symbols)")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)")
axes[0].set_title("Elbow Plot: Inertia vs K")
axes[0].set_xticks(k_values)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(k_values, silhouettes, marker="o", color="darkorange")
axes[1].axvline(x=16, color="red", linestyle="--", alpha=0.6, label="K=16 (true # symbols)")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Coefficient")
axes[1].set_title("Silhouette Score vs K")
axes[1].set_xticks(k_values)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\inertia_silhouette_vs_k.png", dpi=150)
plt.close()
print("Saved: inertia_silhouette_vs_k.png")

km16 = KMeans(n_clusters=16, init="k-means++", n_init=10,
              random_state=RANDOM_STATE)
cluster_labels_16 = km16.fit_predict(X_cart)
centroids_16 = km16.cluster_centers_

plt.figure(figsize=(7, 7))
scatter = plt.scatter(X_cart[:, 0], X_cart[:, 1], c=cluster_labels_16,
                       cmap="tab20", s=8, alpha=0.5)
plt.scatter(centroids_16[:, 0], centroids_16[:, 1],
            c="black", marker="X", s=200, edgecolors="white",
            linewidths=1.5, label="Centroids", zorder=5)
plt.xlabel("Rx_I")
plt.ylabel("Rx_Q")
plt.title("K-means (K=16) on Cartesian Features, SNR = 25 dB")
plt.legend()
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.grid(alpha=0.3)
plt.axis("equal")
plt.tight_layout()
plt.savefig("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\kmeans16_cartesian_scatter.png", dpi=150)
plt.close()
print("Saved: kmeans16_cartesian_scatter.png")

def cluster_purity(cluster_pred, y_true):
    
    df_tmp = pd.DataFrame({"cluster": cluster_pred, "true": y_true})
    correct = 0
    for c in np.unique(cluster_pred):
        subset = df_tmp[df_tmp["cluster"] == c]
        majority_count = subset["true"].value_counts().iloc[0]
        correct += majority_count
    return correct / len(y_true)

X_set1 = df_25[["Rx_I", "Rx_Q"]].values
km_set1 = KMeans(n_clusters=16, init="k-means++", n_init=10,
                  random_state=RANDOM_STATE)
labels_set1 = km_set1.fit_predict(X_set1)
purity_set1 = cluster_purity(labels_set1, y_true)

X_set2 = df_25[["r", "theta"]].values
km_set2 = KMeans(n_clusters=16, init="k-means++", n_init=10,
                  random_state=RANDOM_STATE)
labels_set2 = km_set2.fit_predict(X_set2)
purity_set2 = cluster_purity(labels_set2, y_true)

X_set3_raw = df_25[["Rx_I", "Rx_Q", "r", "theta"]].values
scaler = StandardScaler()
X_set3 = scaler.fit_transform(X_set3_raw)
km_set3 = KMeans(n_clusters=16, init="k-means++", n_init=10,
                  random_state=RANDOM_STATE)
labels_set3 = km_set3.fit_predict(X_set3)
purity_set3 = cluster_purity(labels_set3, y_true)

print("\n=== Cluster Purity @ SNR = 25 dB (K=16) ===")
print(f"Feature Set 1 (Rx_I, Rx_Q)              : {purity_set1:.4f}")
print(f"Feature Set 2 (r, theta)                : {purity_set2:.4f}")
print(f"Feature Set 3 (Rx_I, Rx_Q, r, theta)*    : {purity_set3:.4f}")
print("(*StandardScaler applied before clustering)")

best_set = max(
    [("Feature Set 1 (Cartesian)", purity_set1),
     ("Feature Set 2 (Polar)", purity_set2),
     ("Feature Set 3 (Combined, scaled)", purity_set3)],
    key=lambda x: x[1]
)
print(f"\nHighest purity: {best_set[0]} with purity = {best_set[1]:.4f}")