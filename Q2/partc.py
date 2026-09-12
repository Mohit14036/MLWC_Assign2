
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

RANDOM_STATE = 42
K = 16

df = pd.read_csv("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\qam16_awgn_dataset.csv")
snr_values = sorted(df["SNR_dB"].unique())

def cluster_purity(cluster_pred, y_true):
    df_tmp = pd.DataFrame({"cluster": cluster_pred, "true": y_true})
    correct = 0
    for c in np.unique(cluster_pred):
        subset = df_tmp[df_tmp["cluster"] == c]
        majority_count = subset["true"].value_counts().iloc[0]
        correct += majority_count
    return correct / len(y_true)

adaptive_purity = []

for snr in snr_values:
    df_snr = df[df["SNR_dB"] == snr]
    X = df_snr[["Rx_I", "Rx_Q"]].values
    y_true = df_snr["Symbol_Idx"].values

    km = KMeans(n_clusters=K, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    purity = cluster_purity(labels, y_true)
    adaptive_purity.append(purity)

df_25 = df[df["SNR_dB"] == 25]
X_25 = df_25[["Rx_I", "Rx_Q"]].values

km_template = KMeans(n_clusters=K, init="k-means++", n_init=10,
                      random_state=RANDOM_STATE)
km_template.fit(X_25)   
fixed_purity = []

for snr in snr_values:
    df_snr = df[df["SNR_dB"] == snr]
    X = df_snr[["Rx_I", "Rx_Q"]].values
    y_true = df_snr["Symbol_Idx"].values

    labels = km_template.predict(X)   
    purity = cluster_purity(labels, y_true)
    fixed_purity.append(purity)

results = pd.DataFrame({
    "SNR_dB": snr_values,
    "Adaptive_KMeans_Purity": adaptive_purity,
    "Fixed_Template_Purity": fixed_purity
})
print(results.to_string(index=False))
results.to_csv("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\adaptive_vs_fixed_purity.csv", index=False)

plt.figure(figsize=(8, 6))
plt.plot(snr_values, adaptive_purity, marker="o", linewidth=2,
         color="steelblue", label="Adaptive K-Means (fresh fit per SNR)")
plt.plot(snr_values, fixed_purity, marker="s", linewidth=2,
         color="darkorange", label="Fixed Template (centroids from 25 dB)")
plt.xlabel("SNR (dB)")
plt.ylabel("Cluster Purity")
plt.title("Cluster Purity vs SNR: Adaptive K-Means vs Fixed Template (K=16)")
plt.xticks(snr_values)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\adaptive_vs_fixed_purity.png", dpi=150)
plt.close()
print("\nSaved: adaptive_vs_fixed_purity.png")