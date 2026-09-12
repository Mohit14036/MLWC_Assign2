from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
dataset_dir = (
    Path(__file__).resolve().parent.parent
    / "NLOS_LOS_Classification"
    / "Dataset"
)
raw_path = dataset_dir / "los_nlos_dataset.csv"
augmented_path = dataset_dir / "los_nlos_dataset_with_features.csv"

FEATURE_COLS = [
    "kurtosis",
    "skewness",
    "rise_time_ns",
    "rms_delay_spread_ns",
    "rician_k_factor",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    realCols = [f"h_real_{i}" for i in range(6)]
    imagCols = [f"h_imag_{i}" for i in range(6)]
    tauCols = [f"tau_{i}" for i in range(6)]

    real_array = df[realCols].to_numpy()
    imag_array = df[imagCols].to_numpy()
    tau_array = df[tauCols].to_numpy()

    P = real_array**2 + imag_array**2
    df["kurtosis"] = stats.kurtosis(P, axis=1, fisher=True, bias=True)
    df["skewness"] = stats.skew(P, axis=1, bias=True)
    order = np.argsort(tau_array, axis=1)
    tau_sorted = np.take_along_axis(tau_array, order, axis=1)
    P_sorted = np.take_along_axis(P, order, axis=1)

    cum_energy = np.cumsum(P_sorted, axis=1)
    total_energy = cum_energy[:, -1:]
    cum_norm = cum_energy / total_energy

    def interp_threshold(cum_norm, tau_sorted, thresh):
        n_rows = cum_norm.shape[0]
        result = np.empty(n_rows)
        for i in range(n_rows):
            result[i] = np.interp(thresh, cum_norm[i], tau_sorted[i])
        return result

    t10 = interp_threshold(cum_norm, tau_sorted, 0.10)
    t90 = interp_threshold(cum_norm, tau_sorted, 0.90)
    df["rise_time_ns"] = t90 - t10

    total_energy_flat = total_energy.flatten()
    mean_tau = np.sum(P * tau_array, axis=1) / total_energy_flat
    rms_ds = np.sqrt(
        np.sum(P * (tau_array - mean_tau[:, None]) ** 2, axis=1) / total_energy_flat
    )
    df["rms_delay_spread_ns"] = rms_ds

    P_dominant = P.max(axis=1)
    P_scattered = P.sum(axis=1) - P_dominant
    with np.errstate(divide="ignore", invalid="ignore"):
        k_factor = np.where(P_scattered > 0, P_dominant / P_scattered, np.inf)
    df["rician_k_factor"] = k_factor

    return df


if augmented_path.exists():
    df = pd.read_csv(augmented_path)
else:
    df = pd.read_csv(raw_path)
    df = compute_features(df)
    df.to_csv(augmented_path, index=False)

df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
n_before = len(df)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
n_after = len(df)

SVM_CONFIGS = {
    "SVM-1_kurtosis": ["kurtosis"],
    "SVM-2_skewness": ["skewness"],
    "SVM-3_rise_time": ["rise_time_ns"],
    "SVM-4_rms_delay_spread": ["rms_delay_spread_ns"],
    "SVM-5_rician_k_factor": ["rician_k_factor"],
    "SVM-6_all_features": FEATURE_COLS,
}

snr_values = sorted(df["snr_db"].unique())
results = []

for snr in snr_values:
    df_snr = df[df["snr_db"] == snr]

    X_all = df_snr[FEATURE_COLS]
    y = df_snr["label"]

    X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    row = {"snr_db": snr, "n_samples": len(df_snr)}

    for svm_name, cols in SVM_CONFIGS.items():
        X_train = X_train_all[cols].to_numpy()
        X_test = X_test_all[cols].to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = SVC(kernel="rbf", C=1)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        row[svm_name] = acc

    results.append(row)
    print(f"SNR = {snr} dB done.")

results_df = pd.DataFrame(results)
print("\n=== Test accuracy per SVM per SNR ===")
print(results_df.to_string(index=False))

results_path = dataset_dir / "svm_feature_analysis_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nSaved results to: {results_path}")


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

dataset_dir = (
    Path(__file__).resolve().parent.parent
    / "NLOS_LOS_Classification"
    / "Dataset"
)
results_path = dataset_dir / "svm_feature_analysis_results.csv"

df = pd.read_csv(results_path)

svm_cols = [
    "SVM-1_kurtosis",
    "SVM-2_skewness",
    "SVM-3_rise_time",
    "SVM-4_rms_delay_spread",
    "SVM-5_rician_k_factor",
    "SVM-6_all_features",
]

labels = {
    "SVM-1_kurtosis": "SVM-1: Kurtosis",
    "SVM-2_skewness": "SVM-2: Skewness",
    "SVM-3_rise_time": "SVM-3: Rise time",
    "SVM-4_rms_delay_spread": "SVM-4: RMS delay spread",
    "SVM-5_rician_k_factor": "SVM-5: Rician K-factor",
    "SVM-6_all_features": "SVM-6: All features",
}

markers = ["o", "s", "^", "D", "v", "*"]

plt.figure(figsize=(9, 6))

for col, marker in zip(svm_cols, markers):
    plt.plot(
        df["snr_db"],
        df[col] * 100,
        marker=marker,
        label=labels[col],
        linewidth=2 if col == "SVM-6_all_features" else 1.5,
        markersize=7,
    )

plt.xlabel("SNR (dB)")
plt.ylabel("Classification Accuracy (%)")
plt.title("LOS/NLOS Classification Accuracy vs. SNR (per feature vs. combined)")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = dataset_dir / "accuracy_vs_snr.png"
plt.savefig(output_path, dpi=150)
print(f"Saved plot to: {output_path}")
plt.show()
