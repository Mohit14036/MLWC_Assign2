
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
TRAIN_SNR = 25
dataset_dir = (
    Path(__file__).resolve().parent.parent
    / "NLOS_LOS_Classification"
    / "Dataset"
)
augmented_path = dataset_dir / "los_nlos_dataset_with_features.csv"
part_b_results_path = dataset_dir / "svm_feature_analysis_results.csv"

FEATURE_COLS = [
    "kurtosis",
    "skewness",
    "rise_time_ns",
    "rms_delay_spread_ns",
    "rician_k_factor",
]

df = pd.read_csv(augmented_path)

df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

part_b_results = pd.read_csv(part_b_results_path)

df_train_snr = df[df["snr_db"] == TRAIN_SNR]

X_train_snr = df_train_snr[FEATURE_COLS]
y_train_snr = df_train_snr["label"]

X_train_fixed, _, y_train_fixed, _ = train_test_split(
    X_train_snr, y_train_snr, test_size=0.2, random_state=RANDOM_STATE,
    stratify=y_train_snr
)

scaler_fixed = StandardScaler()
X_train_fixed_scaled = scaler_fixed.fit_transform(X_train_fixed)

fixed_model = SVC(kernel="rbf", C=1)
fixed_model.fit(X_train_fixed_scaled, y_train_fixed)

print(f"Trained fixed SVM-6 model on {len(X_train_fixed)} samples from SNR = {TRAIN_SNR} dB.")

snr_values = sorted(df["snr_db"].unique())
fixed_model_results = []

for snr in snr_values:
    df_snr = df[df["snr_db"] == snr]
    X_all = df_snr[FEATURE_COLS]
    y = df_snr["label"]

    
    _, X_test, _, y_test = train_test_split(X_all, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    X_test_scaled = scaler_fixed.transform(X_test)

    y_pred = fixed_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    fixed_model_results.append({"snr_db": snr, "accuracy_train25": acc})
    print(f"SNR = {snr} dB -> accuracy (model trained at 25 dB): {acc:.4f}")

fixed_results_df = pd.DataFrame(fixed_model_results)

merged = pd.merge(
    part_b_results[["snr_db", "SVM-6_all_features"]],
    fixed_results_df,
    on="snr_db",
)
merged = merged.rename(columns={"SVM-6_all_features": "accuracy_matched"})

print("\n=== Comparison: Train=Test SNR vs. Train at 25 dB ===")
print(merged.to_string(index=False))

output_csv = dataset_dir / "part_c_train25_vs_matched_results.csv"
merged.to_csv(output_csv, index=False)
print(f"\nSaved comparison table to: {output_csv}")

plt.figure(figsize=(9, 6))
plt.plot(
    merged["snr_db"], merged["accuracy_matched"] * 100,
    marker="o", linewidth=2, label="Train = Test SNR (SVM-6, Part b)"
)
plt.plot(
    merged["snr_db"], merged["accuracy_train25"] * 100,
    marker="s", linewidth=2, label=f"Train at {TRAIN_SNR} dB only"
)
plt.axvline(TRAIN_SNR, color="gray", linestyle="--", alpha=0.5,
            label=f"Training SNR = {TRAIN_SNR} dB")

plt.xlabel("Test SNR (dB)")
plt.ylabel("Classification Accuracy (%)")
plt.title("SVM-6 (All Features): Matched-SNR Training vs. Fixed 25 dB Training")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_fig = dataset_dir / "part_c_train25_vs_matched.png"
plt.savefig(output_fig, dpi=150)
print(f"Saved plot to: {output_fig}")

plt.show()
