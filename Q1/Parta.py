from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as stats


dataset_path = (
    Path(__file__).resolve().parent.parent
    / "NLOS_LOS_Classification"
    / "Dataset"
    / "los_nlos_dataset.csv"
)
df = pd.read_csv(dataset_path)
print(df.head())

realCols = ["h_real_0", "h_real_1", "h_real_2", "h_real_3", "h_real_4", "h_real_5"]
imagCols = ["h_imag_0", "h_imag_1", "h_imag_2", "h_imag_3", "h_imag_4", "h_imag_5"]
tauCols  = ["tau_0", "tau_1", "tau_2", "tau_3", "tau_4", "tau_5"]

real_array = df[realCols].to_numpy()
imag_array = df[imagCols].to_numpy()
tau_array  = df[tauCols].to_numpy()

P = real_array**2 + imag_array**2

print("P shape:", P.shape)
print("P values are non-negative:", np.all(P >= 0))
print("Mean total power:", P.sum(axis=1).mean())

#Kurtosis & Skewness of received power
df['kurtosis'] = stats.kurtosis(P, axis=1, fisher=True, bias=True)
df['skewness'] = stats.skew(P, axis=1, bias=True)

# Rising time (10% -> 90% cumulative energy)
order = np.argsort(tau_array, axis=1)
tau_sorted = np.take_along_axis(tau_array, order, axis=1)
P_sorted   = np.take_along_axis(P, order, axis=1)


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
df['rise_time_ns'] = t90 - t10
# RMS delay spread 
total_energy_flat = total_energy.flatten()
mean_tau = np.sum(P * tau_array, axis=1) / total_energy_flat
rms_ds = np.sqrt(
    np.sum(P * (tau_array - mean_tau[:, None])**2, axis=1) / total_energy_flat
)
df['rms_delay_spread_ns'] = rms_ds

# Rician K-factor (moment-based estimator on tap powers)

P_dominant = P.max(axis=1)
P_scattered = P.sum(axis=1) - P_dominant

with np.errstate(divide='ignore', invalid='ignore'):
    k_factor = np.where(P_scattered > 0, P_dominant / P_scattered, np.inf)

df['rician_k_factor'] = k_factor
feature_cols = ['kurtosis', 'skewness', 'rise_time_ns', 'rms_delay_spread_ns', 'rician_k_factor']
print(df[feature_cols].describe())
print(df.groupby('label')[feature_cols].mean())

output_path = dataset_path.parent / "los_nlos_dataset_with_features.csv"
df.to_csv(output_path, index=False)
print(f"Saved augmented dataset to: {output_path}")
