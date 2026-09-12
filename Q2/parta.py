import numpy as np
import pandas as pd

SEED = 67
np.random.seed(SEED)

levels = np.array([-3, -1, 1, 3])
I_grid, Q_grid = np.meshgrid(levels, levels)
I_grid = I_grid.flatten()
Q_grid = Q_grid.flatten()

constellation_raw = I_grid + 1j * Q_grid       

avg_energy_raw = np.mean(I_grid**2 + Q_grid**2)  
scale = 1.0 / np.sqrt(avg_energy_raw)

constellation = constellation_raw * scale          

snr_db_list = [0, 5, 10, 15, 20, 25, 30]
samples_per_point = 200
Es = 1.0   

records = []

for snr_db in snr_db_list:
    snr_linear = 10 ** (snr_db / 10.0)
    N0 = Es / snr_linear                
    noise_std_per_dim = np.sqrt(N0 / 2.0) 
    for idx, sym in enumerate(constellation):
        tx_I = sym.real
        tx_Q = sym.imag
        noise_I = np.random.normal(0.0, noise_std_per_dim, samples_per_point)
        noise_Q = np.random.normal(0.0, noise_std_per_dim, samples_per_point)
        rx_I = tx_I + noise_I
        rx_Q = tx_Q + noise_Q
        for n in range(samples_per_point):
            records.append({
                "Symbol_Idx": idx,
                "Tx_I": tx_I,
                "Tx_Q": tx_Q,
                "SNR_dB": snr_db,
                "Rx_I": rx_I[n],
                "Rx_Q": rx_Q[n],
            })

df = pd.DataFrame(records)

df["r"] = np.sqrt(df["Rx_I"]**2 + df["Rx_Q"]**2)
df["theta"] = np.arctan2(df["Rx_Q"], df["Rx_I"])

print("Dataset shape:", df.shape)
print("Expected rows:", len(snr_db_list) * len(constellation) * samples_per_point)
print("\nColumns:", list(df.columns))
print("\nSample rows:\n", df.sample(5, random_state=SEED))
print("\nAverage received symbol energy per SNR (sanity check ~ decreases as noise ~ constant Es=1):")
print(df.groupby("SNR_dB")[["Rx_I", "Rx_Q"]].apply(lambda g: np.mean(g["Rx_I"]**2 + g["Rx_Q"]**2)))
df.to_csv("D:\\IIITB\\Term7\\MLWC\\Assign2\\NLOS_LOS_Classification\\Dataset\\qam16_awgn_dataset.csv", index=False)
print("\nSaved to qam16_awgn_dataset.csv")