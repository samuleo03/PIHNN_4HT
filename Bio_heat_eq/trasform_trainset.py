import numpy as np
import os

# Carica i dati ed escludi l'ultima colonna
file_path = os.path.join(os.path.dirname(__file__), "train_cool.txt")
a = np.loadtxt(file_path)[:, :5]

# Estrai tempo e le quattro misure (x=0, 0.143, 0.571, 1.0)
t_raw = a[:, 0]
meas = a[:, [4, 3, 2, 1]]  # ordinati da x=0 a x=1

# Costruisci i vettori 1D
num_times = len(t_raw)
x = np.tile([0.0, 0.143, 0.571, 1.0], num_times).reshape(-1, 1)
t = np.repeat(t_raw, 4).reshape(-1, 1)
y = meas.flatten().reshape(-1, 1)

# Salva il file .npz
save_path = os.path.join(os.path.dirname(__file__), "misure_trainset.npz")
np.savez(save_path, x=x, t=t, y=y)
