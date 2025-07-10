import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt


#dde.config.set_random_seed(seed)
np.random.seed(5)
torch.manual_seed(7)

def space_dampening(x, A=0.5):
    return np.exp(-x*A)

def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Carica il dataset dell'equazione del calore da file .npz
    data = np.load("heat_eq_data.npz")
    # Estrai i vettori temporali t, spaziali x e la soluzione esatta (trasponendo usol)
    t, x, exact = data["t"], data["x"], data["usol"].T
    # Crea una griglia 2D di coordinate (x,t)
    xx, tt = np.meshgrid(x, t)
    # Prepara la matrice X di shape (N,2) con tutte le coppie (x,t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    # Prepara il vettore y di shape (N,1) con i valori della soluzione corrispondenti
    y = exact.flatten()[:, None]
    return X, y


# === PLOT DELLE SOLUZIONI SULLA GRIGLIA ===
# Ricarica i dati per ricostruire la griglia spazio-temporale
data = np.load("heat_eq_data.npz")
# Estrai t e x
t = data["t"].flatten()
x = data["x"].flatten()
# Ricrea la griglia 2D e i punti per la predizione
xx, tt = np.meshgrid(x, t)
grid = np.vstack((xx.ravel(), tt.ravel())).T
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Soluzione esatta (transposta come in gen_testdata)
exact_1D = data["usol"].T
print(data["usol"].shape)
print(space_dampening(xx).shape)
real_3D =exact_1D*space_dampening(xx)
diff = np.abs(exact_1D - real_3D) #DA CALCOLARE LA NORMA
l2_real=np.linalg.norm(diff)
print(l2_real)

np.savez("heat_eq_damp_A05", x=x, t=t, usol=real_3D)

# Plot a 3 pannelli: esatta, predetta e differenza
# Crea una figura con 3 pannelli affiancati per confronto
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Pannello 1: mappa di calore della soluzione esatta
im0 = axes[0].imshow(exact_1D, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', origin='lower')
axes[0].set_title("exact_1D")
axes[0].set_xlabel("Spazio x")
axes[0].set_ylabel("Tempo t")
fig.colorbar(im0, ax=axes[0])
# Pannello 2: mappa di calore della soluzione predetta dal modello
im1 = axes[1].imshow(real_3D, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', origin='lower')
axes[1].set_title("dampening 3D")
axes[1].set_xlabel("Spazio x")
axes[1].set_ylabel("Tempo t")
fig.colorbar(im1, ax=axes[1])
# Pannello 3: mappa di calore della differenza assoluta tra esatto e predetto
im2 = axes[2].imshow(diff, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', origin='lower')
axes[2].set_title("Differenza di temperatura")
axes[2].set_xlabel("Spazio x")
axes[2].set_ylabel("Tempo t")
fig.colorbar(im2, ax=axes[2])
plt.tight_layout()
# Ottimizza il layout dei 3 pannelli
plt.show(block=False)
# Visualizza il grafico a tre pannelli senza bloccare
plt.pause(0.1)
# Breve pausa per permettere il rendering dei pannelli
input("Premi Invio per chiudere i grafici...")
# Attende un input per mantenere aperte le finestre finché si desidera

# # === SALVA I PESI DEL MODELLO GRU TRAINATO ===
# torch.save(model.state_dict(), "gru_trained.pth")