import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

#dde.config.set_random_seed(seed)
np.random.seed(5)
torch.manual_seed(7)

def gen_testdata(path_npz="misure_dataset_2.npz"):
    """Importa e pre-processa il dataset da file .npz (t, x, y).
    Supporta due formati:
      1) Griglia densa: t.shape=(n_t,), x.shape=(n_x,), y.shape=(n_t,n_x)
      2) Lista di campioni: t.shape=(N,), x.shape=(N,), y.shape=(N,)
    Restituisce X di shape (N,2), y di shape (N,1), e i vettori grezzi t_raw, x_raw (entrambi di lunghezza N).
    """
    import os
    base_dir = os.path.dirname(__file__)
    npz_path = os.path.join(base_dir, path_npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ non trovato: {npz_path}")
    data = np.load(npz_path)
    if not set(["t", "x", "y"]).issubset(set(data.files)):
        raise ValueError("misure_dataset_2.npz deve contenere le chiavi: 't', 'x', 'y'")
    t = data["t"]
    x = data["x"]
    y = data["y"]

    # Normalizza dimensioni
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)

    # Caso 1: y è una matrice (n_t, n_x) e t/x sono gli assi
    if y.ndim == 2 and t.ndim == 1 and x.ndim == 1 and y.shape == (t.size, x.size):
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((xx.ravel(), tt.ravel())).T  # (x,t)
        y_vec = y.reshape(-1)[:, None]
        # t_raw/x_raw come lista dei campioni (stessa lunghezza di y_vec)
        t_raw = tt.ravel()
        x_raw = xx.ravel()
        return X, y_vec, t_raw, x_raw

    # Caso 2: lista di campioni allineati
    # accetta sia shape (N,) che (N,1) per y, t, x
    t_flat = t.reshape(-1)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    if not (t_flat.size == x_flat.size == y_flat.size):
        raise ValueError("Formati non coerenti: le lunghezze di t, x, y devono coincidere oppure y deve essere (n_t,n_x)")
    X = np.vstack((x_flat, t_flat)).T  # (x,t)
    y_vec = y_flat[:, None]
    return X, y_vec, t_flat, x_flat


# Definizione del GRU da zero con parametri manuali
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens), init_weight(num_hiddens, num_hiddens), nn.Parameter(torch.zeros(num_hiddens)))
        # Inizializza i pesi e bias per l'update gate
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        # Inizializza i pesi e bias per il reset gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        # Inizializza i pesi e bias per lo stato nascosto candidato
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state


# Implementazione della forward pass del GRU scratch
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    outputs = []
    for X in inputs:
        # Calcola il gate di aggiornamento Z
        Z = torch.sigmoid(torch.matmul(X, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z)
        # Calcola il gate di reset R
        R = torch.sigmoid(torch.matmul(X, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r)
        # Calcola lo stato nascosto candidato H_tilde
        H_tilde = torch.tanh(torch.matmul(X, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h)
        # Combina i gate per aggiornare lo stato nascosto H
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H

# Import dei moduli per il DataLoader e TensorDataset
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# Attiva la modalità interattiva di Matplotlib per show non bloccante
plt.ion()

# Generazione dei dati: TRAIN su misure_trainset.npz, TEST su misure_dataset_1.npz e misure_dataset_2.npz
X_train_np, y_train_np, t_train_raw, x_train_raw = gen_testdata("misure_trainset.npz")
X_test1_np, y_test1_np, t_test1_raw, x_test1_raw = gen_testdata("misure_dataset_1.npz")
X_test2_np, y_test2_np, t_test2_raw, x_test2_raw = gen_testdata("misure_dataset_2.npz")

# Parametri
num_inputs = 2
num_hiddens = 64
num_epochs = 100
batch_size = 256
learning_rate = 0.001

# Caricamento dati (senza split casuale): train/test da file separati
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test1 = torch.tensor(X_test1_np, dtype=torch.float32)
y_test1 = torch.tensor(y_test1_np, dtype=torch.float32)
X_test2 = torch.tensor(X_test2_np, dtype=torch.float32)
y_test2 = torch.tensor(y_test2_np, dtype=torch.float32)

# Indici sequenziali per compatibilità con altri moduli
train_indices = torch.arange(X_train.shape[0])
test1_indices = torch.arange(X_test1.shape[0])
test2_indices = torch.arange(X_test2.shape[0])

# Creazione dei DataLoader per training e test
train_dataset = TensorDataset(X_train, y_train)
test_dataset1 = TensorDataset(X_test1, y_test1)
test_dataset2 = TensorDataset(X_test2, y_test2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

# Definizione del modello GRU integrato con layer lineare finale
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq_len=1, input_size)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
    
import torch
import numpy as np

def load_gru_data(split="train"):
    """
    Restituisce X_gru e y_true come torch.Tensor
    """
    if split == "train":
        return X_train.clone(), y_train.clone()
    else:
        return X_test2.clone(),  y_test2.clone()

def get_gru_indices():
    """
    Restituisce due array di indici (train_idx, test_idx) riferiti ai tensori correnti.
    """
    return train_indices.numpy(), test2_indices.numpy()

model = GRURegressor(num_inputs, num_hiddens)
# Definizione della funzione di perdita MSE
loss_fn = nn.MSELoss()
# Definizione dell'ottimizzatore Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Liste per salvare le loss di training e test per ogni epoca
train_losses = []
test1_losses = []
test2_losses = []

# Stampa della loss iniziale a epoca 0
# Initial evaluation at epoch 0
# compute train loss at epoch 0
total_train_loss = 0
with torch.no_grad():
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_train_loss += loss.item() * xb.size(0)
avg_train_loss = total_train_loss / len(train_loader.dataset)

# compute test1 loss at epoch 0
total_test1_loss = 0
with torch.no_grad():
    for xb, yb in test_loader1:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_test1_loss += loss.item() * xb.size(0)
avg_test1_loss = total_test1_loss / len(test_loader1.dataset)

# compute test2 loss at epoch 0
total_test2_loss = 0
with torch.no_grad():
    for xb, yb in test_loader2:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_test2_loss += loss.item() * xb.size(0)
avg_test2_loss = total_test2_loss / len(test_loader2.dataset)

print(f"Epoch 0/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test1 Loss: {avg_test1_loss:.6f}, Test2 Loss: {avg_test2_loss:.6f}")

for epoch in range(num_epochs):
    # Inizio fase di training per l'epoca corrente
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        # Backpropagation dell'errore
        optimizer.zero_grad()
        loss.backward()
        # Aggiorna i pesi del modello
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Valutazione
    # Modalità valutazione: disabilita dropout e autograd
    model.eval()
    total_test1_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader1:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_test1_loss += loss.item() * xb.size(0)
    avg_test1_loss = total_test1_loss / len(test_loader1.dataset)
    test1_losses.append(avg_test1_loss)

    total_test2_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader2:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_test2_loss += loss.item() * xb.size(0)
    avg_test2_loss = total_test2_loss / len(test_loader2.dataset)
    test2_losses.append(avg_test2_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test1 Loss: {avg_test1_loss:.6f}, Test2 Loss: {avg_test2_loss:.6f}")

# === PLOT DELLE LOSS ===
# Impostazione di una nuova figura per visualizzare l'andamento della loss nel tempo
plt.figure(figsize=(12, 6))
# Crea una figura di dimensioni 12x6 pollici
epochs = range(1, num_epochs + 1)
# Definisce le epoche sull'asse x
plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
# Disegna la curva della loss di training con marker circolari
plt.plot(epochs, test1_losses, label='Test1 Loss', marker='s', markersize=4)
# Disegna la curva della loss di test1 con marker quadrati
plt.plot(epochs, test2_losses, label='Test2 Loss', marker='^', markersize=4)
# Disegna la curva della loss di test2 con marker triangolari
plt.xlabel('Epoche')
# Etichetta l'asse x
plt.ylabel('Loss (MSE)')
# Etichetta l'asse y
plt.title('Evoluzione della Loss durante il training e il test')
# Aggiunge un titolo al grafico
plt.legend()
# Mostra la legenda delle curve
plt.yscale('log')  # Scala logaritmica per visualizzare meglio le variazioni
# Imposta scala logaritmica sull'asse y per evidenziare le variazioni
plt.grid(True, which="both", ls="--", linewidth=0.5)
# Aggiunge una griglia leggera al grafico
plt.tight_layout()
# Ottimizza il layout per evitare sovrapposizioni
plt.show(block=False)
# Visualizza il grafico delle loss senza bloccare l'esecuzione
plt.pause(0.1)
# Breve pausa per permettere il rendering della finestra
# Visualizzazione della soluzione esatta, predetta e loro differenza


# === PLOT DELLE SOLUZIONI SULLA GRIGLIA ===
# Dataset 1
# Ricostruisce una griglia (t,x) a partire dai campioni del file di misura
# 1) costruisci griglia regolare dalle coppie osservate
x_unique1 = np.unique(x_test1_raw)
t_unique1 = np.unique(t_test1_raw)
xx1, tt1 = np.meshgrid(x_unique1, t_unique1)
grid1 = np.vstack((xx1.ravel(), tt1.ravel())).T
grid_tensor1 = torch.tensor(grid1, dtype=torch.float32)

# 2) Predizioni del modello su tutta la griglia
model.eval()
with torch.no_grad():
    preds1 = model(grid_tensor1).numpy().reshape(len(t_unique1), len(x_unique1))

# 3) Costruisci la matrice "exact" dalle misure (pivot senza pandas)
# inizializza con NaN per gestire buchi; useremo l'ultimo valore se duplicati
exact1 = np.full((len(t_unique1), len(x_unique1)), np.nan, dtype=float)
# mapping da valore a indice
x_to_idx1 = {val: i for i, val in enumerate(x_unique1)}
t_to_idx1 = {val: i for i, val in enumerate(t_unique1)}
for tt_val, xx_val, yy_val in zip(t_test1_raw, x_test1_raw, y_test1_np.flatten()):
    i = t_to_idx1.get(tt_val)
    j = x_to_idx1.get(xx_val)
    if i is not None and j is not None:
        exact1[i, j] = yy_val
# se ci sono NaN, prova a riempire con interpolazione semplice lungo x
# (fallback: copia vicino più prossimo)
if np.isnan(exact1).any():
    for i in range(exact1.shape[0]):
        row = exact1[i]
        if np.isnan(row).all():
            continue
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            row_interp = np.interp(np.arange(len(row)), np.where(valid)[0], row[valid])
            exact1[i] = row_interp
        else:
            fill_val = row[valid][0] if valid.any() else 0.0
            exact1[i] = np.full_like(row, fill_val)

# 4) Calcolo L2 (norma di Frobenius) senza assoluto preventivo
l2_gru1 = np.linalg.norm(exact1 - preds1)
print(f"L2 error (GRU vs misure) Dataset 1: {l2_gru1}")

# 5) Plot a 3 pannelli
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
im0_1 = axes1[0].imshow(exact1, extent=[x_unique1.min(), x_unique1.max(), t_unique1.min(), t_unique1.max()],
                     aspect='auto', origin='lower')
axes1[0].set_title("Temperatura misurata (exact) Dataset 1")
axes1[0].set_xlabel("Spazio x")
axes1[0].set_ylabel("Tempo t")
fig1.colorbar(im0_1, ax=axes1[0])

im1_1 = axes1[1].imshow(preds1, extent=[x_unique1.min(), x_unique1.max(), t_unique1.min(), t_unique1.max()],
                     aspect='auto', origin='lower')
axes1[1].set_title("Temperatura predetta dalla GRU Dataset 1")
axes1[1].set_xlabel("Spazio x")
axes1[1].set_ylabel("Tempo t")
fig1.colorbar(im1_1, ax=axes1[1])

diff1 = np.abs(exact1 - preds1)
im2_1 = axes1[2].imshow(diff1, extent=[x_unique1.min(), x_unique1.max(), t_unique1.min(), t_unique1.max()],
                     aspect='auto', origin='lower')
axes1[2].set_title("Differenza |exact - pred| Dataset 1")
axes1[2].set_xlabel("Spazio x")
axes1[2].set_ylabel("Tempo t")
fig1.colorbar(im2_1, ax=axes1[2])
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)


# Dataset 2
# Ricostruisce una griglia (t,x) a partire dai campioni del file di misura
# 1) costruisci griglia regolare dalle coppie osservate
x_unique2 = np.unique(x_test2_raw)
t_unique2 = np.unique(t_test2_raw)
xx2, tt2 = np.meshgrid(x_unique2, t_unique2)
grid2 = np.vstack((xx2.ravel(), tt2.ravel())).T
grid_tensor2 = torch.tensor(grid2, dtype=torch.float32)

# 2) Predizioni del modello su tutta la griglia
model.eval()
with torch.no_grad():
    preds2 = model(grid_tensor2).numpy().reshape(len(t_unique2), len(x_unique2))

# 3) Costruisci la matrice "exact" dalle misure (pivot senza pandas)
# inizializza con NaN per gestire buchi; useremo l'ultimo valore se duplicati
exact2 = np.full((len(t_unique2), len(x_unique2)), np.nan, dtype=float)
# mapping da valore a indice
x_to_idx2 = {val: i for i, val in enumerate(x_unique2)}
t_to_idx2 = {val: i for i, val in enumerate(t_unique2)}
for tt_val, xx_val, yy_val in zip(t_test2_raw, x_test2_raw, y_test2_np.flatten()):
    i = t_to_idx2.get(tt_val)
    j = x_to_idx2.get(xx_val)
    if i is not None and j is not None:
        exact2[i, j] = yy_val
# se ci sono NaN, prova a riempire con interpolazione semplice lungo x
# (fallback: copia vicino più prossimo)
if np.isnan(exact2).any():
    for i in range(exact2.shape[0]):
        row = exact2[i]
        if np.isnan(row).all():
            continue
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            row_interp = np.interp(np.arange(len(row)), np.where(valid)[0], row[valid])
            exact2[i] = row_interp
        else:
            fill_val = row[valid][0] if valid.any() else 0.0
            exact2[i] = np.full_like(row, fill_val)

# 4) Calcolo L2 (norma di Frobenius) senza assoluto preventivo
l2_gru2 = np.linalg.norm(exact2 - preds2)
print(f"L2 error (GRU vs misure) Dataset 2: {l2_gru2}")

# 5) Plot a 3 pannelli
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
im0_2 = axes2[0].imshow(exact2, extent=[x_unique2.min(), x_unique2.max(), t_unique2.min(), t_unique2.max()],
                     aspect='auto', origin='lower')
axes2[0].set_title("Temperatura misurata (exact) Dataset 2")
axes2[0].set_xlabel("Spazio x")
axes2[0].set_ylabel("Tempo t")
fig2.colorbar(im0_2, ax=axes2[0])

im1_2 = axes2[1].imshow(preds2, extent=[x_unique2.min(), x_unique2.max(), t_unique2.min(), t_unique2.max()],
                     aspect='auto', origin='lower')
axes2[1].set_title("Temperatura predetta dalla GRU Dataset 2")
axes2[1].set_xlabel("Spazio x")
axes2[1].set_ylabel("Tempo t")
fig2.colorbar(im1_2, ax=axes2[1])

diff2 = np.abs(exact2 - preds2)
im2_2 = axes2[2].imshow(diff2, extent=[x_unique2.min(), x_unique2.max(), t_unique2.min(), t_unique2.max()],
                     aspect='auto', origin='lower')
axes2[2].set_title("Differenza |exact - pred| Dataset 2")
axes2[2].set_xlabel("Spazio x")
axes2[2].set_ylabel("Tempo t")
fig2.colorbar(im2_2, ax=axes2[2])
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# === SALVATAGGIO RISULTATI IN NPZ (cartella risultati_2) ===
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = Path(script_dir) / "risultati_2"
out_dir.mkdir(parents=True, exist_ok=True)
save_npz_path = out_dir / "gru_preds_bio_heat.npz"
# Salviamo entrambe le griglie (dataset 1 e 2) così come le abbiamo costruite
np.savez(
    save_npz_path,
    # dataset 1
    x1=x_unique1, t1=t_unique1, exact1=exact1, preds1=preds1, diff1=diff1, l2_gru1=l2_gru1,
    # dataset 2
    x2=x_unique2, t2=t_unique2, exact2=exact2, preds2=preds2, diff2=diff2, l2_gru2=l2_gru2
)
print(f"[SAVE] Risultati GRU (predizioni + L2) salvati in: {save_npz_path}")

# --- Salvataggi separati per correttezza (uno per blocco) ---
save_npz_path_1 = out_dir / "gru_preds_bio_heat_ds1.npz"
np.savez(
    save_npz_path_1,
    x=x_unique1,
    t=t_unique1,
    exact=exact1,
    preds=preds1,
    diff=diff1,
    l2_gru=l2_gru1,
)
print(f"[SAVE] Risultati GRU Dataset 1 salvati in: {save_npz_path_1}")

save_npz_path_2 = out_dir / "gru_preds_bio_heat_ds2.npz"
np.savez(
    save_npz_path_2,
    x=x_unique2,
    t=t_unique2,
    exact=exact2,
    preds=preds2,
    diff=diff2,
    l2_gru=l2_gru2,
)
print(f"[SAVE] Risultati GRU Dataset 2 salvati in: {save_npz_path_2}")
input("Premi Invio per chiudere i grafici...")

# Salvataggio pesi anche dentro risultati_2 per coerenza
weights_path = out_dir / "gru_trained.pth"
torch.save(model.state_dict(), str(weights_path))
print(f"[SAVE] Pesi GRU salvati in: {weights_path}")