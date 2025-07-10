import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

def gen_testdata():
    """Importa e pre-processa il dataset da file txt.
       Colonne: tempo | y1 | gt1 | gt2 | y2 | ignorata
       Input: (t, x, y1, gt1, gt2, y2)
       Target: gt1
    """
    data = np.loadtxt("meas_cool_1.txt")
    data = data[:, :5]  # usa solo le prime 5 colonne: t, y1, gt1, gt2, y2

    t = data[:, 0:1]     # tempo
    y1 = data[:, 1:2]    # y1
    gt1 = data[:, 2:3]   # gt1
    gt2 = data[:, 3:4]   # gt2
    y2 = data[:, 4:5]    # y2

    x_coord = np.full_like(t, 0.07)  # aggiungi la coordinata spaziale x=0.07
    X = np.hstack((t, x_coord, y1, gt1, gt2, y2))  # input = t, x, y1, gt1, gt2, y2
    y = np.hstack((y1, gt1, gt2, y2))  # target = tutte le temperature

    # Divisione 80/20 randomica
    total_rows = X.shape[0]
    indices = np.random.permutation(total_rows)
    split_index = int(total_rows * 0.8)
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


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
# plt.ion()

# Generazione dei dati di input e target
X_train, y_train, X_test, y_test = gen_testdata()
# Parametri
num_inputs = 6  # t, x, y1, gt1, gt2, y2
num_hiddens = 64
num_epochs = 100
batch_size = 256
learning_rate = 0.001

# Caricamento dati e split
X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
y_tensor_train = torch.tensor(y_train, dtype=torch.float32)
X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
y_tensor_test = torch.tensor(y_test, dtype=torch.float32)

n_train = X_tensor_train.shape[0]
indices_train = torch.randperm(n_train)

X_train, y_train = X_tensor_train[indices_train], y_tensor_train[indices_train]

n_test = X_tensor_test.shape[0]
indices_test = torch.randperm(n_test)

X_test, y_test = X_tensor_test[indices_test], y_tensor_test[indices_test]

# Creazione dei DataLoader per training e test
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definizione del modello GRU integrato con layer lineare finale
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

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
        return X_test.clone(),  y_test.clone()

def get_gru_indices():
    """
    Restituisce due array di indici (train_idx, test_idx)
    usati nel DataFusion per allineare i dati PINN.
    """
    return indices_train.numpy(), indices_test.numpy()

model = GRURegressor(num_inputs, num_hiddens)
# Definizione della funzione di perdita MSE
loss_fn = nn.MSELoss()
# Definizione dell'ottimizzatore Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Liste per salvare le loss di training e test per ogni epoca
train_losses = []
test_losses = []

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

# compute test loss at epoch 0
total_test_loss = 0
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_test_loss += loss.item() * xb.size(0)
avg_test_loss = total_test_loss / len(test_loader.dataset)

print(f"Epoch 0/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

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
    total_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
    avg_test_loss = total_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

# === PLOT DELLE LOSS ===
# Impostazione di una nuova figura per visualizzare l'andamento della loss nel tempo
plt.figure(figsize=(12, 6))
# Crea una figura di dimensioni 12x6 pollici
epochs = range(1, num_epochs + 1)
# Definisce le epoche sull'asse x
plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
# Disegna la curva della loss di training con marker circolari
plt.plot(epochs, test_losses, label='Test Loss', marker='s', markersize=4)
# Disegna la curva della loss di test con marker quadrati
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


# === PLOT DELLE SOLUZIONI SULLA GRIGLIA CON HEATMAP ===
# Costruzione del dataset completo su tutte le posizioni e tempi per predizione e confronto
data = np.loadtxt("meas_cool_1.txt")[:, :5]  # ignora la sesta colonna
t_vals = data[:, 0]                         # vettore dei tempi
temps = data[:, 1:5]                        # y1, gt1, gt2, y2

x_positions = np.array([0.07, 0.04, 0.01, 0.00])  # posizioni fisse

# Ripeti ogni t 4 volte (una per ciascuna x)
T = np.repeat(t_vals, 4)

# Ripeti l'intero vettore x per ogni t
X = np.tile(x_positions, len(t_vals))

# Preleva le temperature giuste da temps per ciascuna x
Y_vals = np.empty_like(T)
for i, x in enumerate(x_positions):
    Y_vals[i::4] = temps[:, i]  # ogni quarta riga, offset i

# Costruisci le altre colonne come nel modello: y1, gt1, gt2, y2 sempre uguali
Y1_rep = np.tile(temps[:, 0], 4)
GT1_rep = np.tile(temps[:, 1], 4)
GT2_rep = np.tile(temps[:, 2], 4)
Y2_rep = np.tile(temps[:, 3], 4)

grid_full = np.column_stack([T, X, Y1_rep, GT1_rep, GT2_rep, Y2_rep])
# Converti in tensore per il modello
grid_tensor = torch.tensor(grid_full, dtype=torch.float32)
# Predizioni del modello
model.eval()
with torch.no_grad():
    # Calcola tutte le predizioni: (2880, 4)
    preds_all = model(grid_tensor).detach().numpy()

    # Estrai la temperatura predetta corrispondente alla posizione x
    preds = []
    for row, x in zip(preds_all, grid_full[:, 1]):
        if np.isclose(x, 0.07):
            preds.append(row[0])
        elif np.isclose(x, 0.04):
            preds.append(row[1])
        elif np.isclose(x, 0.01):
            preds.append(row[2])
        elif np.isclose(x, 0.00):
            preds.append(row[3])
    preds = np.array(preds)
# Estrazione dei valori esatti per confronto
exact = []
for r in grid_full:
    x_val = r[1]
    if np.isclose(x_val, 0.07):
        exact.append(r[2])
    elif np.isclose(x_val, 0.04):
        exact.append(r[3])
    elif np.isclose(x_val, 0.01):
        exact.append(r[4])
    elif np.isclose(x_val, 0.00):
        exact.append(r[5])
exact = np.array(exact)
# Calcola la differenza assoluta per la heatmap di errore
diff = np.abs(exact - preds)
# Ricostruisci matrici per soluzione esatta, predizioni e differenza
# Ricostruisci matrici per soluzione esatta, predizioni e differenza
exact_mat = exact.reshape(len(t_vals), len(x_positions))
preds_mat = preds.reshape(len(t_vals), len(x_positions))
diff_mat = np.abs(exact_mat - preds_mat)
# Riordina per x crescente (da sinistra a destra nei plot)
sort_idx = np.argsort(x_positions)
x_positions = np.array(x_positions)[sort_idx]
exact_mat = exact_mat[:, sort_idx]
preds_mat = preds_mat[:, sort_idx]
diff_mat = diff_mat[:, sort_idx]
# Crea grafici heatmap affiancati
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(exact_mat, origin='lower', aspect='auto',
                     extent=[min(x_positions), max(x_positions), min(t_vals), max(t_vals)])
axes[0].set_title("Temperatura esatta")
axes[0].set_xlabel("Spazio x")
axes[0].set_ylabel("Tempo t")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(preds_mat, origin='lower', aspect='auto',
                     extent=[min(x_positions), max(x_positions), min(t_vals), max(t_vals)])
axes[1].set_title("Temperatura predetta dalla GRU")
axes[1].set_xlabel("Spazio x")
axes[1].set_ylabel("Tempo t")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(diff_mat, origin='lower', aspect='auto',
                     extent=[min(x_positions), max(x_positions), min(t_vals), max(t_vals)])
axes[2].set_title("Differenza di temperatura")
axes[2].set_xlabel("Spazio x")
axes[2].set_ylabel("Tempo t")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
#plt.show(block=True)

plt.show(block=False)
# Visualizza il grafico a tre pannelli senza bloccare
plt.pause(0.1)
# Breve pausa per permettere il rendering dei pannelli

# === CONFRONTO PROFILI TEMPERATURA IN 4 POSIZIONI (y1, gt1, gt2, y2) ===
plt.figure(figsize=(14, 4))
label_pos = {
    0.07: "y1 (x=0.07)",
    0.04: "gt1 (x=0.04)",
    0.01: "gt2 (x=0.01)",
    0.00: "y2 (x=0.00)"
}

for i, xpos in enumerate([0.07, 0.04, 0.01, 0.00]):
    plt.subplot(1, 4, i+1)
    idx = np.where(np.isclose(grid_full[:, 1], xpos))[0]
    t_fixed = grid_full[idx, 0]
    exact_vals = exact[idx]
    pred_vals = preds[idx]
    plt.plot(t_fixed, exact_vals, label="Esatta", color="red")
    plt.plot(t_fixed, pred_vals, label="GRU", color="blue")
    
    plt.title(label_pos[xpos])
    plt.xlabel("Tempo t")
    if i == 0:
        plt.ylabel("Temperatura")
    plt.grid(True)
    plt.legend()
plt.suptitle("Confronto temporale tra Temperatura esatta e predetta dalla GRU")
plt.subplots_adjust(top=0.85)


plt.tight_layout()
#plt.show(block=True)

plt.show(block=False)
# Visualizza il grafico a tre pannelli senza bloccare
plt.pause(0.1)
# Breve pausa per permettere il rendering dei pannelli
input("Premi Invio per chiudere i grafici...")