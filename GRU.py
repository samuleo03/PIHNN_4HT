import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

#dde.config.set_random_seed(seed)
np.random.seed(5)
torch.manual_seed(7)

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

# Generazione dei dati di input e target
X, y = gen_testdata()
# Parametri
num_inputs = 2
num_hiddens = 64
num_epochs = 100
batch_size = 256
learning_rate = 0.001

# Caricamento dati e split
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

n = X_tensor.shape[0]
indices = torch.randperm(n)
split = int(n * 0.8)
train_indices = indices[:split]
test_indices = indices[split:]

X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

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
        return X_test.clone(),  y_test.clone()

def get_gru_indices():
    """
    Restituisce due array di indici (train_idx, test_idx)
    usati nel DataFusion per allineare i dati PINN.
    """
    return train_indices.numpy(), test_indices.numpy()

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
# Calcola le predizioni su tutta la griglia
model.eval()
with torch.no_grad():
    preds = model(grid_tensor).numpy().reshape(len(t), len(x))
# Soluzione esatta (transposta come in gen_testdata)
exact = data["usol"].T
diff = np.abs(exact - preds) #DA CALCOLARE LA NORMA
l2_gru=np.linalg.norm(diff)
print(l2_gru)

# Plot a 3 pannelli: esatta, predetta e differenza
# Crea una figura con 3 pannelli affiancati per confronto
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Pannello 1: mappa di calore della soluzione esatta
im0 = axes[0].imshow(exact, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', origin='lower')
axes[0].set_title("Temperatura esatta")
axes[0].set_xlabel("Spazio x")
axes[0].set_ylabel("Tempo t")
fig.colorbar(im0, ax=axes[0])
# Pannello 2: mappa di calore della soluzione predetta dal modello
im1 = axes[1].imshow(preds, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', origin='lower')
axes[1].set_title("Temperatura predetta dalla GRU")
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