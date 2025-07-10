import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

np.set_printoptions(precision=4, suppress=True, threshold=10)

# Costruzione del dominio
x=np.linspace(0,1, num=100) #parte da 0 arriva ad 1, 100 step
t=np.linspace(0,1, num=100)

xx,tt=np.meshgrid(x,t) #grid di x e t 2D spazio tempo

a=0.4 # Coeff di diffusione
n=1 # Primo termine della serie di Fourier

######################
# EQUAZIONE DI CALORE
######################

def solution(x, t):
    return np.exp(-(n**2 * np.pi**2 * a * t)) * np.sin(n * np.pi * x) #l=1

def gen_exact_solution():
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:
    x_dim, t_dim = (100, 100)

    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0.0)
    x_max, t_max = (1.0, 1.0)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # Obtain the value of the exact solution for each generated point:
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = solution(x[i], t[j]) 

    print("📌 Esempio di valori di u(x,t):")
    print(usol[:5, :5])

    #Condizioni al contorno: Se il problema numerico che stai risolvendo impone che u(0,t) = 0, allora è normale che tutti i valori di u in x=0 siano zero.

    # Save solution sul file compresso .npz:
    np.savez("heat_eq_data", x=x, t=t, usol=usol) #salva un archivio file compresso dove ci sono tre colonne


def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Load the data:
    data = np.load("heat_eq_data.npz")
    # Obtain the values for t, x, and the excat solution:
    t, x, exact = data["t"], data["x"], data["usol"].T
    # crea la mesh grid per i dati di input
    xx, tt = np.meshgrid(x, t)

    #Flatten: ogni riga sarà una coppia(x, t), e 'y' il valore della soluzione
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T #shape: (N, 2)
    y = exact.flatten()[:, None] # shape: (N, 1)
    return X, y

gen_exact_solution()
coord,T=gen_testdata()
#print(coord,T)
print("📌 Esempio di coordinate (x, t):")
for idx in range(0, len(coord), max(1, len(coord)//10)):
    print(f"x={coord[idx][0]:.4f}, t={coord[idx][1]:.4f} => u={T[idx][0]:.4f}")


# Il dataloader divide i dati in piccoli batch e li mescola ad ogni epoca
class HeatEquationDataset(Dataset):

    def __init__(self): #carica e converte i dati in tensori PYTORCH
        self.X, self.y = gen_testdata() #carica i dati
        self.X = torch.tensor(self.X, dtype=torch.float32) #converto in tensori Pytorch che può essere passato alla rete neurale che lavora sola con i tensori
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self): #restituisce il numero totale id campioni
        return self.X.shape[0] #numero totale di campioni

    def __getitem__(self, idx): #restituiisce un singolo campione
        return self.X[idx], self.y[idx] #restituisce una coppia (input, target)

# Suddivisione del dataset in training e test set
dataset = HeatEquationDataset() # CREA DATASET
train_size = int(0.8 * len(dataset)) # TRAINING SET SEDICATO ALL'80%
test_size = len(dataset) - train_size # TEST SET 20%
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # I DATI VENGONO DIVISI IN MANIERA CASUALE TRA I DUE SET

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------------
# Definizione della RNN
# ----------------------
import torch.nn as nn

class HeatEquationRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1, num_layers=1):
    #def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=1):
        super(HeatEquationRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # reshape per la RNN: (batch_size, seq_len=1, input_size)
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # prende solo l'output finale
        return out

# ----------------------
# Inizializzazione modello, loss, ottimizzatore
# ----------------------
model = HeatEquationRNN()
criterion = nn.MSELoss() # Squared Error Loss #capitolo 3.5.2
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Usando ottimizzazione Adam della loss
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # TASSO DI APPRENDIMENTO CAPITOLO 3.4.3

# ----------------------
# Training loop con suddivisione in train e test
# ----------------------
loss_history = []
loss_history_test = [] # BISOGNA TENER TRACCIA DELLA LOSS GENERATA SUI DATI NEL TEST SET
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    #for inputs, targets in dataloader:
    for inputs, targets in train_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calcolo della loss sul test set
    total_loss_test = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss_test += loss.item()

    if epoch % 10 == 0:
        loss_history.append(total_loss)
        loss_history_test.append(total_loss_test)
        print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Test Loss: {total_loss_test:.4f}")

import matplotlib.pyplot as plt

#plt.style.use("seaborn-v0_8-darkgrid")  # Tema più moderno

# Creazione del grafico
plt.figure(figsize=(8, 5))
plt.plot(range(0, num_epochs, 10), loss_history, label='Training Loss', color='#1f77b4', linewidth=2, marker='o')
plt.plot(range(0, num_epochs, 10), loss_history_test, label='Test Loss', color='#ff7f0e', linewidth=2, marker='s')

# Etichette e titolo
plt.xlabel("Epoche", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.title("Evoluzione della Loss nel Training e nel Testing della RNN", fontsize=14, fontweight='bold')

# Legenda migliorata
plt.legend(loc='upper right', fontsize=10)

# Aggiunta di una griglia più leggera
plt.grid(True, linestyle="--", alpha=0.7)

# Ottimizzazione del layout
plt.tight_layout()


# Creazione della figura con spazi adeguati
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.05, 1, 0.05, 1])  # Spazio per le colorbar accanto alle prime due immagini


# Confronto con la soluzione esatta
model.eval()
X_test, y_test = dataset.X, dataset. y
with torch.no_grad () :
    y_pred = model(X_test).numpy ()

# Prima immagine: Soluzione esatta
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(y_test.numpy().reshape(100, 100), extent=[0,1,0,1], origin='lower', aspect='auto', vmin=0, vmax=1)
ax1.set_title("Temperatura esatta")
ax1.set_xlabel("Spazio x")
ax1.set_ylabel("Tempo t")

# Colorbar per la prima immagine
cbar_ax1 = fig.add_subplot(gs[1])
fig.colorbar(im1, cax=cbar_ax1, fraction=0.046, pad=0.04)

# Seconda immagine: Predizione RNN
ax2 = fig.add_subplot(gs[2])
im2 = ax2.imshow(y_pred.reshape(100, 100), extent=[0,1,0,1], origin='lower', aspect='auto', vmin=0, vmax=1)
ax2.set_title("Temperatura predetta dalla RNN")
ax2.set_xlabel("Spazio x")
ax2.set_ylabel("Tempo t")

# Colorbar per la seconda immagine
cbar_ax2 = fig.add_subplot(gs[3])
fig.colorbar(im2, cax=cbar_ax2, fraction=0.046, pad=0.04)

# Terza immagine: Differenze
ax3 = fig.add_subplot(gs[4])
im3 = ax3.imshow(np.abs(y_test.numpy() - y_pred).reshape(100, 100), extent=[0,1,0,1], origin='lower', aspect='auto')
ax3.set_title("Differenza di temperatura")
ax3.set_xlabel("Spazio x")
ax3.set_ylabel("Tempo t")

# Aggiunta della colorbar per la terza immagine
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout() # Ottimizza la disposizione
plt.show()

########################
#GLI IPERPARAMETRI SONO: #capitolo 3.6 
#hidden_size=32
#num_layers=1
#learning_rate=0.001
#batch_size=64
#num_epochs=100
#vanno modificati se ci sono problemi del tipo
#La loss non converge bene
#Vuoi migliorare l'accuratezza
#Il modello mostra overfitting (loss di training molto più bassa della test loss)
#Il problema fosse più complesso (es. condizioni al contorno non lineari)
########################
