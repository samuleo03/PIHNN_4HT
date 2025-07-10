# PINN per l’equazione del calore unidimensionale usando DeepXDE e PyTorch
# Import di DeepXDE per la definizione di PINN
import torch
import deepxde as dde
# Import di NumPy per operazioni numeriche
import numpy as np
import matplotlib.pyplot as plt  # per il plotting delle loss
# Abilita la modalità interattiva di Matplotlib per mostrare i plot
plt.ion()

dde.config.set_random_seed(4)
np.random.seed(5)
torch.manual_seed(7)

def space_dampening(x, A=0.5):
    return np.exp(-x*A)

class PINNModel:
    """
    Wrapper per utilizzare il modello PINN (DeepXDE) come callable.
    Usa il modello DeepXDE già istanziato e restaurato sopra.
    """
    def __init__(self):
        # 'model' è l'istanza dde.Model già creata e restaurata prima
        self.model = model
        import glob
        ckpt_list = sorted(glob.glob("pinn_trained-*.pt"))
        if ckpt_list:
            self.model.restore(ckpt_list[-1])
            print(f"✅ Modello PINN caricato da: {ckpt_list[-1]}")
        else:
            print("⚠️  Nessun checkpoint PINN trovato (pinn_trained-*.pt). Hai dimenticato di addestrarlo?")

    def __call__(self, X_pinn):
        """
        Args:
            X_pinn (torch.Tensor): shape (batch, 2), coppie (x, t)
        Returns:
            torch.Tensor: shape (batch,1), predizioni PINN come tensore
        """
        # Se arriva un tensore, convertilo in numpy
        X_np = X_pinn.detach().cpu().numpy() if isinstance(X_pinn, torch.Tensor) else X_pinn
        # Chiama DeepXDE
        y_np = self.model.predict(X_np)
        # Ritorna torch.Tensor
        return torch.from_numpy(y_np).float()

# Funzione analitica della soluzione esatta u(x,t) per condizioni iniziali sinusoidali
def heat_eq_exact_solution(x, t):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    # Calcolo di u = exp(-n^2 pi^2 a t / L^2) * sin(n pi x / L)
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)


# Generazione della soluzione esatta su griglia regolare e salvataggio su file .npz
def gen_exact_solution():
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Dimensione della griglia: 256 punti in x, 201 in t
    x_dim, t_dim = (256, 201)

    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0.0)
    x_max, t_max = (L, 1.0)

    # Definizione dei valori di x e t uniformemente spazializzati
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)
    usolA05 = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)
    usolA035 = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # Calcolo della soluzione esatta in ogni punto della griglia
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = heat_eq_exact_solution(x[i], t[j])
            usolA05[i][j] = heat_eq_exact_solution(x[i], t[j])*space_dampening(x[i])
            usolA035[i][j] = heat_eq_exact_solution(x[i], t[j])*space_dampening(x[i], A=0.35)

    # Salvataggio dei dati (x, t, usol) in heat_eq_data.npz
    np.savez("heat_eq_data", x=x, t=t, usol=usol)
    np.savez("heat_eq_damp_A05", x=x, t=t, usol=usolA05)
    np.savez("heat_eq_damp_A035", x=x, t=t, usol=usolA035)


# Caricamento e preprocessing del dataset per training/test
def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Carica i dati salvati in heat_eq_data.npz
    data = np.load("heat_eq_data.npz")
    # Obtain the values for t, x, and the excat solution:
    t, x, exact = data["t"], data["x"], data["usol"].T
    # Creazione della griglia 2D da vettori x e t
    xx, tt = np.meshgrid(x, t)
    # Costruzione della matrice X di input con coppie (x,t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    # Costruzione del vettore y con soluzioni corrispondenti
    y = exact.flatten()[:, None]
    return X, y


# Definizione del residuo PDE: du/dt - a d2u/dx2 = 0
def pde(x, y):
    """Expresses the PDE residual of the heat equation."""
    # Derivata parziale rispetto al tempo
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    # Derivata seconda rispetto allo spazio
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    # Residuo dell’equazione del calore
    return dy_t - a * dy_xx


# Problem parameters:
a = 0.4  # Thermal diffusivity
L = 1  # Length of the bar
n = 1  # Frequency of the sinusoidal initial conditions

# Generate a dataset with the exact solution (if you dont have one):
gen_exact_solution()

# Definizione del dominio spaziale [0, L]
geom = dde.geometry.Interval(0, L)
# Definizione del dominio temporale [0, 1]
timedomain = dde.geometry.TimeDomain(0, 1)
# Combinazione di spazio e tempo in un dominio 1+1D
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Condizione al contorno Dirichlet (u=0 su bordo spaziale)
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
# Condizione iniziale u(x,0) = sin(n pi x / L)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)

# Creazione dei punti di collocazione per dominio, bordo, iniziale e test
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)
# Definizione dell’architettura FNN: 2 input, 3 hidden da 20 neuroni, 1 output
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
# Costruzione del modello PINN basato su data e rete
model = dde.Model(data, net)

# TRAINING PINN: stampo iterazioni di loss su train e test
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000, display_every=1000)
model.compile("L-BFGS") #per sistemare e smussare gli errori residui

losshistory, train_state = model.train(model_save_path="./tesi/models/pinn", display_every=1000)

model.save("pinn_trained")

# === PLOT PULITO: SOLO PDE LOSS DELLA PINN ===
lt_train = np.array(losshistory.loss_train)[:, 0]  # solo il primo termine (PDE)
lt_test  = np.array(losshistory.loss_test)[:, 0]   # solo PDE test
# Ascisse (ogni display_every iterazioni)
num_iters = lt_train.shape[0]
iterations = [i * 1000 for i in range(1, num_iters + 1)]

plt.figure(figsize=(10, 5))
plt.plot(iterations,
             lt_train,
             label="PDE's training loss",
             marker='o',
             markersize=4)
plt.plot(iterations,
             lt_test,
             label="PDE's test loss",
             marker='x',
             markersize=4)
plt.xlabel('Iterazioni')
plt.ylabel('PDE residual loss')
plt.title('Confronto PDE residual training vs test')
plt.yscale('log')
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
 # Mostra il grafico e attendi che l'utente lo chiuda manualmente

plt.show(block=False)

# === PLOT 3 HEATMAP PINN SULLA GRIGLIA ===
import torch

# Ricarica i dati per ricostruire la griglia spazio-temporale
data = np.load("heat_eq_data.npz")
t = data["t"].flatten()
x = data["x"].flatten()
xx, tt = np.meshgrid(x, t)
grid = np.vstack((xx.ravel(), tt.ravel())).T

# Predizione PINN su tutta la griglia
y_pinn = model.predict(grid).reshape(len(t), len(x))

# Soluzione esatta
exact = data["usol"].T
diff = np.abs(exact - y_pinn) #voglio la norma di questo vettore-> per quantificare accuratezza della piNN
l2_pinn=np.linalg.norm(diff)
print(l2_pinn) #ci devi fare una tabella degli errori rispoetto a pINN o GRU

# Crea una figura con 3 heatmap affiancate
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Heatmap 1: Soluzione esatta
im0 = axes[0].imshow(
    exact,
    extent=[x.min(), x.max(), t.min(), t.max()],
    origin='lower',
    aspect='auto'
)
axes[0].set_title("Soluzione Esatta")
axes[0].set_xlabel("Spazio x")
axes[0].set_ylabel("Tempo t")
fig.colorbar(im0, ax=axes[0], label="Temperatura")

# Heatmap 2: Predizione PINN
im1 = axes[1].imshow(
    y_pinn,
    extent=[x.min(), x.max(), t.min(), t.max()],
    origin='lower',
    aspect='auto'
)
axes[1].set_title("Soluzione PINN")
axes[1].set_xlabel("Spazio x")
axes[1].set_ylabel("Tempo t")
fig.colorbar(im1, ax=axes[1], label="Temperatura")

# Heatmap 3: Differenza assoluta
im2 = axes[2].imshow(
    diff,
    extent=[x.min(), x.max(), t.min(), t.max()],
    origin='lower',
    aspect='auto'
)
axes[2].set_title("Differenza Assoluta")
axes[2].set_xlabel("Spazio x")
axes[2].set_ylabel("Tempo t")
fig.colorbar(im2, ax=axes[2], label="Errore")

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
input("Premi Invio per chiudere i grafici...")

# # Configurazione per ottimizzatore L-BFGS
# model.compile("L-BFGS")
# # Ripristino dei pesi da checkpoint pre-addestrato
# model.restore("./tesi/models/pinn-20399.pt")

# # Generazione dei dati di test per valutazione
# X, y_true = gen_testdata()
# # Predizione della rete sui punti di test
# y_pred = model.predict(X)
# # Calcolo del residual PDE medio sui punti di test
# f = model.predict(X, operator=pde)
# # Stampa del valore medio del residuo PDE
# print("Mean residual:", np.mean(np.absolute(f)))
# # Stampa dell’errore L2 relativo tra predizioni e esatto

# print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
# # Salva il grafico in un file PNG
# plt.savefig("pinn_loss_plot.png")
# # Salvataggio dei risultati di test in file test.dat
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))