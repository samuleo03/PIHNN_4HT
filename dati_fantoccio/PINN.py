# PINN per l’equazione del calore unidimensionale usando DeepXDE e PyTorch
# Import di DeepXDE per la definizione di PINN
import torch
import deepxde as dde
# Import di NumPy per operazioni numeriche
import numpy as np
import matplotlib.pyplot as plt  # per il plotting delle loss
# Abilita la modalità interattiva di Matplotlib per mostrare i plot
plt.ion()

# Parametri scalati PBHE (Tabella VI)
a1 = 18.992
a2 = 34185.667
a3 = 0.000
a4 = 3.570
a5 = 1.167

# Coefficienti IC (formula 22) b1..b4
b1 = 3.304480
b2 = -5.114949
b3 = 1.310054
b4 = 0.564588

wb=2.0e-04

# Funzione v̄(τ) per la condizione di Robin
vbar = lambda tau: 0 * np.ones_like(tau)  # tau shape (n,1)

def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

class PINNModel:
    """
    Wrapper per utilizzare il modello PINN (DeepXDE) come callable.
    Usa il modello DeepXDE già istanziato e restaurato sopra.
    """
    def __init__(self):
        # 'model' è l'istanza dde.Model già creata e restaurata prima
        self.model = model

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

# Definizione del residuo PDE: versione semplificata (senza termine Pbar)
def pde(x, y): #formula 20
    # x[:,0] = X, x[:,1] = τ
    dtheta_tau = dde.grad.jacobian(y, x, i=0, j=1)
    d2theta_xx = dde.grad.hessian(y, x, i=0, j=0)
    return a1 * dtheta_tau - d2theta_xx + a2 * y *wb #termine wb perfusione manca

# Definizione del dominio spaziale [0, 1]
geom = dde.geometry.Interval(0, 1)
# Definizione del dominio temporale [0, 1]
timedomain = dde.geometry.TimeDomain(0, 1)
# Combinazione di spazio e tempo in un dominio 1+1D
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Condizione di Robin in X=0: ∂Xθ = a5 (v̄(τ) - θ(0,τ)) formula 21
# bc = dde.icbc.NeumannBC(geomtime, #no neumann
#                         lambda x: a5 * (vbar(x[:,1:2]) - 0), #teta non è 0
#                         lambda x, on_boundary: on_boundary and np.isclose(x[0], 0))

def bc0_fun(x, theta, _):
        
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        # replace with vbar
        vbar=0
        flusso = a5 * (vbar - theta) 

        return dtheta_x + flusso

bc_0 = dde.icbc.OperatorBC(geomtime, bc0_fun, boundary_0)

# Condizione di Dirichlet in X=1: θ(1,τ) = 0 #formula 21
bc2 = dde.icbc.DirichletBC(geomtime,
                           lambda x: 0,
                           lambda x, on_boundary: on_boundary and np.isclose(x[0], 1))

# Condizione iniziale θ0(X) = b1 X^3 + b2 X^2 + b3 X + b4 ##formula 22
ic = dde.icbc.IC(
    geomtime,
    lambda x: b1 * x[:,0:1] ** 3 + b2 * x[:,0:1] ** 2 + b3 * x[:,0:1] + b4,
    lambda x, on_initial: on_initial,
)

def ic_fun(x):
    # restituisce θ0(X) per x[:,0]
    return b1 * x[:,0:1]**3 + b2 * x[:,0:1]**2 + b3 * x[:,0:1] + b4

def output_transform(x, y): #formula hard constraints
    X = x[:, 0:1]
    τ = x[:, 1:2]
    return τ * (1 - X) * y + ic_fun(x)

# Creazione dei punti di collocazione per dominio, bordo, iniziale e test
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_0],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)
# Definizione dell’architettura FNN: 2 input, 3 hidden da 20 neuroni, 1 output
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)
# Costruzione del modello PINN basato su data e rete
model = dde.Model(data, net)

# TRAINING PINN: stampo iterazioni di loss su train e test
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000, display_every=1000)
model.compile("L-BFGS") #per sistemare e smussare gli errori residui

losshistory, train_state = model.train(model_save_path="./tesi/models/pinn", display_every=1000)

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

# === HEATMAP: Temperatura finale θ(X, τ=1) ===

# Crea i punti in x a tempo τ=1
n_x = 100
x_vals = np.linspace(0, 1, n_x).reshape(-1, 1)
t_vals = np.ones_like(x_vals)  # τ = 1
XT_final = np.hstack((x_vals, t_vals))

# Predizione finale
theta_final = model.predict(XT_final).flatten()

# Mostra come heatmap 2D (1 riga, n_x colonne)
fig, ax = plt.subplots(figsize=(8, 2))
im = ax.imshow(
    theta_final.reshape(1, -1),
    extent=[x_vals.min(), x_vals.max(), 0, 1],
    aspect="auto",
    cmap="viridis"
)
ax.set_title("Temperatura finale θ(X, τ=1)")
ax.set_xlabel("Spazio x")
ax.set_ylabel("Tempo τ")
ax.set_yticks([0.5])  # mostra un tick centrale
ax.set_yticklabels(["τ=1"])  # etichetta esplicita
fig.colorbar(im, ax=ax, label="Temperatura")

plt.tight_layout()
#plt.show(block=True)

plt.show(block=False)
# Visualizza il grafico a tre pannelli senza bloccare
plt.pause(0.1)
# Breve pausa per permettere il rendering dei pannelli
input("Premi Invio per chiudere i grafici...")