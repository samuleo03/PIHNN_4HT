import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def prepare_datasets(test_ratio=0.2, seed=42):
    """Load data and split into training and testing sets."""
    data = np.load("heat_eq_data.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int((1 - test_ratio) * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test

# GRU-based Surrogate Model
class PhysicsInformedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PhysicsInformedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x): #metodo forward
        h, _ = self.gru(x) #attraverso il layer della GRU: evluzioni temporali
        return self.fc(h) #output della GRU passa per un full connected layer: così riusciamo a legare output della cella con predizione di temperatura

# Fully Connected Neural Network (FCNN) for T(x, t)
class TemperatureNN(nn.Module): #serve per approssimare la temperatura nel tempo e nello spazio
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(TemperatureNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, output_dim))
        #ha due hidden layers che corrispondono alle funzioni di attivazione tanh
    
    def forward(self, x, t):
        xt = torch.cat((x, t), dim=-1)
        output = self.net(xt)

        # Dirichlet boundary conditions: u(0,t) = u(1,t) = 0
        zero_mask = (x == 0) | (x == 1)
        output[zero_mask] = 0.0

        # Initial condition: u(x,0) = sin(nπx) with n=1
        n = 1
        init_mask = (t == 0)
        output[init_mask] = torch.sin(n * torch.pi * x[init_mask])

        return output

# Autodiff-based Physics Loss
def physics_loss(T_nn, x, t, k, omega, rho, c_p, rho_b, c_b, T_b, Q_m, Q_ext):
    """
    Compute the residual of the Pennes Bioheat equation using autograd.
    """
    T_pred = T_nn(x, t)
    T_pred.requires_grad_(True)  # Ensure autograd tracks computation

    # Compute dT/dt using autodiff
    dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0] #estrapola dalla tupla (la derivata) il primo elemento che è il tensore 

    # Compute d²T/dx² using autodiff
    dT_dx = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]

    # Compute bioheat equation residual
    perfusion_term = omega * rho_b * c_b * (T_b - T_pred)
    bioheat_residual = rho * c_p * dT_dt - k * d2T_dx2 - perfusion_term - Q_m - Q_ext

    return torch.mean(bioheat_residual**2)  # Penalize equation violation

# Training Loop
def train_model(gru_model, train_loader, test_loader, epochs=100, lr=0.001):
    optimizer_gru = optim.Adam(gru_model.parameters(), lr=lr)
    optimizer_nn = optim.Adam(nn_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []
    loss_history_test = []
    
    for epoch in range(epochs):
        gru_model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer_gru.zero_grad()
            optimizer_nn.zero_grad()

            x_batch = x_batch.unsqueeze(1)
            output = gru_model(x_batch)
            data_loss = criterion(output, y_batch)
            loss = data_loss
            
            loss.backward()
            optimizer_gru.step()
            optimizer_nn.step()

            train_loss += data_loss.item()

        train_loss /= len(train_loader)
        loss_history.append(train_loss)

        # Evaluate test loss
        gru_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.unsqueeze(1)
                output = gru_model(x_batch)
                data_loss = criterion(output, y_batch)
                test_loss += data_loss.item()

        test_loss /= len(test_loader)
        loss_history_test.append(test_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Plot the training and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss', marker='o', linestyle='-', color='C0')
    plt.plot(loss_history_test, label='Test Loss', marker='s', linestyle='--', color='C1')
    plt.xlabel('Epoche', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Evoluzione della Loss nel Training e nel Testing della RNN', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example Usage
input_size = 2  # Input features are (x, t)
hidden_size = 32
output_size = 1
num_layers = 1

gru_model = PhysicsInformedGRU(input_size, hidden_size, output_size, num_layers)
nn_model = TemperatureNN()

X_train, y_train, X_test, y_test = prepare_datasets()
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_model(gru_model, train_loader, test_loader)