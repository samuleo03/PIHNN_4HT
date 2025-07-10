# DataFusion.py: combina le predizioni di due modelli (GRU e PINN) in un unico output
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from GRU import load_gru_data, get_gru_indices, GRURegressor
from PINN import gen_testdata, PINNModel


# Definizione del blocco di fusione dati che impara a pesare le due predizioni
class DataFusionBlock(nn.Module):
    """
    Data Fusion Block per PIHNN:
    Combina le predizioni di una GRU (black-box) e di una PINN (first-principle)
    in un’unica uscita y_fusion = MLP([X_df, y_gru, y_pinn]) secondo le formule (15)–(16).
    """
    def __init__(self, gru_model, pinn_model, df_input_dim: int, hidden_dim: int = 16):
        """
        Args:
            gru_model:  istanza di GRURegressor addestrata
            pinn_model: istanza di PINNModel addestrata
            df_input_dim: dimensione delle feature X_df
            hidden_dim: numero di neuroni nello strato nascosto del MLP
        """
        super().__init__()
        self.gru  = gru_model
        self.pinn = pinn_model
        # Costruzione di un gate MLP che calcola un peso sulla base di feature di contesto e predizioni
        self.mlp = nn.Sequential(
            nn.Linear(df_input_dim + 2, hidden_dim),
            nn.Tanh(), # formula (15)
            nn.Linear(hidden_dim, 1), # formula (16)
        )

    def forward(self,
                X_gru:  torch.Tensor,
                X_pinn: torch.Tensor,
                X_df:   torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_gru:  Tensor (batch, n_inputs_gru) per la GRU
            X_pinn: Tensor (batch, n_inputs_pinn) per la PINN
            X_df:   Tensor (batch, df_input_dim) feature di contesto
        Returns:
            Tensor (batch,1): predizione ibrida y_fusion
        """
        # 1) Ottieni le predizioni dai modelli GRU e PINN
        y_gru  = self.gru(X_gru)      # (batch,1)
        y_pinn = self.pinn(X_pinn)    # (batch,1)

        #y_gru  = self.gru(X_df)      
        #y_pinn = self.pinn(X_df)

        # 2) Combina feature di contesto con le due predizioni per il gate
        z = torch.cat([X_df, y_gru, y_pinn], dim=1)  # (batch, df_input_dim+2)

        # 3) Restituisce l'output del gate calcolato secondo le formule (15)-(16)
        return self.mlp(z) 


if __name__ == "__main__":
    # --- Main: caricamento dati e training del blocco di fusione ---
    plt.close('all')
    # Carico le feature e target per il modello GRU
    # --- 2) Carica i dati GRU e indici di split PINN ---
    X_gru_train, y_train = load_gru_data(split="train")
    X_gru_test,  y_test  = load_gru_data(split="test")
    train_idx, test_idx   = get_gru_indices()

   # Carico i dati PINN e allineo con gli indici di training e test
    # --- 3) Carica i dati PINN e allineali con gli indici ---
    X_pinn_full, y_full = gen_testdata()# numpy array (N,2) and (N,1) experimental data
    # Infer the grid shape from the PINN input coordinates
    coords = X_pinn_full
    xs = np. unique(coords[:,0])
    ts = np. unique(coords[:, 1])
    nx, nt = len(xs), len(ts)
    # Reshape the reference solution to match the grid
    y_full = torch. from_numpy (y_full). float() .reshape(nt, nx)
    X_pinn_full = torch. from_numpy (X_pinn_full).float ()
    X_pinn_train   = X_pinn_full[train_idx]
    X_pinn_test    = X_pinn_full[test_idx]

    # Costruisco le feature di contesto concatenando le feature GRU e PINN
    # --- 4) Costruisci X_df concatenando GRU + PINN features ---
    X_df_train = torch.cat([X_gru_train, X_pinn_train], dim=1)
    X_df_test  = torch.cat([X_gru_test,  X_pinn_test ], dim=1)

    # Preparo i DataLoader per il training e test del blocco di fusione dati
    # --- 5) DataLoader per training/test del blocco di fusione ---
    train_ds     = TensorDataset(X_gru_train, X_pinn_train, X_df_train, y_train)
    test_ds      = TensorDataset(X_gru_test,  X_pinn_test,  X_df_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    # Instanzio i modelli GRU, PINN e il blocco di fusione dati
    # --- 6) Instanzia i modelli e il DataFusionBlock ---
    input_size  = X_gru_train.shape[1]
    hidden_size = 64  # come in GRU.py
    gru  = GRURegressor(input_size, hidden_size)
    pinn = PINNModel()  # carica internamente il checkpoint se presente
    df   = DataFusionBlock(
        gru_model=gru,
        pinn_model=pinn,
        df_input_dim=X_df_train.shape[1],
        hidden_dim=16
    )

    # Definisco la loss e l'ottimizzatore per addestrare solo il MLP di fusione
    # --- 7) Definisci loss e ottimizzatore (solo per il MLP di fusione) ---
    criterion = nn.MSELoss() #minimizziamo l'MSE tra l'output fuso e tagert che è quello corretto
    optimizer = torch.optim.Adam(df.parameters(), lr=1e-3)

    #così facendo inseriamo solo i tensori du peso  e bias dell'mlp e non quelli della gru e della PINN. così aggiorniamo solo i pesi dell'mlp

    # Addestramento del blocco di fusione dati e raccolta delle loss
    # --- 8) Training del DataFusionBlock e raccolta delle loss ---
    train_losses, test_losses = [], []
    epochs = 20

    for ep in range(1, epochs + 1):
        # Training
        df.train()
        tot_tr = 0.0
        for Xg, Xp, Xd, y in train_loader:
            optimizer.zero_grad() 
            y_hat = df(Xg, Xp, Xd) 
            loss  = criterion(y_hat, y) #formula 13
            loss.backward()
            optimizer.step()
            tot_tr += loss.item() * Xg.size(0)
        train_losses.append(tot_tr / len(train_ds))

        # Validation
        df.eval()
        tot_te = 0.0
        with torch.no_grad():
            for Xg, Xp, Xd, y in test_loader:
                y_hat = df(Xg, Xp, Xd)
                tot_te += criterion(y_hat, y).item() * Xg.size(0)
        test_losses.append(tot_te / len(test_ds))

        print(f"Epoca {ep}/{epochs}, "
              f"DF Train Loss: {train_losses[-1]:.6f}, "
              f"DF Test Loss: {test_losses[-1]:.6f}")

    # Visualizzo l'andamento delle loss durante il training
    # --- 9) Plot delle loss di fusione ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), train_losses, label="DF Training Loss", marker='o')
    plt.plot(range(1, epochs+1), test_losses,  label="DF Test Loss",     marker='s')
    plt.yscale('log')
    plt.xlabel("Epoca")
    plt.ylabel("MSE Loss")
    plt.title("Andamento della Loss del DataFusionBlock")
    plt.legend()
    plt.grid(True, which='both', ls='--', linewidth=0.5)
    plt.tight_layout()
    plt.show(block=True)

    # Costruzione della griglia (x,t) e visualizzazione dell'heatmap della soluzione fusa
    # --- 10) Heatmap della soluzione fusa ---
    # Genera i dati di test per calcolo U_fused
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, 1, nt)
    X, T = torch.meshgrid(x, t, indexing='xy')
    
    # Crea i tensori per i tre input necessari al DataFusionBlock
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    
    # 1) Input per la PINN (coppie x,t)
    X_pinn_heatmap = torch.cat([X_flat, T_flat], dim=1)
    
    # 2) Input per la GRU (dobbiamo creare feature compatibili)
    # Assumendo che X_gru abbia la stessa dimensionalità dell'esempio
    # Possiamo replicare le feature medie o usare valori rappresentativi
    # Usa gli stessi (x,t) della PINN come input GRU
    X_gru_heatmap = X_pinn_heatmap.clone()
    # Oppure possiamo usare valori specifici se conosciamo il significato delle feature
    # X_gru_heatmap[:,0] = X_flat.squeeze()  # se la prima feature corrisponde a x
    # X_gru_heatmap[:,1] = T_flat.squeeze() # se la seconda feature corrisponde a t
    
    # 3) Input per il DataFusionBlock (concatenazione delle feature)
    X_df_heatmap = torch.cat([X_gru_heatmap, X_pinn_heatmap], dim=1)
    
    # Calcola le predizioni fuse
    df.eval()
    with torch.no_grad():
        U_fused = df(X_gru_heatmap, X_pinn_heatmap, X_df_heatmap)
        U_fused = U_fused.numpy().reshape(nt, nx)
    
    # Carica i dati sperimentali da file .npz
    data_exp = np.load("heat_eq_data.npz")
    t_exp = data_exp["t"].flatten()
    x_exp = data_exp["x"].flatten()
    exact = data_exp["usol"].T
    diff = np.abs(exact - U_fused)

    # Plot delle heatmap: fusione vs dati sperimentali
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Heatmap 1: soluzione fusa
    im0 = axes[0].imshow(
        exact,
        origin='lower',
        extent=[x_exp.min(), x_exp.max(), t_exp.min(), t_exp.max()],
        aspect='auto',
        cmap='viridis'
    )
    axes[0].set_xlabel('Spazio x')
    axes[0].set_ylabel('Tempo t')
    axes[0].set_title('Dati Sperimentali')
    fig.colorbar(im0, ax=axes[0], label='Temperatura')

    # Heatmap 2: dati sperimentali
    im1 = axes[1].imshow(
        U_fused,
        origin='lower',
        extent=[x_exp.min(), x_exp.max(), t_exp.min(), t_exp.max()],
        aspect='auto',
        cmap='viridis'
    )
    axes[1].set_xlabel('Spazio x')
    axes[1].set_ylabel('Tempo t')
    axes[1].set_title('Soluzione Fusa (DataFusionBlock)')
    fig.colorbar(im1, ax=axes[1], label='Temperatura')

    # Heatmap 3: differenza tra soluzione fusa e dati sperimentali
    
    im2 = axes[2].imshow(
        diff,
        origin='lower',
        extent=[x_exp.min(), x_exp.max(), t_exp.min(), t_exp.max()],
        aspect='auto',
        cmap='viridis'
    )
    axes[2].set_xlabel('Spazio x')
    axes[2].set_ylabel('Tempo t')
    axes[2].set_title('Differenza (Sperimentale-Fusa)')
    fig.colorbar(im2, ax=axes[2], label='Errore')

    plt.tight_layout()
    plt.show(block=True)