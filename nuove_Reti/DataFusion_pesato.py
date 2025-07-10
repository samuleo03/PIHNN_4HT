# DataFusion.py: combina le predizioni di due modelli (GRU e PINN) in un unico output
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from GRU_l2_err import load_gru_data, get_gru_indices, GRURegressor
from PINN_l2_err import gen_testdata, PINNModel


# Definizione del blocco di fusione dati che combina le predizioni con un peso fisso
class DataFusionBlock(nn.Module):
    """
    Data Fusion Block semplice: combina le predizioni di GRU e PINN usando un peso fisso lamda.
    """
    def __init__(self, gru_model, pinn_model, fusion_weight: float = 0.5):
        super().__init__()
        self.gru = gru_model
        self.pinn = pinn_model
        self.lamda = fusion_weight

    def forward(self,
                X_gru:  torch.Tensor,
                X_pinn: torch.Tensor,
                X_df:   torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_gru: input per la GRU
            X_pinn: input per la PINN
            X_df: ignorato in questa versione
        Returns:
            Tensor (batch,1): predizione ibrida y_fusion = lamda*y_GRU + (1-lamda)*y_PINN
        """
        y_gru = self.gru(X_gru)
        y_pinn = self.pinn(X_pinn)
        return self.lamda * y_gru + (1 - self.lamda) * y_pinn


if __name__ == "__main__":
    # --- Main: caricamento dati e selezione del miglior peso di fusione ---
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

    # Instanzio i modelli GRU e PINN
    # --- 6) Instanzia i modelli ---
    input_size  = X_gru_train.shape[1]
    hidden_size = 64  # come in GRU.py
    gru  = GRURegressor(input_size, hidden_size)
    gru.load_state_dict(torch.load("gru_trained.pth"))
    gru.eval()
    pinn = PINNModel()  # carica internamente il checkpoint se presente

    criterion = nn.MSELoss()

    # Costruzione della griglia (x,t) per heatmap
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
    
    # 3) Input per il DataFusionBlock (concatenazione delle feature)
    X_df_heatmap = torch.cat([X_gru_heatmap, X_pinn_heatmap], dim=1)

    # Carica i dati sperimentali da file .npz
    data_exp = np.load("heat_eq_damp_A035.npz")
    t_exp = data_exp["t"].flatten()
    x_exp = data_exp["x"].flatten()
    exact = data_exp["usol"].T  # Dati sperimentali in forma (nt, nx)

    lamda_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    #lamda_list = [0.25, 0.5, 0.75]

    # Calcolo del massimo errore assoluto globale tra tutti i lamda
    max_diff_global = 0.0
    for lamda in lamda_list:
        df = DataFusionBlock(gru, pinn, fusion_weight=lamda)
        df.eval()
        with torch.no_grad():
            U_fused = df(X_gru_heatmap, X_pinn_heatmap, X_df_heatmap)
            U_fused = U_fused.cpu().numpy().reshape(nt, nx)
        diff = np.abs(exact - U_fused)
        max_diff_global = max(max_diff_global, np.max(diff))
    print(f"Global maximum absolute error: {max_diff_global:.6f}")

    for lamda in lamda_list:
        print(f"\n====== Valutazione con lamda = {lamda:.2f} ======")
        df = DataFusionBlock(gru, pinn, fusion_weight=lamda)
        df.eval()

        # Imposta range massimo per la differenza assoluta (errore)
        #vmax_diff = np.max(diff)  # scala dinamica per evidenziare errori locali

        # # Calcolo MSE su test set
        # total_loss = 0.0
        # with torch.no_grad():
        #     for Xg, Xp, Xd, y in test_loader:
        #         y_hat = df(Xg, Xp, Xd)
        #         loss = criterion(y_hat, y)
        #         total_loss += loss.item() * Xg.size(0)
        # mse = total_loss / len(test_ds)
        # print(f"MSE: {mse:.6f}")

        # Costruzione della griglia (x,t) e predizione
        with torch.no_grad():
            U_fused = df(X_gru_heatmap, X_pinn_heatmap, X_df_heatmap)
            U_fused = U_fused.cpu().numpy().reshape(nt, nx)
        # Controllo le dimensioni tra exact e U_fused
        if exact.shape != U_fused.shape:
            print(f"Shape mismatch: exact={exact.shape}, U_fused={U_fused.shape}")
            raise ValueError("Dimensioni non compatibili tra exact e U_fused")

        # Calcola la differenza senza trasporre U_fused
        diff = np.abs(exact - U_fused)
        # Calcola e stampa la norma L2 globale
        l2_error = np.linalg.norm(diff)
        print(f"L2 error (norma euclidea) per lamda={lamda:.2f}: {l2_error:.6f}")

        # Plot delle heatmap per ciascun lamda
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(exact, origin='lower', extent=[x_exp.min(), x_exp.max(), t_exp.min(), t_exp.max()], aspect='auto', cmap='viridis')
        axes[0].set_title('Dati Sperimentali')
        fig.colorbar(im0, ax=axes[0], label='Temperatura')

        im1 = axes[1].imshow(U_fused, origin='lower', extent=[x_exp.min(), x_exp.max(), t_exp.min(), t_exp.max()], aspect='auto', cmap='viridis')
        axes[1].set_title(f'Fusione GRU+PINN (lamda={lamda:.2f})')
        fig.colorbar(im1, ax=axes[1], label='Temperatura')

        im2 = axes[2].imshow(diff, origin='lower',
                             extent=[x_exp.min(), x_exp.max(), t_exp.min(), t_exp.max()],
                             aspect='auto', cmap='viridis', vmin=0, vmax=max_diff_global)
        axes[2].set_title('Differenza Assoluta')
        fig.colorbar(im2, ax=axes[2], label='Errore Assoluto')

        plt.tight_layout()
        input(f"Premi Invio per continuare dopo lamda = {lamda:.2f}...")
        plt.show()