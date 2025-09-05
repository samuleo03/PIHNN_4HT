# DataFusion.py: combina le predizioni di due modelli (GRU e PINN) in un unico output
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


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
    plt.close('all')

    # --- Modalità restore-only e patch DeepXDE ---
    os.environ["RESTORE_ONLY"] = "1"
    try:
        import deepxde as dde
        # Evita qualunque train accidentale
        def _no_train(self, *args, **kwargs):
            return None
        dde.Model.train = _no_train
        # Safe-restore: inizializza optimizer se mancante
        _orig_restore = dde.Model.restore
        def _safe_restore(self, path, *args, **kwargs):
            if getattr(self, "opt", None) is None:
                try:
                    self.compile("adam", lr=1e-3)
                except Exception:
                    pass
            return _orig_restore(self, path, *args, **kwargs)
        dde.Model.restore = _safe_restore
    except Exception:
        pass

    # Importa i modelli SOLO ora (dopo le patch)
    # from GRU_l2_err import GRURegressor
    # from PINN_l2_err import PINNModel

    # --- Carica dati sperimentali e ricava griglia ---
    data_exp = np.load("heat_eq_damp_A035.npz")
    t_exp = data_exp["t"].flatten()
    x_exp = data_exp["x"].flatten()
    exact = data_exp["usol"].T  # (nt, nx)
    nt, nx = exact.shape

    # === Carica predizioni PINN/GRU dai .npz in nuove_Reti/risultati ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    risultati_dir = os.path.join(base_dir, "risultati")
    pinn_npz_path = os.path.join(risultati_dir, "pinns_pred_heat.npz")
    gru_npz_path  = os.path.join(risultati_dir, "gru_preds_heat.npz")
    if not os.path.exists(pinn_npz_path):
        raise FileNotFoundError(f"File PINN non trovato: {pinn_npz_path}")
    if not os.path.exists(gru_npz_path):
        raise FileNotFoundError(f"File GRU non trovato: {gru_npz_path}")

    pinn_data = np.load(pinn_npz_path)
    gru_data  = np.load(gru_npz_path)
    Y_pinn = pinn_data["y_pinn"].astype(np.float32)
    Y_gru  = gru_data["preds"].astype(np.float32)

    # Allineamento a shape di 'exact' (nt, nx), con trasposizione automatica se necessario
    def _maybe_T(A, target_shape):
        return A if A.shape == target_shape else (A.T if A.T.shape == target_shape else None)

    tmp = _maybe_T(Y_pinn, exact.shape)
    if tmp is None:
        raise ValueError(f"Shape mismatch Y_pinn vs exact: {Y_pinn.shape} vs {exact.shape}")
    if tmp is not Y_pinn:
        print(f"Trasposto Y_pinn: {Y_pinn.shape} -> {tmp.shape}")
    Y_pinn = tmp

    tmp = _maybe_T(Y_gru, exact.shape)
    if tmp is None:
        raise ValueError(f"Shape mismatch Y_gru vs exact: {Y_gru.shape} vs {exact.shape}")
    if tmp is not Y_gru:
        print(f"Trasposto Y_gru: {Y_gru.shape} -> {tmp.shape}")
    Y_gru = tmp

    # Griglia normalizzata [0,1]
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, 1, nt)
    X, T = torch.meshgrid(x, t, indexing='xy')
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    # 1) Input per la PINN (coppie x,t)
    X_pinn_heatmap = torch.cat([X_flat, T_flat], dim=1)
    # 2) Input per la GRU (stesse feature [x,t])
    X_gru_heatmap = X_pinn_heatmap.clone()
    # 3) Input concatenato (non usato internamente ma mantenuto per compatibilità)
    X_df_heatmap = torch.cat([X_gru_heatmap, X_pinn_heatmap], dim=1)

    # --- Restore GRU ---
    # input_size = 2
    # hidden_size = 64
    # gru = GRURegressor(input_size, hidden_size)

    # Prefer checkpoint in ./tesi/models, con fallback
    # gru_ckpt_candidates = [
    #     "tesi/models/gru_trained.pth",
    #     "models/gru_trained.pth",
    #     "gru_trained.pth",
    # ]
    # gru_ckpt = next((p for p in gru_ckpt_candidates if os.path.exists(p)), None)
    # if gru_ckpt is None:
    #     raise FileNotFoundError("Checkpoint GRU non trovato (tesi/models/gru_trained.pth).")

    # gru.load_state_dict(torch.load(gru_ckpt, map_location="cpu"))
    # gru.eval()
    # for p in gru.parameters():
    #     p.requires_grad = False

    # --- Restore PINN ---
    # pinn_obj = PINNModel()  # solo costruzione struttura DeepXDE
    # pinn_ckpt_candidates = [
    #     "tesi/models/pinn-25761.pt",
    #     "models/pinn-25761.pt",
    #     "pinn-25761.pt",
    # ]
    # pinn_ckpt = next((p for p in pinn_ckpt_candidates if os.path.exists(p)), None)
    # if pinn_ckpt is None:
    #     raise FileNotFoundError("Checkpoint PINN non trovato (pinn-20399.pt).")
    # pinn_obj.restore(pinn_ckpt, lr=1e-3, load_optimizer=False)

    # Wrapper torch-like per usare la PINN in forward
    # class _PinnWrapper(nn.Module):
    #     def __init__(self, dde_model):
    #         super().__init__()
    #         self.m = dde_model
    #     @torch.no_grad()
    #     def forward(self, xt: torch.Tensor) -> torch.Tensor:
    #         y = self.m.predict(xt.detach().cpu().numpy())
    #         return torch.from_numpy(np.asarray(y)).float()

    # pinn = _PinnWrapper(pinn_obj.model)

    # Costruisco le feature di contesto concatenando le feature GRU e PINN
    # --- 4) Costruisci X_df concatenando GRU + PINN features ---
    X_df_train = torch.cat([X_gru_heatmap, X_pinn_heatmap], dim=1)
    X_df_test  = torch.cat([X_gru_heatmap,  X_pinn_heatmap], dim=1)

    # Preparo i DataLoader per il training e test del blocco di fusione dati
    # --- 5) DataLoader per training/test del blocco di fusione ---
    # Qui non abbiamo dati training/test reali, quindi non usiamo DataLoader per training in questo script

    criterion = nn.MSELoss()

    lamda_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    #lamda_list = [0.25, 0.5, 0.75]

    # Calcolo del massimo errore assoluto globale tra tutti i lamda (usando array NPZ)
    max_diff_global = 0.0
    for lamda in lamda_list:
        # df = DataFusionBlock(gru, pinn, fusion_weight=lamda)  # (OLD) non più usato
        # df.eval()
        # with torch.no_grad():
        #     U_fused = df(X_gru_heatmap, X_pinn_heatmap, X_df_heatmap)
        #     U_fused = U_fused.cpu().numpy().reshape(nt, nx)
        U_fused = lamda * Y_gru + (1 - lamda) * Y_pinn
        diff = np.abs(exact - U_fused)
        max_diff_global = max(max_diff_global, np.max(diff))
    print(f"Global maximum absolute error: {max_diff_global:.6f}")

    for lamda in lamda_list:
        print(f"\n====== Valutazione con lamda = {lamda:.2f} ======")
        # df = DataFusionBlock(gru, pinn, fusion_weight=lamda)  # (OLD) non più usato
        # df.eval()

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
        # with torch.no_grad():
        #     U_fused = df(X_gru_heatmap, X_pinn_heatmap, X_df_heatmap)
        #     U_fused = U_fused.cpu().numpy().reshape(nt, nx)
        U_fused = lamda * Y_gru + (1 - lamda) * Y_pinn
        # Controllo le dimensioni tra exact e U_fused
        if exact.shape != U_fused.shape:
            print(f"Shape mismatch: exact={exact.shape}, U_fused={U_fused.shape}")
            raise ValueError("Dimensioni non compatibili tra exact e U_fused")

        # # Calcola la differenza senza trasporre U_fused
        # diff = np.abs(exact - U_fused)
        # # Calcola e stampa la norma L2 globale
        # l2_error = np.linalg.norm(diff)

                # Calcolo della differenza signed e norma L2 globale
        diff_signed = exact - U_fused  # errore punto-punto
        # Calcolo dell'L2 error:
        # 1. Norma L2 per ogni punto spaziale (colonna, nel tempo)
        # 2. Norma L2 finale sui punti spaziali
        l2_per_spazio = np.linalg.norm(diff_signed, axis=0)
        l2_error = np.linalg.norm(l2_per_spazio)
        diff = np.abs(diff_signed)  # per visualizzazione grafica

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
        plt.show(block=False)
        plt.pause(0.1)
        input(f"Premi Invio per continuare dopo lamda = {lamda:.2f}...")
        plt.close(fig)