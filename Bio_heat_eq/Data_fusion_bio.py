# Data_fusion_bio.py — Fonde PINN (pinns_meas_cool_1/2.txt) con GRU (gru_preds_bio_heat.npz)
# 1↔1 e 2↔2, con sweep su lambda e salvataggi in risultati_2/fused_meas/

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Risolutore robusto del path al file GRU npz (con fallback a scansione)
def _resolve_gru_npz(base_dir: Path) -> Path:
    required_keys = {"x1", "t1", "preds1", "x2", "t2", "preds2"}
    candidates = [
        base_dir / "risultati_2" / "gru_preds_bio_heat.npz",
        Path("risultati_2") / "gru_preds_bio_heat.npz",
        base_dir.parent / "risultati_2" / "gru_preds_bio_heat.npz",
        Path("/mnt/data/gru_preds_bio_heat.npz"),
    ]
    for p in candidates:
        if p.exists():
            print(f"[FOUND] GRU npz: {p}")
            return p
    # Se non trovato, prova a cercare qualsiasi .npz compatibile nelle cartelle rilevanti
    search_roots = [base_dir, base_dir / "risultati_2", base_dir.parent, Path.cwd()]
    best: Path | None = None
    best_mtime = -1.0
    for root in search_roots:
        if not root.exists():
            continue
        for npz_path in root.rglob("*.npz"):
            try:
                with np.load(npz_path) as z:
                    if required_keys.issubset(set(z.files)):
                        mtime = npz_path.stat().st_mtime
                        if mtime > best_mtime:
                            best, best_mtime = npz_path, mtime
            except Exception:
                continue
    if best is not None:
        print(f"[FOUND by scan] GRU npz: {best}")
        return best
    # Stampa diagnostica utile e ritorno di default
    print("[DEBUG] Nessun file GRU trovato con i nomi attesi o tramite scansione.")
    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] base_dir:", base_dir)
    print("[DEBUG] Tried:", *[str(c) for c in candidates], sep="\n  - ")
    return candidates[0]

# Risolutore dei due file separati (dataset 1 e 2)
def _resolve_gru_npz_split(base_dir: Path) -> dict:
    ds1 = base_dir / "risultati_2" / "gru_preds_bio_heat_ds1.npz"
    ds2 = base_dir / "risultati_2" / "gru_preds_bio_heat_ds2.npz"
    out = {}
    if ds1.exists():
        out["ds1"] = ds1
        print(f"[FOUND] GRU npz ds1: {ds1}")
    if ds2.exists():
        out["ds2"] = ds2
        print(f"[FOUND] GRU npz ds2: {ds2}")
    return out

# ======================
# Config
# ======================
PINN_TXT = {
    "meas_cool_1": "pinns_meas_cool_1.txt",
    "meas_cool_2": "pinns_meas_cool_2.txt",
}
LAMBDAS = [0.00, 0.25, 0.50, 0.75, 1.00]
SAVE_TXT_FLAT = True     # salva anche triplette x t U_fused
INTERACTIVE = True       # mostra tutte le figure e chiudi con Invio alla fine

# ======================
# Utility
# ======================
def load_triplets_txt(path: str):
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path}: attese 3 colonne (x t T)")
    return arr[:,0], arr[:,1], arr[:,2]

def grid_from_triplets(x, t, T):
    xs = np.unique(x)
    ts = np.unique(t)
    nx, nt = len(xs), len(ts)
    # ordina per (t,x) per sicurezza
    order = np.lexsort((x, t))
    x, t, T = x[order], t[order], T[order]
    # mappa
    idx_x = {v:i for i,v in enumerate(xs)}
    idx_t = {v:j for j,v in enumerate(ts)}
    G = np.empty((len(ts), len(xs)), dtype=float)
    for xi,ti,Ti in zip(x,t,T):
        G[idx_t[ti], idx_x[xi]] = Ti
    return xs, ts, G

def interp_to(src_xyT, dst_xy):
    from scipy.interpolate import griddata
    xs, ts, Ts = src_xyT
    Xs, Ts_ = np.meshgrid(xs, ts)
    pts_src = np.c_[Xs.ravel(), Ts_.ravel()]
    vals = Ts.ravel()
    xd, td = dst_xy
    Xd, Td = np.meshgrid(xd, td)
    pts_dst = np.c_[Xd.ravel(), Td.ravel()]
    Ud = griddata(pts_src, vals, pts_dst, method="linear")
    # fallback nearest per eventuali NaN
    nan = np.isnan(Ud)
    if nan.any():
        Ud[nan] = griddata(pts_src, vals, pts_dst[nan], method="nearest")
    return Ud.reshape(len(td), len(xd))

def l2_rmse(A,B):
    D = A - B
    return np.sqrt(np.mean(D*D))

def grids_equal(xs_a, ts_a, xs_b, ts_b, rtol=1e-8, atol=1e-12):
    """Ritorna True se le due griglie coincidono (stessa lunghezza e stessi valori entro tolleranza)."""
    if len(xs_a) != len(xs_b) or len(ts_a) != len(ts_b):
        return False
    return np.allclose(xs_a, xs_b, rtol=rtol, atol=atol) and np.allclose(ts_a, ts_b, rtol=rtol, atol=atol)

# ======================
# Main
# ======================
if __name__ == "__main__":
    base = Path(os.path.dirname(os.path.abspath(__file__)))

    # Preferiamo i file separati (ds1/ds2); se non presenti, fallback al file combinato
    split_paths = _resolve_gru_npz_split(base)
    use_split = ("ds1" in split_paths) and ("ds2" in split_paths)

    out_root = base / "risultati_2" / "fused_meas"
    out_root.mkdir(parents=True, exist_ok=True)

    if not use_split:
        GRU_NPZ = _resolve_gru_npz(base)
        if not GRU_NPZ.exists():
            raise FileNotFoundError(f"File GRU non trovato: {GRU_NPZ}")
        gru = np.load(GRU_NPZ)
        # Preleva blocchi GRU 1 e 2 (modalità combinata)
        x1_g, t1_g = gru["x1"], gru["t1"]
        x2_g, t2_g = gru["x2"], gru["t2"]
        preds1_g   = gru["preds1"]
        preds2_g   = gru["preds2"]
        exact1 = gru["exact1"] if "exact1" in gru.files else None
        exact2 = gru["exact2"] if "exact2" in gru.files else None
    else:
        # Modalità split: carichiamo i due dataset separati
        gru1 = np.load(split_paths["ds1"])  # chiavi: x,t,preds,(exact opzionale)
        gru2 = np.load(split_paths["ds2"])  # chiavi: x,t,preds,(exact opzionale)
        x1_g, t1_g = gru1["x"], gru1["t"]
        x2_g, t2_g = gru2["x"], gru2["t"]
        preds1_g   = gru1["preds"]
        preds2_g   = gru2["preds"]
        exact1     = gru1["exact"] if "exact" in gru1.files else None
        exact2     = gru2["exact"] if "exact" in gru2.files else None
        print("[MODE] Usando file GRU separati (ds1/ds2)")

    # Loop sui due dataset 1↔1, 2↔2
    for key in ["meas_cool_1", "meas_cool_2"]:
        pinn_path = base / PINN_TXT[key]
        if not pinn_path.exists():
            raise FileNotFoundError(f"File PINN non trovato: {pinn_path}")

        # Output dedicato per visualizzare i risultati separatamente
        out_dir = out_root / key
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Carica PINN e ricostruisci griglia
        xp, tp, Tp_flat = load_triplets_txt(str(pinn_path))
        xs_p, ts_p, T_pinn = grid_from_triplets(xp, tp, Tp_flat)  # (nt_p, nx_p)

        # 2) Scegli il blocco GRU corrispondente
        if key.endswith("_1"):
            xs_g, ts_g, T_gru = x1_g, t1_g, preds1_g
            T_exact = exact1
        else:
            xs_g, ts_g, T_gru = x2_g, t2_g, preds2_g
            T_exact = exact2

        print(f"[PAIR] {key}  ⇄  GRU block: {'#1' if key.endswith('_1') else '#2'}")

        # 3) Se le griglie differiscono, interpola la GRU sulla griglia PINN
        if (not grids_equal(xs_p, ts_p, xs_g, ts_g)) or (T_gru.shape != T_pinn.shape):
            T_gru_on_pinn = interp_to((xs_g, ts_g, T_gru), (xs_p, ts_p))
        else:
            T_gru_on_pinn = T_gru

        vmin_T = min(T_pinn.min(), T_gru_on_pinn.min())
        vmax_T = max(T_pinn.max(), T_gru_on_pinn.max())
        # Porta anche 'exact' sulla griglia PINN se disponibile
        T_exact_on_pinn = None
        if T_exact is not None:
            if grids_equal(xs_p, ts_p, xs_g, ts_g) and T_exact.shape == T_pinn.shape:
                T_exact_on_pinn = T_exact
            else:
                T_exact_on_pinn = interp_to((xs_g, ts_g, T_exact), (xs_p, ts_p))
        ref = T_exact_on_pinn if T_exact_on_pinn is not None else T_pinn
        ref_name = "Exact" if T_exact_on_pinn is not None else "PINN"

        # Etichetta leggibile del dataset per i titoli
        dataset_label = "meas cool 1" if key.endswith("_1") else "meas cool 2"

        # vmax per pannello di errore
        max_abs = 0.0
        for lam in LAMBDAS:
            U = lam*T_gru_on_pinn + (1-lam)*T_pinn
            max_abs = max(max_abs, np.max(np.abs(ref - U)))

        scores = []
        Xg, Tg = np.meshgrid(xs_p, ts_p)
        for lam in LAMBDAS:
            # FUSIONE: self.lamda * y_gru + (1 - self.lamda) * y_pinn
            # qui usiamo 'lam' come scalare: U = lam * GRU + (1 - lam) * PINN
            U = lam*T_gru_on_pinn + (1-lam)*T_pinn
            l2 = l2_rmse(U, ref)
            scores.append((lam, l2))
            print(f"[{key}] λ={lam:.2f}  L2={l2:.6f} (ref={ref_name})")

            # Salvataggi
            np.savez(out_dir / f"{key}_fused_lambda{lam:.2f}.npz",
                     U_fused=U, x=xs_p, t=ts_p, T_pinn=T_pinn, T_gru=T_gru_on_pinn)
            if SAVE_TXT_FLAT:
                np.savetxt(out_dir / f"{key}_fused_lambda{lam:.2f}.txt",
                           np.c_[Xg.ravel(), Tg.ravel(), U.ravel()],
                           fmt="%.6f", header="x t U_fused")

            # Figura
            fig, axes = plt.subplots(1,4, figsize=(22,5))
            im0 = axes[0].imshow(T_pinn, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                                 aspect='auto', cmap='viridis', vmin=vmin_T, vmax=vmax_T)
            axes[0].set_title(f'PINN ({dataset_label})'); plt.colorbar(im0, ax=axes[0], label='T')
            im1 = axes[1].imshow(T_gru_on_pinn, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                                 aspect='auto', cmap='viridis', vmin=vmin_T, vmax=vmax_T)
            axes[1].set_title(f'GRU ({dataset_label})'); plt.colorbar(im1, ax=axes[1], label='T')
            im2 = axes[2].imshow(U, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                                 aspect='auto', cmap='viridis', vmin=vmin_T, vmax=vmax_T)
            axes[2].set_title(f'Fused (λ={lam:.2f}) — {dataset_label}'); plt.colorbar(im2, ax=axes[2], label='T')
            im3 = axes[3].imshow(np.abs(ref - U), origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                                 aspect='auto', cmap='viridis', vmin=0, vmax=max_abs)
            axes[3].set_title(f'|{ref_name} - Fused| ({dataset_label})'); plt.colorbar(im3, ax=axes[3], label='|Δ|')
            plt.tight_layout()
            fig.savefig(out_dir / f"{key}_fused_lambda{lam:.2f}.png", dpi=200)
            # Rimosso input per-lambda (INTERACTIVE)


        lam_best, l2_best = sorted(scores, key=lambda z: z[1])[0]
        print(f"[{key}] BEST λ={lam_best:.2f}  L2={l2_best:.6f}  (ref={ref_name})")
        with open(out_dir / f"{key}_fused_summary.txt", "w") as f:
            for lam, l2 in scores:
                f.write(f"lambda={lam:.2f}, L2={l2:.6f}\n")
            f.write(f"\nBEST lambda={lam_best:.2f}, L2={l2_best:.6f} (ref={ref_name})\n")

# === Visualizzazione finale (dopo entrambi i dataset) ===
nfig = len(plt.get_fignums())
print(f"[INFO] Figure aperte totali: {nfig}", flush=True)
if INTERACTIVE and nfig > 0:
    plt.draw()
    plt.show(block=False)
    try:
        input("Premi Invio per chiudere tutte le figure…")
    except KeyboardInterrupt:
        pass
    plt.close('all')
    plt.pause(0.05)
else:
    plt.show()