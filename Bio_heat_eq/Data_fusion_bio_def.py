# Data_fusion_bio_def.py — Fusione per-punto con λ random da {0,0.25,0.50,0.75,1.00},
# ricerca su N_TRIALS e salvataggio SOLO del caso migliore (L2 minimo) per ciascun dataset.

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
LAMBDAS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
SAVE_TXT_FLAT = True     # salva anche triplette x t U_fused
INTERACTIVE = True       # mostra figure finali e chiudi con Invio alla fine
N_TRIALS = 200           # numero di prove random per la mappa di lambda per-punto
RNG_SEED = 42            # seed per riproducibilita'
PRINT_EVERY = 10         # frequenza di logging intermedio nel terminale
SAVE_LAMBDA_POINTS_CSV = True   # salva tabella (i,j,x,t,lambda) per ogni punto del best
LAMBDA_POINTS_PREVIEW = 10      # quante righe mostrare a terminale come anteprima (ridotto)
PRINT_ONLY_ON_IMPROVEMENT = True  # stampa i trial solo quando migliora il best (oltre al primo)
SHOW_LAMBDA_MAP_FULL = False      # stampa o meno l'intera mappa λ
COL_MODES_SHOW = 10               # quante colonne mostrare nel riepilogo per-colonna

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

def _histbar(counter: Counter, total: int) -> str:
    # Restituisce una mini-bar chart testuale per i conteggi dei lambda
    parts = []
    for v in LAMBDAS:
        c = int(counter.get(float(v), 0))
        frac = c / max(total, 1)
        bars = int(round(frac * 20))
        parts.append(f"{v:>4}: " + ("#" * bars) + f" {c}")
    return " | ".join(parts)

def _sample_lambda_map(rng: np.random.Generator, shape: tuple[int,int]) -> np.ndarray:
    # Estrae, indipendentemente per ogni cella, un valore da LAMBDAS
    return rng.choice(LAMBDAS, size=shape, replace=True)

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

        # 3b) Costruisci riferimento (exact se disponibile, altrimenti PINN stesso)
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

        rng = np.random.default_rng(RNG_SEED)

        # Pre-calcolo range comuni per le figure del caso migliore
        vmin_T = min(T_pinn.min(), T_gru_on_pinn.min())
        vmax_T = max(T_pinn.max(), T_gru_on_pinn.max())

        best = {
            "l2": np.inf,
            "U": None,
            "lam_map": None,
            "trial": -1,
            "counts": None,
        }

        total_cells = T_pinn.size
        for trial in range(1, N_TRIALS + 1):
            lam_map = _sample_lambda_map(rng, T_pinn.shape)
            U = lam_map * T_gru_on_pinn + (1.0 - lam_map) * T_pinn
            l2 = l2_rmse(U, ref)

            improved = False
            if l2 < best["l2"]:
                counts = Counter(lam_map.ravel())
                best.update({"l2": float(l2), "U": U, "lam_map": lam_map, "trial": trial, "counts": counts})
                improved = True

            # Logging intermedio leggibile in terminale (semplificato)
            do_print = (trial == 1)
            if PRINT_ONLY_ON_IMPROVEMENT:
                do_print = do_print or improved
            else:
                do_print = do_print or ((trial % PRINT_EVERY) == 0)
            if do_print:
                hist = _histbar(Counter(lam_map.ravel()), total_cells)
                print(f"[{key}] trial {trial:4d}/{N_TRIALS}  L2={l2:.6f}  (best={best['l2']:.6f} @ {best['trial']})\n    λ dist: {hist}")

        # Report finale per il dataset corrente
        best_hist = _histbar(best["counts"], total_cells) if best["counts"] else ""
        print(f"[{key}] BEST per-pixel λ-map found @ trial {best['trial']}: L2={best['l2']:.6f} (ref={ref_name})")
        print(f"[{key}] λ-map distribution (best): {best_hist}")

        # --- Report dettagliato dei λ nel caso migliore ---
        lam_map_best = best["lam_map"]

        # Intestazione chiara
        print(f"\n[{key}] === λ-map BEST details ===")

        # 1) Info shape
        print(f"  Shape griglia: {lam_map_best.shape}")

        # 2) Distribuzione valori unici
        vals, counts = np.unique(lam_map_best, return_counts=True)
        print("  Conteggi unici:")
        for v, c in zip(vals, counts):
            print(f"    λ={v:.2f}  ->  {c} celle")

        # 3) Moda globale dei λ
        try:
            vals_all, counts_all = np.unique(lam_map_best, return_counts=True)
            idx_all = np.argmax(counts_all)
            moda_glob = float(vals_all[idx_all])
            frac_all = counts_all[idx_all] / lam_map_best.size
            print(f"  Moda globale: λ={moda_glob:.2f} ({counts_all[idx_all]}/{lam_map_best.size}, {frac_all*100:.1f}%)")
        except Exception as e:
            print(f"  [WARN] moda globale non disponibile: {e}")

        # 4) Mode per colonna (prime COL_MODES_SHOW)
        try:
            nx = lam_map_best.shape[1]
            print(f"  Mode per colonna (prime {min(COL_MODES_SHOW, nx)}):")
            for j in range(min(COL_MODES_SHOW, nx)):
                vals_j, counts_j = np.unique(lam_map_best[:, j], return_counts=True)
                idx = np.argmax(counts_j)
                mode_j = float(vals_j[idx])
                frac_j = counts_j[idx] / lam_map_best.shape[0]
                print(f"    col{j:02d}: λ={mode_j:.2f} ({frac_j*100:.1f}%)")
        except Exception as e:
            print(f"  [WARN] per-column summary non disponibile: {e}")

        # 4) Mappa completa (opzionale, molto lunga)
        if SHOW_LAMBDA_MAP_FULL:
            print("  Mappa completa λ (attenzione: output grande):")
            print(lam_map_best)
        print(f"[{key}] === fine report λ-map ===\n")

        # --- Tabella per-punto dei λ usati (i, j, x, t, λ) ---
        # Costruiamo meshgrid delle coordinate per affiancare gli indici
        Xg, Tg = np.meshgrid(xs_p, ts_p)
        ii, jj = np.indices(lam_map_best.shape)
        rows = np.c_[ii.ravel(), jj.ravel(), Xg.ravel(), Tg.ravel(), lam_map_best.ravel()]


        # Salva CSV completo se richiesto, con header multi-linea che evidenzia il best
        if SAVE_LAMBDA_POINTS_CSV:
            csv_path = out_dir / f"{key}_lambda_points_best.csv"
            header = (
                "BEST random-per-pixel fusion\n"
                f"L2={best['l2']:.6f}, trial={best['trial']}, ref={ref_name}\n"
                "i,j,x,t,lambda"
            )
            np.savetxt(
                csv_path,
                rows,
                delimiter=",",
                fmt=["%d","%d","%.10f","%.10f","%.2f"],
                header=header,
                comments="# "
            )
            print(f"[{key}] File CSV λ per-punto (BEST) salvato: {csv_path}")

        # Salvataggi del caso migliore
        U_best = best["U"]
        lam_map_best = best["lam_map"]

        np.savez(out_dir / f"{key}_fused_best_random.npz",
                 U_fused=U_best, x=xs_p, t=ts_p, T_pinn=T_pinn, T_gru=T_gru_on_pinn,
                 lam_map=lam_map_best, trial=best["trial"], L2=best["l2"], ref_name=ref_name)

        if SAVE_TXT_FLAT:
            np.savetxt(out_dir / f"{key}_fused_best_random.txt",
                       np.c_[Xg.ravel(), Tg.ravel(), U_best.ravel()],
                       fmt="%.6f", header="x t U_fused (best random λ per-pixel)")

        # Figura unica del caso migliore (PINN, GRU, Fused-best, |ref-Fused|)
        max_abs = np.max(np.abs(ref - U_best))
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        im0 = axes[0].imshow(T_pinn, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                             aspect='auto', cmap='viridis', vmin=vmin_T, vmax=vmax_T)
        axes[0].set_title(f'PINN ({dataset_label})'); plt.colorbar(im0, ax=axes[0], label='T')
        im1 = axes[1].imshow(T_gru_on_pinn, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                             aspect='auto', cmap='viridis', vmin=vmin_T, vmax=vmax_T)
        axes[1].set_title(f'GRU ({dataset_label})'); plt.colorbar(im1, ax=axes[1], label='T')
        im2 = axes[2].imshow(U_best, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                             aspect='auto', cmap='viridis', vmin=vmin_T, vmax=vmax_T)
        axes[2].set_title(f'Fused BEST (random λ) — {dataset_label}\nL2={best["l2"]:.6f} (best found at trial {best["trial"]})'); plt.colorbar(im2, ax=axes[2], label='T')
        im3 = axes[3].imshow(np.abs(ref - U_best), origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                             aspect='auto', cmap='viridis', vmin=0, vmax=max_abs)
        axes[3].set_title(f'|{ref_name} - Fused BEST| ({dataset_label})'); plt.colorbar(im3, ax=axes[3], label='|Δ|')
        plt.tight_layout()
        fig.savefig(out_dir / f"{key}_fused_best_random.png", dpi=200)

        # (Opzionale) salva la mappa di λ come immagine discreta, ma NON mostrarla
        try:
            fig_l, ax_l = plt.subplots(1, 1, figsize=(6, 4))
            m = ax_l.imshow(lam_map_best, origin='lower', extent=[xs_p.min(), xs_p.max(), ts_p.min(), ts_p.max()],
                            aspect='auto', cmap='viridis')
            ax_l.set_title(f'λ-map (best) — {dataset_label}')
            plt.colorbar(m, ax=ax_l, label='λ')
            fig_l.savefig(out_dir / f"{key}_lambda_map_best.png", dpi=200)
            plt.close(fig_l)
        except Exception as e:
            print(f"[WARN] Impossibile salvare la λ-map come immagine: {e}")

        # Scrivi un breve summary file per il dataset
        with open(out_dir / f"{key}_fused_summary.txt", "w") as f:
            f.write(f"BEST random-per-pixel fusion\n")
            f.write(f"L2={best['l2']:.6f}, trial={best['trial']}, ref={ref_name}\n")
            f.write(f"λ distribution (best): {best_hist}\n")

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