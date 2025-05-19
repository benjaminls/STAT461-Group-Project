#!/usr/bin/env python


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER CONFIGURATION -----------------
EVENT_ID         = "000001000"   # seven-digit ID   (e.g. 000001000)
DATA_DIR         = Path("./")    # folder containing Parquet files
OUT_DIR          = Path("./")    # where PNGs will be written
PT_CUT_GEV       = 2.0           # pT threshold for "low-pT" views
POINT_SIZE       = 1             # scatter-point size
ALPHA            = 0.3           # point transparency
# ------------------------------------------------------

def load_event(event_id: str, data_dir: Path) -> pd.DataFrame:
    """Return one dataframe with columns:
       hit_id,  x, y, z,   pt  (GeV/c)."""
    hits = pd.read_parquet(data_dir / f"event{event_id}-hits.parquet",
                           columns=["hit_id", "x", "y", "z"])
    truth = pd.read_parquet(data_dir / f"event{event_id}-truth.parquet",
                            columns=["hit_id", "particle_id","tpx", "tpy","tpz"])
    df = hits.merge(truth, on="hit_id", how="left")
    df["pt"] = np.hypot(df["tpx"], df["tpy"])
    return df

# ------------------------------------------------------------------
# UPDATED HELPERS – orange low-pT plots
# ------------------------------------------------------------------
def scatter_ax(ax, x, y, title, color=None):
    """Draw a scatter axis with common styling."""
    ax.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, color=color)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5)

"""
TrackML geometry plots: XY, YZ, XZ (with & without pT filter)
-------------------------------------------------------------
* Reads `<event>-hits.parquet`  and  `<event>-truth.parquet`
* Computes per-hit transverse momentum   pT = sqrt(px^2 + py^2)
* Creates 6 PNG files in the working directory:
    eventXXXXXXX_xy_hits.png
    eventXXXXXXX_xy_hits_lowpt.png
    eventXXXXXXX_yz_hits.png
    eventXXXXXXX_yz_hits_lowpt.png
    eventXXXXXXX_xz_hits.png
    eventXXXXXXX_xz_hits_lowpt.png
"""
def make_geometry_plot(df: pd.DataFrame, xcol: str, ycol: str,
              view_tag: str, xlabel: str, ylabel: str):
    """Create two-panel (all vs low-pT) + single low-pT figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Panel A – all hits (default colour)
    scatter_ax(axes[0], df[xcol], df[ycol], "All hits")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)

    # Panel B – low-pT hits (orange)
    low_df = df[df["pt"] < PT_CUT_GEV]
    scatter_ax(axes[1], low_df[xcol], low_df[ycol],
               f"Hits with pT < {PT_CUT_GEV} GeV/c",
               color="tab:orange")
    axes[1].set_xlabel(xlabel)

    fig.suptitle(f"TrackML Event {EVENT_ID} – {view_tag} Hit Distribution", y=0.95)
    plt.tight_layout()

    composite_path = OUT_DIR / f"event{EVENT_ID}_{view_tag.lower()}_hits.png"
    plt.savefig(composite_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {composite_path.resolve()}")

    # --- single-panel low-pT (orange) ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    scatter_ax(ax2, low_df[xcol], low_df[ycol],
               f"{view_tag} – pT < {PT_CUT_GEV} GeV/c",
               color="tab:orange")
    ax2.set_xlabel(xlabel); ax2.set_ylabel(ylabel)
    plt.tight_layout()

    lowpt_path = OUT_DIR / f"event{EVENT_ID}_{view_tag.lower()}_hits_lowpt.png"
    plt.savefig(lowpt_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] {lowpt_path.resolve()}")

# ------------------------------------------------------------------
#   SAFE η–φ plot helper  (replace the old one)
# ------------------------------------------------------------------
def make_eta_phi_plot(df: pd.DataFrame):
    """Create η–φ scatter plots (all vs low-pT) and save PNGs."""
    # ---------- 1.  η, φ computation (robust) ----------
    pt  = df["pt"]                              # already √(px²+py²)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.arcsinh(df["tpz"] / pt)        # stable, gives ±inf if pt==0
    phi = np.arctan2(df["tpy"], df["tpx"])      # φ in (−π, π]

    df_eta_phi = df.assign(eta=eta, phi=phi)

    # ---------- 2.  keep only finite points ----------
    finite_mask = np.isfinite(df_eta_phi["eta"]) & np.isfinite(df_eta_phi["phi"])
    df_eta_phi  = df_eta_phi[finite_mask]
    low_df      = df_eta_phi[(df_eta_phi["pt"] < PT_CUT_GEV)]

    # ---------- 3.  composite figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    scatter_ax(axes[0], df_eta_phi["eta"], df_eta_phi["phi"], "All hits")
    axes[0].set_xlabel("η");  axes[0].set_ylabel("φ [rad]")

    scatter_ax(axes[1], low_df["eta"], low_df["phi"],
               f"Hits with pT < {PT_CUT_GEV} GeV/c",
               color="tab:orange")
    axes[1].set_xlabel("η")

    # identical, finite axis limits
    eta_min, eta_max = df_eta_phi["eta"].min(), df_eta_phi["eta"].max()
    for ax in axes:
        ax.set_xlim(eta_min, eta_max)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, linestyle=":", linewidth=0.5)

    fig.suptitle(f"TrackML Event {EVENT_ID} – η-φ Distribution", y=0.95)
    plt.tight_layout()

    comp_path = OUT_DIR / f"event{EVENT_ID}_etaphi.png"
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {comp_path.resolve()}")

    # ---------- 4.  single-panel low-pT ----------
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    scatter_ax(ax2, low_df["eta"], low_df["phi"],
               f"η-φ – pT < {PT_CUT_GEV} GeV/c",
               color="tab:orange")
    ax2.set_xlabel("η");  ax2.set_ylabel("φ [rad]")
    ax2.set_xlim(eta_min, eta_max);  ax2.set_ylim(-np.pi, np.pi)
    ax2.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

    low_path = OUT_DIR / f"event{EVENT_ID}_etaphi_lowpt.png"
    plt.savefig(low_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] {low_path.resolve()}")

# ------------------------------------------------------------------
#  Momentum histogram (pT < PT_CUT_GEV)   – log-scaled y-axis
# ------------------------------------------------------------------
def make_momentum_histogram(df: pd.DataFrame):
    """Histogram of total momentum p for hits with pT < PT_CUT_GEV."""
    # ----- compute total momentum -----
    p = np.sqrt(df["tpx"]**2 + df["tpy"]**2 + df["tpz"]**2)

    # ----- apply pT filter -----
    mask = df["pt"] < PT_CUT_GEV
    p_lowpt = p[mask]

    # ----- plot histogram -----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(p_lowpt, bins="auto", histtype="stepfilled", alpha=0.7)
    ax.set_xlabel("Total momentum  p  [GeV/c]")
    ax.set_ylabel("Hit count (log scale)")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5, which="both")
    ax.set_title(f"TrackML Event {EVENT_ID} – Momentum distribution\n"
                 f"(hits with pT < {PT_CUT_GEV} GeV/c)")

    plt.tight_layout()
    out_path = OUT_DIR / f"event{EVENT_ID}_momentum_p_hist.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.resolve()}")

# ------------------------------------------------------------------
#  pT histogram  (hits already filtered to pT < PT_CUT_GEV)
# ------------------------------------------------------------------
def make_transverse_histogram(df: pd.DataFrame):
    """Histogram of transverse momentum pT (after pT filter)."""
    pt_filtered = df.loc[df["pt"] < PT_CUT_GEV, "pt"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pt_filtered, bins="auto", histtype="stepfilled", alpha=0.7)
    ax.set_xlabel(r"Transverse momentum  $p_T$  [GeV/c]")
    ax.set_ylabel("Hit count (log scale)")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5, which="both")
    ax.set_title(f"TrackML Event {EVENT_ID} – $p_T$ distribution\n"
                 f"(hits with $p_T$ < {PT_CUT_GEV} GeV/c)")

    plt.tight_layout()
    out_path = OUT_DIR / f"event{EVENT_ID}_transverse_momentum_hist.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.resolve()}")

# ------------------------------------------------------------------
#  XY plot + a few truth tracks
# ------------------------------------------------------------------
from itertools import cycle
TRACK_COLORS = cycle(["tab:red", "tab:blue", "tab:green",
                      "tab:purple", "tab:brown"])

def make_xy_truth_tracks(df: pd.DataFrame, n_tracks: int = 3):
    """
    Scatter all hits in XY and overlay up to n_tracks truth trajectories
    as coloured lines.
    """
    # ----- pick the most populated real particles -----
    track_counts = (
        df[df["particle_id"] > 0]            # exclude noise (particle_id == 0)
        .groupby("particle_id")
        .size()
        .sort_values(ascending=False)
    )
    chosen = track_counts.head(n_tracks).index.tolist()
    if not chosen:
        print("No real tracks found!")
        return

    # ----- base scatter of all hits -----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["x"], df["y"], s=1, alpha=0.2, color="lightgrey")

    # ----- overlay each chosen track -----
    for pid, color in zip(chosen, TRACK_COLORS):
        hits_track = df[df["particle_id"] == pid].copy()

        # order hits by polar angle so the line looks smooth
        hits_track["angle"] = np.arctan2(hits_track["y"], hits_track["x"])
        hits_track.sort_values("angle", inplace=True)

        ax.plot(hits_track["x"], hits_track["y"],
                lw=1.2, color=color, label=f"particle {pid}")

    # ----- cosmetics -----
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"TrackML Event {EVENT_ID} – XY hits with truth tracks")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path = OUT_DIR / f"event{EVENT_ID}_xy_truth_tracks.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.resolve()}")

# ------------------------------------------------------------------
#  YZ plot + a few truth tracks
# ------------------------------------------------------------------
def make_yz_truth_tracks(df: pd.DataFrame, n_tracks: int = 3):
    """
    Scatter all hits in YZ and overlay up to n_tracks truth trajectories
    as coloured lines.
    """
    # ----- choose most populated real particles -----
    track_counts = (
        df[df["particle_id"] > 0]          # exclude noise
        .groupby("particle_id")
        .size()
        .sort_values(ascending=False)
    )
    chosen = track_counts.head(n_tracks).index.tolist()
    if not chosen:
        print("No real tracks found!")
        return

    # ----- base scatter -----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["z"], df["y"], s=1, alpha=0.2, color="lightgrey")

    # ----- overlay each track -----
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:brown"]
    for pid, color in zip(chosen, colors):
        hits_track = df[df["particle_id"] == pid].copy()
        hits_track.sort_values("z", inplace=True)       # smooth in YZ view
        ax.plot(hits_track["z"], hits_track["y"],
                lw=1.2, color=color, label=f"particle {pid}")

    # ----- formatting -----
    ax.set_aspect("equal")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"TrackML Event {EVENT_ID} – YZ hits with truth tracks")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path = OUT_DIR / f"event{EVENT_ID}_yz_truth_tracks.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.resolve()}")


def main():
    df = load_event(EVENT_ID, DATA_DIR)

    # XY:  x vs y   (x horizontal)
    make_geometry_plot(df, xcol="x", ycol="y", view_tag="XY",
              xlabel="x [mm]", ylabel="y [mm]")

    # YZ:  z vs y   (z horizontal)
    make_geometry_plot(df, xcol="z", ycol="y", view_tag="YZ",
              xlabel="z [mm]", ylabel="y [mm]")

    # XZ:  z vs x   (z horizontal)
    make_geometry_plot(df, xcol="z", ycol="x", view_tag="XZ",
              xlabel="z [mm]", ylabel="x [mm]")

    make_eta_phi_plot(df)
    make_momentum_histogram(df)
    make_transverse_histogram(df)
    make_xy_truth_tracks(df)
    make_yz_truth_tracks(df)
if __name__ == "__main__":
    main()
