"""
analyze_features.py

Standalone feature analysis script for the atomic energy level ML project.
Reads element CSV files directly (does not require AtomicDataset), applies
the same preprocessing logic, and produces a PDF of diagnostic plots.

Usage:
    python analyze_features.py --elements K
    python analyze_features.py --elements K Na Li --data-dir data
    python analyze_features.py --elements K Na --out reports/features_KNa.pdf

Plots produced:
    1.  Feature distributions — histogram per feature, coloured by element
    2.  Correlation matrix — catches redundancies (e.g. L_qn vs val_e1_l)
    3.  Mutual information ranking — 2×2 subplots for log_binding_energy,
        raw_energy (Level cm-1), binding_energy, inverse_binding_energy
    4.  Feature vs log-binding-energy scatter — reveals Rydberg structure
    4b. Feature vs raw-energy (Level cm-1) scatter
    4c. Feature vs binding-energy scatter
    5.  Rydberg residuals — (true level − rydberg_pred) vs n_star, coloured by l
    6.  n* distribution — validates quantum defect per element & series
    7.  Quantum defect stability — δ vs n, coloured by l (ideally flat lines)
    8.  Feature range comparison across elements — shows scale mismatch
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Constants (same as AtomicDataset.py)
# ---------------------------------------------------------------------------

IONIZATION_ENERGIES = {
    "Li": (3,  43487.114),
    "Na": (11, 41449.451),
    "K":  (19, 35009.814),
    "Rb": (37, 33690.81),
    "Cs": (55, 31406.467),
    "Fr": (87, 32848.872),
}
R_INF = 109737.316   # Rydberg constant, cm⁻¹

ELEMENT_COLORS = {
    "Li": "#E24B4A", "Na": "#BA7517", "K": "#1D9E75",
    "Rb": "#378ADD", "Cs": "#7F77DD", "Fr": "#D4537E",
}
L_NAMES = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
L_COLORS = ["#1D9E75", "#378ADD", "#BA7517", "#D4537E", "#7F77DD", "#888780"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Feature analysis for atomic energy level data")
    p.add_argument("--elements", nargs="+", default=["Li","Na","K","Rb","Cs","Fr"],
                   help="Element symbols to analyse (default: K)")
    p.add_argument("--data-dir", default="../data",
                   help="Directory containing {Element}_features.csv files")
    p.add_argument("--out", default=None,
                   help="Output PDF path (default: reports/features_{'_'.join(elements)}.pdf)")
    p.add_argument("--level-col", default="Level (cm-1)",
                   help="Name of the energy level column in CSV")
    p.add_argument("--no-rydberg", action="store_true",
                   help="Skip Rydberg feature computation")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_element(element: str, data_dir: str, level_col: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{element}_features.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found")
    df = pd.read_csv(path)
    df["Element"] = element
    if level_col not in df.columns:
        sys.exit(f"ERROR: column '{level_col}' not in {path}. Available: {df.columns.tolist()}")
    return df


def add_valence_encoding(df: pd.DataFrame, max_valence: int = 1) -> pd.DataFrame:
    """
    Minimal valence encoding matching AtomicDataset logic:
    sorts electrons by Madelung rule, keeps last max_valence, pads at front.
    """
    import re
    orbital_pattern = re.compile(r"^(\d+)([spdfgh])$")
    orbital_cols = [c for c in df.columns if orbital_pattern.match(c)]

    def orbital_order(col):
        m = orbital_pattern.match(col)
        n, l_let = int(m.group(1)), m.group(2)
        l = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}[l_let]
        return (n + l, n, l)

    orbital_cols_sorted = sorted(orbital_cols, key=orbital_order)

    feature_cols = [f"val_e{i+1}_n" for i in range(max_valence)] + \
                   [f"val_e{i+1}_l" for i in range(max_valence)]
    # Interleave: val_e1_n, val_e1_l, val_e2_n, val_e2_l, ...
    feature_cols = []
    for i in range(max_valence):
        feature_cols.extend([f"val_e{i+1}_n", f"val_e{i+1}_l"])

    rows = []
    for _, row in df.iterrows():
        electrons = []
        for col in orbital_cols_sorted:
            val = row[col]
            if pd.isna(val) or int(val) == 0:
                continue
            m = orbital_pattern.match(col)
            n, l_let = int(m.group(1)), m.group(2)  # group(1) captures full digit sequence
            l = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}[l_let]
            electrons.extend([(n, l)] * int(val))

        valence = electrons[-max_valence:] if len(electrons) >= max_valence else electrons
        while len(valence) < max_valence:
            valence.insert(0, (0, 0))

        flat = []
        for (n, l) in valence:
            flat.extend([n, l])
        rows.append(flat)

    enc = pd.DataFrame(rows, columns=feature_cols, index=df.index)
    return pd.concat([df, enc], axis=1)


def add_rydberg_features(df: pd.DataFrame, level_col: str) -> pd.DataFrame:
    """
    Compute quantum defect (δ) from all rows, then per-element per-l mean,
    and derive n_star and rydberg_pred. Uses the outermost valence electron.
    """
    ALKALI = set(IONIZATION_ENERGIES.keys())

    max_ev = sum(1 for c in df.columns if c.startswith("val_e") and c.endswith("_n"))
    n_cols = [f"val_e{i+1}_n" for i in range(max_ev)]
    l_cols = [f"val_e{i+1}_l" for i in range(max_ev)]

    def outer_electron(row):
        for nc, lc in zip(reversed(n_cols), reversed(l_cols)):
            n = row.get(nc, 0)
            if pd.notna(n) and int(n) > 0:
                return int(n), int(row[lc])
        return None, None

    # Step 1: compute quantum defect for every alkali row
    defects = []
    for _, row in df.iterrows():
        el = row["Element"]
        if el not in ALKALI:
            defects.append(np.nan)
            continue
        n, l = outer_electron(row)
        if n is None:
            defects.append(np.nan)
            continue
        E_ion = IONIZATION_ENERGIES[el][1]
        level = float(row[level_col])
        binding = E_ion - level
        if binding <= 0:
            defects.append(np.nan)
            continue
        n_star_val = np.sqrt(R_INF / binding)
        defects.append(n - n_star_val)

    df = df.copy()
    df["quantum_defect"] = defects

    # Step 2: mean δ per (element, l) — use ALL data as proxy for training
    delta_map = (
        df.groupby(["Element", "val_e1_l"])["quantum_defect"]
        .mean()
        .to_dict()
    )

    # Step 3: apply
    n_stars, ryd_preds, one_over = [], [], []
    for _, row in df.iterrows():
        el = row["Element"]
        if el not in ALKALI:
            n_stars.append(np.nan); ryd_preds.append(np.nan); one_over.append(np.nan)
            continue
        n, l = outer_electron(row)
        if n is None:
            n_stars.append(np.nan); ryd_preds.append(np.nan); one_over.append(np.nan)
            continue
        E_ion = IONIZATION_ENERGIES[el][1]
        delta = delta_map.get((el, l), 0.0)
        n_eff = n - delta
        if n_eff <= 0:
            n_stars.append(np.nan); ryd_preds.append(np.nan); one_over.append(np.nan)
            continue
        n_stars.append(n_eff)
        ryd_preds.append(E_ion - R_INF / n_eff**2)
        one_over.append(1.0 / n_eff**2)

    df["n_star"]            = n_stars
    df["rydberg_pred"]      = ryd_preds
    df["one_over_nstar_sq"] = one_over
    df["binding_energy"]    = df.apply(
        lambda r: IONIZATION_ENERGIES.get(r["Element"], (None, np.nan))[1] - r[level_col],
        axis=1
    )
    return df

# -----
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features: total electrons, valence electrons, core electrons,
    unpaired electrons, max principal quantum number.

    Bug fixed: uses regex to parse n from orbital names so n>=10 (e.g. 10s, 46d)
    are handled correctly. The old code used int(col[0]) which gave 1 for '10s'.
    """
    import re
    orbital_pattern = re.compile(r"^(\d+)([spdfgh])$")
    orbital_cols = [c for c in df.columns if orbital_pattern.match(c)]

    df = df.copy()
    total_electrons = []
    max_principal_ns = []
    valence_electrons_list = []

    for _, row in df.iterrows():
        total = 0
        max_n = 0
        for col in orbital_cols:
            val = row.get(col, 0)
            if pd.isna(val):
                continue
            count = int(val)
            if count > 0:
                m = orbital_pattern.match(col)
                n = int(m.group(1))  # correct: captures "46" from "46d"
                total += count
                max_n = max(max_n, n)
        total_electrons.append(total)
        max_principal_ns.append(max_n)

        # Valence electrons: those in orbitals with n == max_n
        valence = 0
        for col in orbital_cols:
            val = row.get(col, 0)
            if pd.isna(val):
                continue
            count = int(val)
            if count > 0:
                m = orbital_pattern.match(col)
                n = int(m.group(1))
                if n == max_n:
                    valence += count
        valence_electrons_list.append(valence)

    df["total_electrons"]   = total_electrons
    df["max_principal_n"]   = max_principal_ns
    df["valence_electrons"] = valence_electrons_list
    df["core_electrons"]    = df["total_electrons"] - df["valence_electrons"]
    if "S_qn" in df.columns:
        df["unpaired_electrons"] = (2 * df["S_qn"]).round().astype(int)
    else:
        df["unpaired_electrons"] = 0

    return df

# Plotting helpers
# ---------------------------------------------------------------------------

FIGSIZE_FULL = (14, 8)
STYLE = {"alpha": 0.75, "edgecolor": "none"}


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=13, fontweight="bold", y=1.01)


def _legend_elements(elements):
    from matplotlib.patches import Patch
    return [Patch(facecolor=ELEMENT_COLORS.get(e, "gray"), label=e) for e in elements]

# ---------------------------------------------------------------------------
# Plot 1: Feature distributions
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, features: list, elements: list,
                                level_col: str, pdf: PdfPages):
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in df.columns:
            ax.set_visible(False)
            continue
        for el in elements:
            sub = df[df["Element"] == el][feat].dropna()
            if sub.empty:
                continue
            color = ELEMENT_COLORS.get(el, "gray")
            ax.hist(sub, bins=20, color=color, label=el,
                    alpha=0.6, edgecolor="none", density=True)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("value", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if len(elements) > 1:
        fig.legend(handles=_legend_elements(elements), loc="upper right",
                   fontsize=9, title="Element")

    _suptitle(fig, "Feature distributions (by element)")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, features: list, pdf: PdfPages,
                             targets: list = None):
    """
    targets: list of additional column names (target variables) to append to the matrix,
    letting you see which features correlate with each target type.
    """
    all_cols = list(features)
    if targets:
        all_cols += [t for t in targets if t in df.columns and t not in all_cols]
    available = [f for f in all_cols if f in df.columns]
    corr = df[available].corr()
    n = len(available)
    n_feat = len([f for f in features if f in df.columns])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.75),
                                    max(6, n * 0.72)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Bold target column/row labels
    xlabels = [f"★ {c}" if (targets and c in targets) else c for c in available]
    ylabels = xlabels
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ylabels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if abs(v) > 0.25:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.7 else "black")
    # Draw separator line between features and targets
    if targets:
        sep = n_feat - 0.5
        ax.axhline(sep, color="white", lw=2)
        ax.axvline(sep, color="white", lw=2)
    _suptitle(fig, "Feature correlation matrix  (★ = target variables)\n"
              "|r|>0.8 suggests redundancy; look for L_qn≈val_e1_l, S_qn constant")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3: Mutual information ranking — multiple targets as subplots
# ---------------------------------------------------------------------------

def plot_mutual_information(df: pd.DataFrame, features: list,
                             targets: list, pdf: PdfPages):
    """
    Render MI ranking for every target in *targets* as subplots on one PDF page.

    targets: list of (column_name, display_label) tuples.
             column_name must exist in df; display_label is shown in the subplot title.
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    n_targets = len(targets)
    ncols = 2
    nrows = (n_targets + ncols - 1) // ncols
    bar_height_per_feat = 0.38
    subplot_h = max(4, len(features) * bar_height_per_feat)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(18, subplot_h * nrows))
    axes = np.array(axes).flatten()

    for i, (col_name, label) in enumerate(targets):
        ax = axes[i]
        if col_name not in df.columns:
            ax.set_visible(False)
            continue

        available = [f for f in features if f in df.columns and f != col_name]
        target_vals = df[col_name].dropna()
        common_idx = df[available].dropna().index.intersection(target_vals.index)
        if len(common_idx) == 0:
            ax.set_visible(False)
            continue

        X = df.loc[common_idx, available].values
        y = df.loc[common_idx, col_name].values

        X_sc = StandardScaler().fit_transform(X)
        mi = mutual_info_regression(X_sc, y, random_state=42)

        order = np.argsort(mi)[::-1]
        sorted_feats = [available[k] for k in order]
        sorted_mi = mi[order]
        median_mi = np.median(sorted_mi)

        colors = ["#1D9E75" if v > median_mi else "#888780" for v in sorted_mi]
        bars = ax.barh(sorted_feats[::-1], sorted_mi[::-1],
                       color=colors[::-1], height=0.6)
        ax.set_xlabel("Mutual information", fontsize=9)
        ax.axvline(median_mi, color="#BA7517", ls="--", lw=1, label="median")
        ax.legend(fontsize=8)
        ax.set_title(f"target: {label}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        x_pad = max(sorted_mi) * 0.01 if max(sorted_mi) > 0 else 0.001
        for bar, val in zip(bars, sorted_mi[::-1]):
            ax.text(val + x_pad, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)

    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    _suptitle(fig,
              "Mutual information ranking — feature importance per target\n"
              "(higher = more informative; near-zero = essentially noise for that target)")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4: Feature vs target scatter
# ---------------------------------------------------------------------------

def plot_feature_target_scatter(df: pd.DataFrame, features: list,
                                  target: str, elements: list, pdf: PdfPages):
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in df.columns or target not in df.columns:
            ax.set_visible(False)
            continue
        for el in elements:
            sub = df[df["Element"] == el][[feat, target]].dropna()
            if sub.empty:
                continue
            ax.scatter(sub[feat], sub[target], s=12, alpha=0.5,
                       color=ELEMENT_COLORS.get(el, "gray"), label=el)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(target, fontsize=8)
        ax.set_title(f"{feat} vs {target}", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if len(elements) > 1:
        fig.legend(handles=_legend_elements(elements), loc="upper right",
                   fontsize=9, title="Element")

    _suptitle(fig, f"Feature vs target ({target}) scatter plots")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5: Rydberg residuals
# ---------------------------------------------------------------------------

def plot_rydberg_residuals(df: pd.DataFrame, level_col: str,
                            elements: list, pdf: PdfPages):
    if "rydberg_pred" not in df.columns or "n_star" not in df.columns:
        return

    df = df.copy()
    df["rydberg_residual"] = df[level_col] - df["rydberg_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: residual vs n_star, coloured by l
    ax = axes[0]
    unique_ls = sorted(df["val_e1_l"].dropna().unique())
    for l_val in unique_ls:
        sub = df[df["val_e1_l"] == l_val][["n_star", "rydberg_residual"]].dropna()
        if sub.empty:
            continue
        color = L_COLORS[int(l_val) % len(L_COLORS)]
        ax.scatter(sub["n_star"], sub["rydberg_residual"], s=15, alpha=0.6,
                   color=color, label=L_NAMES.get(int(l_val), str(int(l_val))))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("n* (effective quantum number)", fontsize=10)
    ax.set_ylabel("Residual: true − Rydberg prediction (cm⁻¹)", fontsize=10)
    ax.set_title("Rydberg residuals vs n*", fontsize=11)
    ax.legend(title="l", fontsize=9, title_fontsize=9)
    ax.tick_params(labelsize=8)

    # Right: residual vs n_star, coloured by element
    ax = axes[1]
    for el in elements:
        sub = df[df["Element"] == el][["n_star", "rydberg_residual"]].dropna()
        if sub.empty:
            continue
        ax.scatter(sub["n_star"], sub["rydberg_residual"], s=15, alpha=0.6,
                   color=ELEMENT_COLORS.get(el, "gray"), label=el)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("n* (effective quantum number)", fontsize=10)
    ax.set_ylabel("Residual (cm⁻¹)", fontsize=10)
    ax.set_title("Rydberg residuals by element", fontsize=11)
    ax.legend(title="Element", fontsize=9)
    ax.tick_params(labelsize=8)

    _suptitle(fig,
        "Rydberg residuals  (true level − Rydberg prediction)\n"
        "Large residuals at low n* = extrapolation challenge; "
        "ideally residuals should be small and symmetric around 0")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 6: Quantum defect stability
# ---------------------------------------------------------------------------

def plot_quantum_defect(df: pd.DataFrame, elements: list, pdf: PdfPages):
    if "quantum_defect" not in df.columns or "val_e1_n" not in df.columns:
        return

    unique_ls = sorted(df["val_e1_l"].dropna().unique())
    ncols = min(len(unique_ls), 3)
    nrows = (len(unique_ls) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, l_val in enumerate(unique_ls):
        ax = axes[i]
        for el in elements:
            sub = df[(df["Element"] == el) & (df["val_e1_l"] == l_val)][
                ["val_e1_n", "quantum_defect"]
            ].dropna()
            if sub.empty:
                continue
            sub = sub.sort_values("val_e1_n")
            ax.plot(sub["val_e1_n"], sub["quantum_defect"],
                    "o-", color=ELEMENT_COLORS.get(el, "gray"), ms=5,
                    alpha=0.8, label=el)
        ax.set_title(f"l={int(l_val)} ({L_NAMES.get(int(l_val), '?')})", fontsize=10)
        ax.set_xlabel("n (principal quantum number)", fontsize=9)
        ax.set_ylabel("δ = n − n*", fontsize=9)
        ax.tick_params(labelsize=8)
        # Ideal: flat line (constant quantum defect)
        mean_delta = df[df["val_e1_l"] == l_val]["quantum_defect"].mean()
        if not np.isnan(mean_delta):
            ax.axhline(mean_delta, color="gray", ls="--", lw=0.8, label=f"mean δ={mean_delta:.3f}")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    _suptitle(fig,
        "Quantum defect δ = n − n* per angular momentum series\n"
        "Ideally a flat horizontal line; slope indicates Rydberg-Ritz correction needed")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 7: Feature range comparison across elements
# ---------------------------------------------------------------------------

def plot_feature_ranges(df: pd.DataFrame, features: list,
                         elements: list, pdf: PdfPages):
    available = [f for f in features if f in df.columns]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(max(10, 2 * n), 5))
    if n == 1:
        axes = [axes]

    for i, feat in enumerate(available):
        ax = axes[i]
        data_by_el = []
        labels = []
        for el in elements:
            sub = df[df["Element"] == el][feat].dropna()
            if not sub.empty:
                data_by_el.append(sub.values)
                labels.append(el)
        if not data_by_el:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(data_by_el, showmedians=True,
                               showextrema=True)
        for j, pc in enumerate(parts["bodies"]):
            el = labels[j]
            pc.set_facecolor(ELEMENT_COLORS.get(el, "gray"))
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("value", fontsize=8)
        ax.tick_params(labelsize=7)

    _suptitle(fig,
        "Feature value ranges per element (violin plots)\n"
        "Mismatched ranges across elements can confuse StandardScaler — look for outliers")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 8: n_star distribution — validates Rydberg computation
# ---------------------------------------------------------------------------

def plot_nstar_distribution(df: pd.DataFrame, elements: list, pdf: PdfPages):
    if "n_star" not in df.columns:
        return

    fig, axes = plt.subplots(1, len(elements), figsize=(5 * len(elements), 4),
                              squeeze=False)
    axes = axes[0]

    for i, el in enumerate(elements):
        ax = axes[i]
        sub = df[df["Element"] == el][["n_star", "val_e1_l"]].dropna()
        unique_ls = sorted(sub["val_e1_l"].unique())
        for l_val in unique_ls:
            lsub = sub[sub["val_e1_l"] == l_val]["n_star"]
            color = L_COLORS[int(l_val) % len(L_COLORS)]
            ax.hist(lsub, bins=15, color=color, alpha=0.6, edgecolor="none",
                    label=L_NAMES.get(int(l_val), str(int(l_val))))
        ax.set_title(f"{el}", fontsize=11)
        ax.set_xlabel("n* (effective quantum number)", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(title="l", fontsize=8, title_fontsize=8)
        ax.tick_params(labelsize=8)

    _suptitle(fig,
        "n* (effective principal quantum number) distribution per element and series\n"
        "Gaps at low n* identify states where Rydberg extrapolation is required")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    elements = args.elements

    out_path = args.out or os.path.join(
        "reports", f"features_{'_'.join(elements)}.pdf"
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Load data
    print(f"Loading elements: {elements}")
    dfs = []
    for el in elements:
        d = load_element(el, args.data_dir, args.level_col)
        d = add_valence_encoding(d, max_valence=1)
        dfs.append(d)
        print(f"  {el}: {len(d)} samples, level range "
              f"{d[args.level_col].min():.0f}–{d[args.level_col].max():.0f} cm⁻¹")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(df)} samples")

    if not args.no_rydberg:
        print("Computing Rydberg features...")
        df = add_rydberg_features(df, args.level_col)
        df["log_binding_energy"]     = np.log(df["binding_energy"].clip(lower=1e-6))
        df["inverse_binding_energy"] = 1.0 / df["binding_energy"].replace(0, np.nan)
        print("  Done.")

    print("Computing derived features...")
    df = add_derived_features(df)
    print("  Done.")

    # Define feature sets to analyse
    base_features = ["val_e1_n", "val_e1_l", "J"]
    quantum_features = ["S_qn", "L_qn"]
    atomic_features = ["Z", "A"]
    rydberg_features = ["n_star", "rydberg_pred", "one_over_nstar_sq"]
    derived_features = ["total_electrons", "max_principal_n", "valence_electrons",
                        "core_electrons", "unpaired_electrons"]

    all_features = (
        base_features +
        [f for f in quantum_features if f in df.columns] +
        [f for f in atomic_features if f in df.columns] +
        [f for f in rydberg_features if f in df.columns] +
        [f for f in derived_features if f in df.columns]
    )
    all_features = [f for f in all_features if f in df.columns]

    # Build target columns for correlation analysis
    target_cols = []
    if "log_binding_energy" in df.columns:
        target_cols.append("log_binding_energy")
    if "binding_energy" in df.columns:
        target_cols.append("binding_energy")
    if args.level_col in df.columns:
        target_cols.append(args.level_col)

    # MI targets: (column_name, display_label) — keep only those present in df
    _mi_candidates = [
        ("log_binding_energy",     "log_binding_energy"),
        (args.level_col,           f"raw_energy ({args.level_col})"),
        ("binding_energy",         "binding_energy"),
        ("inverse_binding_energy", "inverse_binding_energy"),
    ]
    mi_targets = [(col, lbl) for col, lbl in _mi_candidates if col in df.columns]

    # Scatter targets beyond log_binding_energy
    scatter_extra_targets = [t for t in [args.level_col, "binding_energy"]
                              if t in df.columns]

    target_for_log_scatter = "log_binding_energy" if "log_binding_energy" in df.columns \
        else args.level_col

    print(f"\nFeatures to analyse: {all_features}")
    print(f"MI targets: {[lbl for _, lbl in mi_targets]}")
    print(f"\nWriting report to: {out_path}")

    with PdfPages(out_path) as pdf:

        # Title page
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        ax.text(0.5, 0.65,
                "Feature Analysis Report",
                ha="center", va="center", fontsize=22, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.5,
                f"Elements: {', '.join(elements)}    ·    n={len(df)} total samples",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes, color="#5a5a5a")
        ax.text(0.5, 0.38,
                f"Features analysed: {', '.join(all_features)}",
                ha="center", va="center", fontsize=11,
                transform=ax.transAxes, color="#7a7a7a")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        print("  [1] Feature distributions...")
        plot_feature_distributions(df, all_features, elements, args.level_col, pdf)

        print("  [2] Correlation matrix...")
        plot_correlation_matrix(df, all_features, pdf, targets=target_cols)

        print("  [3] Mutual information ranking (all targets)...")
        plot_mutual_information(df, all_features, mi_targets, pdf)

        print("  [4] Feature vs log-binding-energy scatter...")
        plot_feature_target_scatter(df, all_features, target_for_log_scatter, elements, pdf)

        for extra_target in scatter_extra_targets:
            print(f"  [4+] Feature vs {extra_target} scatter...")
            plot_feature_target_scatter(df, all_features, extra_target, elements, pdf)

        if not args.no_rydberg:
            print("  [5] Rydberg residuals...")
            plot_rydberg_residuals(df, args.level_col, elements, pdf)

            print("  [6] Quantum defect stability...")
            plot_quantum_defect(df, elements, pdf)

        print("  [7] Feature ranges by element...")
        plot_feature_ranges(df, all_features, elements, pdf)

        if not args.no_rydberg:
            print("  [8] n* distribution...")
            plot_nstar_distribution(df, elements, pdf)

    print(f"\nDone. Report: {out_path}")


if __name__ == "__main__":
    main()
