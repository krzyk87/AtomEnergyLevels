"""
element_data.py

Single source of truth for per-species physical constants (atomic number, mass
number, ionization energy, valence-slot count, spin-orbit constant, ...).

All constants live in one spreadsheet, ``data/element_constants.xlsx`` (sheet
``species``), one row per *species* — an element in a given ionization stage,
e.g. ``Co I`` (neutral), ``Co II`` (singly ionized), ``K I``. Every script reads
the constants from here instead of hardcoding them, so a growing list of elements
and ions is maintained in exactly one place.

Schema (sheet ``species``)::

    species  symbol  ion_stage  Z  A  ionization_energy_cm-1  max_valence  zeta_3d  is_alkali  source

Typical use::

    from element_data import get_species_constants, get_ionization_energy
    c = get_species_constants("Co I")          # dict of all constants
    e_ion = get_ionization_energy("Co I")      # 63564.6
    e_ion = get_ionization_energy("Co")        # bare symbol → neutral "Co I"
"""

from __future__ import annotations

import os

import pandas as pd

# Default location of the constants table, relative to the project root.
DEFAULT_TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "element_constants.xlsx"
)
SHEET_NAME = "species"
IONIZATION_COL = "ionization_energy_cm-1"

# Cache: {absolute_path: (mtime, DataFrame)} so repeated lookups are cheap but a
# table edited on disk is picked up automatically.
_CACHE: dict = {}


def normalize_species(name) -> str:
    """
    Normalize a species identifier.

    A bare element symbol is interpreted as the neutral atom::

        "Co"     -> "Co I"
        "Co I"   -> "Co I"
        "Co II"  -> "Co II"
        " k i "  -> "K I"     (symbol title-cased, stage upper-cased)
    """
    parts = str(name).strip().split()
    if not parts:
        raise ValueError("Empty species identifier.")
    symbol = parts[0][:1].upper() + parts[0][1:]
    stage = parts[1].upper() if len(parts) > 1 else "I"
    return f"{symbol} {stage}"


def load_species_table(path: str = DEFAULT_TABLE) -> pd.DataFrame:
    """
    Load the constants table, indexed by the normalized ``species`` key.

    Cached on the file's modification time so an edited table is re-read.
    """
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(
            f"Element constants table not found: {abspath}. "
            f"Generate it with tools/build_element_constants.py."
        )
    mtime = os.path.getmtime(abspath)
    cached = _CACHE.get(abspath)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    df = pd.read_excel(abspath, sheet_name=SHEET_NAME)
    if "species" not in df.columns:
        raise ValueError(
            f"Sheet '{SHEET_NAME}' in {abspath} must have a 'species' column; "
            f"found {list(df.columns)}."
        )
    df["species"] = df["species"].map(normalize_species)
    df = df.set_index("species", drop=False)
    _CACHE[abspath] = (mtime, df)
    return df


def get_species_constants(species, path: str = DEFAULT_TABLE) -> dict:
    """
    Return all constants for *species* (or a bare symbol → neutral atom) as a dict.

    Raises KeyError with the available species listed when the lookup misses.
    """
    table = load_species_table(path)
    key = normalize_species(species)
    if key not in table.index:
        raise KeyError(
            f"Species '{key}' not found in {os.path.abspath(path)}. "
            f"Available: {sorted(table.index)}"
        )
    return table.loc[key].to_dict()


def get_ionization_energy(species, path: str = DEFAULT_TABLE) -> float:
    """Return the ionization energy (cm^-1) for *species* (or a bare symbol)."""
    return float(get_species_constants(species, path)[IONIZATION_COL])


def get_ionization_energy_map(path: str = DEFAULT_TABLE) -> dict:
    """Return {species: ionization_energy_cm-1} for vectorized lookups."""
    table = load_species_table(path)
    return {sp: float(e) for sp, e in table[IONIZATION_COL].items()}
