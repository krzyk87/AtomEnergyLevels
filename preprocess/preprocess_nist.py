"""
preprocess/preprocess_nist.py

Converts raw NIST atomic energy level database files to feature CSVs
suitable for model training.

Input:  data/nist/{Element}_i.csv  (Excel-exported NIST data)
Output: data/{Element}_features.csv

The script:
  1. Reads the NIST Excel-exported CSV (special quoting format)
  2. Parses electron configurations into per-orbital occupancy columns
  3. Derives quantum numbers J, S_qn, L_qn, parity from the Term symbol
  4. Adds atomic constants Z, A, proton_number, neutron_number
  5. Filters out rows with approximated energies (marked with square brackets)
  6. Saves the resulting feature CSV

Usage:
    python preprocess/preprocess_nist.py --element K
    python preprocess/preprocess_nist.py --element Na
    python preprocess/preprocess_nist.py --element all
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Element registry
# ---------------------------------------------------------------------------

ELEMENTS = {
    'Li': {'Z': 3,   'A': 7},
    'Na': {'Z': 11,  'A': 23},
    'K':  {'Z': 19,  'A': 39},
    'Rb': {'Z': 37,  'A': 85},
    'Cs': {'Z': 55,  'A': 133},
    'Fr': {'Z': 87,  'A': 223},
}

# Aufbau filling order: (orbital_name, max_electrons)
# Covers up to Z=118; sufficient for all supported elements.
AUFBAU_ORDER = [
    ('1s', 2),  ('2s', 2),  ('2p', 6),  ('3s', 2),  ('3p', 6),
    ('4s', 2),  ('3d', 10), ('4p', 6),  ('5s', 2),  ('4d', 10),
    ('5p', 6),  ('6s', 2),  ('4f', 14), ('5d', 10), ('6p', 6),
    ('7s', 2),  ('5f', 14), ('6d', 10), ('7p', 6),  ('8s', 2),
    ('6f', 14), ('7d', 10), ('8p', 6),
]

# Spectroscopic letter → l quantum number
L_LETTER_TO_INT = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}

# Term L-letter → L quantum number (different font, capital letters)
TERM_L_MAP = {
    'S': 0, 'P': 1, 'D': 2, 'F': 3,
    'G': 4, 'H': 5, 'I': 6, 'K': 7,
}

# Regex for one orbital segment: e.g. "3p6", "4s", "23d"
_ORBITAL_RE = re.compile(r'^(\d+)([spdfghi])(\d*)$')

# ---------------------------------------------------------------------------
# NIST CSV parser
# ---------------------------------------------------------------------------

def read_nist_csv(filepath: str) -> pd.DataFrame:
    """
    Parse a NIST CSV file in Excel-exported formula format.

    Each data row is stored as a single outer-quoted CSV field containing
    comma-separated Excel formula cells of the form =""value"".  The
    function strips this encoding and returns a clean DataFrame.
    """
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    # First line is a plain comma-separated header (no Excel encoding)
    headers = [h.strip().strip('"') for h in lines[0].strip().split(',')]

    rows = []
    for raw_line in lines[1:]:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Python's csv.reader handles the outer CSV double-quote wrapping.
        # The entire data row arrives as a single string in outer_parsed[0].
        outer_parsed = list(csv.reader([raw_line]))[0]
        inner = outer_parsed[0] if len(outer_parsed) == 1 else ','.join(outer_parsed)

        # inner is comma-separated Excel formula cells: =""val1"",=""val2"",...
        # Some cells contain commas in their values (e.g. J = "7/2,9/2"), so
        # splitting naively on ',' would misalign all subsequent columns.
        # Instead, split on the inter-cell boundary '","=""' which only appears
        # between cells, never inside a value.
        parts = inner.split('","=""')
        values = []
        for i, part in enumerate(parts):
            if i == 0:
                # First cell: leading =" (the inter-cell separator consumed the
                # trailing " of this cell, so nothing to strip on the right).
                val = re.sub(r'^[="]+', '', part)
            else:
                # Subsequent cells: trailing """ artifact left after split.
                val = re.sub(r'"+$', '', part)
            values.append(val.strip())

        if values:
            rows.append(dict(zip(headers, values[:len(headers)])))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Configuration → orbital occupancies
# ---------------------------------------------------------------------------

def parse_explicit_orbitals(config_str: str) -> dict:
    """
    Parse the explicitly listed orbitals from a NIST configuration string.

    The string is dot-separated orbital segments, e.g.:
        '3p6.4s'   → {'3p': 6, '4s': 1}
        '4p6.5s'   → {'4p': 6, '5s': 1}
        '7s'       → {'7s': 1}          (Fr-style, no core prefix)
        '3p6.23d'  → {'3p': 6, '23d': 1}

    If the electron count is omitted, it defaults to 1.
    """
    explicit = {}
    for segment in config_str.split('.'):
        segment = segment.strip()
        m = _ORBITAL_RE.match(segment)
        if m:
            n = int(m.group(1))
            l_letter = m.group(2)
            count = int(m.group(3)) if m.group(3) else 1
            orbital = f'{n}{l_letter}'
            explicit[orbital] = count
    return explicit


def fill_orbital_occupancies(config_str: str, Z: int) -> dict:
    """
    Return the complete orbital occupancy dict for a configuration string.

    Fills implicit core electrons (Z minus explicit electrons) using the
    Aufbau principle, skipping orbitals that are already listed explicitly.

    Returns a dict mapping orbital name → electron count.
    """
    explicit = parse_explicit_orbitals(config_str)
    total_explicit = sum(explicit.values())
    core_remaining = Z - total_explicit

    if core_remaining < 0:
        raise ValueError(
            f"Config '{config_str}' has {total_explicit} explicit electrons "
            f"but Z={Z} — core count would be negative"
        )

    core = {}
    for orbital, max_e in AUFBAU_ORDER:
        if core_remaining <= 0:
            break
        if orbital in explicit:
            continue  # this orbital is already accounted for in explicit
        electrons = min(core_remaining, max_e)
        core[orbital] = electrons
        core_remaining -= electrons

    if core_remaining != 0:
        raise ValueError(
            f"Config '{config_str}' Z={Z}: Aufbau fill left {core_remaining} "
            f"electrons unassigned — extend AUFBAU_ORDER"
        )

    return {**core, **explicit}


def collect_orbital_columns(configs: list, Z: int) -> list:
    """
    Determine the sorted list of orbital column names for the output CSV.

    Includes all orbitals that appear in any configuration (explicit or core).
    Sorted by principal quantum number n first, then by l (s < p < d < f < g).
    """
    all_orbitals = set()
    for config in configs:
        try:
            occupancies = fill_orbital_occupancies(config, Z)
            all_orbitals.update(occupancies.keys())
        except ValueError:
            all_orbitals.update(parse_explicit_orbitals(config).keys())

    def sort_key(name):
        m = _ORBITAL_RE.match(name)
        if not m:
            return (999, 999)
        return (int(m.group(1)), L_LETTER_TO_INT.get(m.group(2), 99))

    return sorted(all_orbitals, key=sort_key)


# ---------------------------------------------------------------------------
# Term symbol → quantum numbers
# ---------------------------------------------------------------------------

def parse_term(term_str: str) -> tuple:
    """
    Parse a spectroscopic term symbol into (S_qn, L_qn, parity).

    Examples:
        '2S'      → (0.5, 0, 0)     even parity, S state
        '2P*'     → (0.5, 1, 1)     odd parity (asterisk), P state
        '3F*'     → (1.0, 3, 1)
        '2[3/2]*' → (0.5, None, 1)  bracket notation: L not well defined
        ''        → (None, None, 0)

    Returns:
        (S_qn, L_qn, parity) where parity=0 is even, parity=1 is odd.
        L_qn is None for bracket-notation terms.
    """
    if not term_str:
        return (None, None, 0)

    parity = 1 if '*' in term_str else 0
    clean = term_str.replace('*', '').strip()

    # Bracket notation: e.g. '2[3/2]'
    if '[' in clean:
        m = re.match(r'^(\d+)\[', clean)
        S_qn = (int(m.group(1)) - 1) / 2 if m else None
        return (S_qn, None, parity)

    # Standard: <multiplicity><L-letter>
    m = re.match(r'^(\d+)([A-Z])', clean)
    if m:
        S_qn = (int(m.group(1)) - 1) / 2
        L_qn = TERM_L_MAP.get(m.group(2), None)
        return (S_qn, L_qn, parity)

    return (None, None, parity)


# ---------------------------------------------------------------------------
# J fraction parser
# ---------------------------------------------------------------------------

def parse_j(j_str: str):
    """
    Convert a J string to float.

    Handles fractions ('1/2' → 0.5, '3/2' → 1.5) and integers ('2' → 2.0).
    Returns None if the string is empty or unparseable.
    """
    j_str = j_str.strip()
    if not j_str:
        return None
    try:
        if '/' in j_str:
            num, den = j_str.split('/', 1)
            return int(num) / int(den)
        return float(j_str)
    except (ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_element(element: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Preprocess NIST data for a single element and return the feature DataFrame.
    """
    if element not in ELEMENTS:
        raise ValueError(
            f"Unknown element '{element}'. "
            f"Supported elements: {', '.join(ELEMENTS.keys())}"
        )

    Z = ELEMENTS[element]['Z']
    A = ELEMENTS[element]['A']
    nist_path = os.path.join(data_dir, 'nist', f'{element}_i.csv')

    if not os.path.exists(nist_path):
        raise FileNotFoundError(f"NIST file not found: {nist_path}")

    print(f"\nProcessing {element}  (Z={Z}, A={A})")
    print(f"  Input : {nist_path}")

    # Step 1 — load raw NIST data
    df = read_nist_csv(nist_path)
    print(f"  Raw rows read: {len(df)}")

    # Step 2 — filter out approximated energies (Prefix = '[')
    if 'Prefix' in df.columns:
        bracketed = df['Prefix'].str.strip() == '['
        n_removed = int(bracketed.sum())
        df = df[~bracketed].reset_index(drop=True)
        print(f"  Removed {n_removed} approximated rows (Prefix='[')")
    else:
        print("  Warning: 'Prefix' column not found — skipping bracket filter")

    print(f"  Rows after filtering: {len(df)}")

    if df.empty:
        raise ValueError(f"No data remaining for {element} after filtering")

    # Step 3 — determine orbital columns from all configurations
    configs = df['Configuration'].str.strip().tolist()
    orbital_cols = collect_orbital_columns(configs, Z)
    print(f"  Orbital columns ({len(orbital_cols)}): "
          f"{orbital_cols[0]} … {orbital_cols[-1]}")

    # Step 4 — build output rows
    parse_errors = 0
    out_rows = []

    multi_j_rows = 0

    for _, row in df.iterrows():
        config      = str(row.get('Configuration', '')).strip()
        term        = str(row.get('Term', '')).strip()
        j_str       = str(row.get('J', '')).strip()
        prefix      = str(row.get('Prefix', '')).strip()
        suffix      = str(row.get('Suffix', '')).strip()
        level_str   = str(row.get('Level (cm-1)', '')).strip()
        unc_str     = str(row.get('Uncertainty (cm-1)', '')).strip()
        lande       = str(row.get('Lande', '')).strip() if 'Lande' in df.columns else ''
        reference   = str(row.get('Reference', '')).strip()

        S_qn, L_qn, parity = parse_term(term)

        try:
            occupancies = fill_orbital_occupancies(config, Z)
        except ValueError as exc:
            print(f"  Warning (config parse): {exc}")
            occupancies = parse_explicit_orbitals(config)
            parse_errors += 1

        try:
            level_val = float(level_str)
        except ValueError:
            level_val = None

        try:
            unc_val = float(unc_str)
        except ValueError:
            unc_val = None

        # Split multi-valued J fields (e.g. '7/2,9/2') into separate rows.
        j_parts = [v.strip() for v in j_str.split(',')]
        if len(j_parts) > 1:
            multi_j_rows += 1

        for j_part in j_parts:
            out_row = {
                'Configuration':    config,
                'Term':             term,
                'Prefix':           prefix,
                'Suffix':           suffix,
                'Lande':            lande,
                'Reference':        reference,
                'Z':                Z,
                'A':                A,
                'proton_number':    Z,
                'neutron_number':   A - Z,
                'J':                parse_j(j_part),
                'S_qn':             S_qn,
                'L_qn':             L_qn,
                'parity':           parity,
            }

            out_row['Level (cm-1)'] = level_val
            out_row['Uncertainty (cm-1)'] = unc_val

            for orb in orbital_cols:
                out_row[orb] = occupancies.get(orb, 0)

            out_rows.append(out_row)

    if multi_j_rows:
        print(f"  {multi_j_rows} row(s) with multiple J values expanded into separate rows")

    if parse_errors:
        print(f"  {parse_errors} rows had configuration parse errors "
              f"(saved with partial orbital data)")

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess NIST atomic energy level data into feature CSVs'
    )
    parser.add_argument(
        '--element',
        type=str,
        required=True,
        help=(
            'Element symbol to process, or "all" to process every available element. '
            f'Supported: {", ".join(ELEMENTS.keys())}'
        ),
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Root data directory containing nist/ subfolder (default: data)',
    )
    args = parser.parse_args()

    if args.element.lower() == 'all':
        elements_to_process = list(ELEMENTS.keys())
    else:
        elements_to_process = [args.element]

    success = 0
    for element in elements_to_process:
        nist_file = os.path.join(args.data_dir, 'nist', f'{element}_i.csv')
        if not os.path.exists(nist_file):
            print(f"Skipping {element}: NIST file not found ({nist_file})")
            continue

        try:
            feature_df = preprocess_element(element, args.data_dir)
            out_path = os.path.join(args.data_dir, f'{element}_features.csv')
            feature_df.to_csv(out_path, index=False)
            print(f"  Output: {out_path}  ({len(feature_df)} rows)")
            success += 1
        except Exception as exc:
            print(f"\nError processing {element}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {success}/{len(elements_to_process)} element(s) processed successfully.")


if __name__ == '__main__':
    main()
