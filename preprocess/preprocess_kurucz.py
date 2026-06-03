"""
preprocess/preprocess_kurucz.py

Converts Kurucz atomic energy level text files for Co I into a feature CSV
that matches the schema produced by preprocess_nist.py.

Input:  data/kurucz/Co-atom-AI.txt       (even-parity levels)
        data/kurucz/Coi-odd-atom-AI.txt  (odd-parity levels)

Output: data/kurucz/Co_kurucz_raw.xlsx   (intermediate; original Kurucz columns)
        data/Co_features_kurucz.csv       (final; same schema as Co_features.csv)

Text-file format
----------------
  • First line: column-header (skipped).
  • J-value headers: 'J=3/2' (even file) or 'J= 3/2 \\' (odd file).
    The current J is carried forward to all rows that follow until the next header.
  • Data rows: space-separated fields
        [OBS.LEVEL]  EIGENVALUE  T-W  <config> ; <term>  calc.gJ  [obs.gJ  [OBS-CALC]]
    OBS.LEVEL may be blank (odd file, purely theoretical levels) → row skipped.
    OBS.LEVEL == 1.000 is a placeholder used in the even file → row skipped.
  • Configuration notation: concatenated orbital segments and coupling terms,
    e.g. '3d7(4F)4s2(1S)' — converted to NIST dot-notation for the orbital parser.

Usage:
    python preprocess/preprocess_kurucz.py
    python preprocess/preprocess_kurucz.py --data_dir data
"""

import argparse
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Import shared helpers from the co-located NIST preprocessing module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_nist import (
    ELEMENTS,
    fill_orbital_occupancies,
    parse_explicit_orbitals,
    collect_orbital_columns,
    parse_term,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ELEMENT = 'Co'
_Z = ELEMENTS[_ELEMENT]['Z']   # 27
_A = ELEMENTS[_ELEMENT]['A']   # 59

# Matches J-value header lines in both file variants:
#   even file: 'J=1/2'  or ' J=3/2'
#   odd  file: 'J= 1/2 \\'
_J_HEADER_RE = re.compile(r'^\s*J\s*=\s*(\d+(?:/\d+)?)')

# Matches a token that is a pure number (int or float, optionally signed)
_PURE_NUM_RE = re.compile(r'^-?[\d.]+$')


# ---------------------------------------------------------------------------
# Configuration-string conversion
# ---------------------------------------------------------------------------

def kurucz_config_to_nist(config_str: str) -> str:
    """
    Convert a Kurucz configuration string to NIST dot-notation.

    Kurucz writes orbital segments and coupling terms concatenated without
    separators: e.g. '3d7(4F)4s2(1S)' or '3d8(3F)4p  (2P)'.
    NIST uses dot-separated tokens:  '3d7.(4F).4s2.(1S)'.

    The tokeniser recognises:
      orbital segments  \d+[spdfghi]\d*   →  e.g. '3d7', '4s2', '4p'
      coupling terms    \([^)]+\)          →  e.g. '(4F)', '(3P)'
    Spaces are stripped first (handles '3d8(3F)4p  (2P)').
    Unclosed parentheses are silently discarded.

    The resulting string is accepted by the existing fill_orbital_occupancies()
    function, which ignores the coupling terms in parentheses.
    """
    s = config_str.replace(' ', '')
    tokens = re.findall(r'\([^)]+\)|\d+[spdfghi]\d*', s)
    return '.'.join(tokens) if tokens else s


# ---------------------------------------------------------------------------
# Line parser
# ---------------------------------------------------------------------------

def _parse_kurucz_line(line: str):
    """
    Parse one Kurucz data row (call only after J-header detection).

    Returns a dict of raw fields, or None when the row should be skipped.

    Left of ';':  [OBS.LEVEL]  EIGENVALUE  T-W  <configuration text>
                  OBS.LEVEL is detected by counting leading pure-numeric tokens.
                  When only one numeric token is present, OBS.LEVEL is absent
                  (purely theoretical level in the odd file → skip).
                  When OBS.LEVEL == 1.000 it is an even-file placeholder → skip.

    Right of ';': <term>  [calc.gJ  [obs.gJ  [OBS-CALC]]]
    """
    line = line.rstrip()
    if ';' not in line:
        return None

    semi = line.index(';')
    left_tokens  = line[:semi].split()
    right_tokens = line[semi + 1:].split()

    # Separate leading pure-numeric fields from configuration text
    n_i = 0
    while n_i < len(left_tokens) and _PURE_NUM_RE.match(left_tokens[n_i]):
        n_i += 1

    num_fields = [float(t) for t in left_tokens[:n_i]]
    config_raw = ''.join(left_tokens[n_i:])   # join fragments (removes any internal spaces)

    # Map numeric fields to named columns
    if len(num_fields) == 3:
        obs_level, eigenvalue, t_w = num_fields
    elif len(num_fields) == 2:
        obs_level, eigenvalue, t_w = num_fields[0], num_fields[1], None
    elif len(num_fields) == 1:
        # Only EIGENVALUE present — no observed level
        obs_level, eigenvalue, t_w = None, num_fields[0], None
    else:
        return None

    # Skip rows without an observed level
    #   blank  → purely theoretical (odd file)
    #   1.000  → placeholder flag used in the even file
    if obs_level is None or obs_level == 1.0:
        return None

    if not right_tokens:
        return None

    term_raw = right_tokens[0]

    # Remaining right tokens are gJ values and their difference
    gj_vals = []
    for tok in right_tokens[1:]:
        try:
            gj_vals.append(float(tok))
        except ValueError:
            pass

    return {
        'Configuration_raw': config_raw,
        'Term_raw':          term_raw,
        'OBS.LEVEL':         obs_level,
        'EIGENVALUE':        eigenvalue,
        'T-W':               t_w,
        'calc.gJ':           gj_vals[0] if len(gj_vals) >= 1 else None,
        'obs.gJ':            gj_vals[1] if len(gj_vals) >= 2 else None,
    }


# ---------------------------------------------------------------------------
# File-level parser
# ---------------------------------------------------------------------------

def parse_txt_file(filepath: str, parity_flag: int) -> list:
    """
    Parse one Kurucz text file (even or odd parity).

    Args:
        filepath:     Path to the .txt file.
        parity_flag:  0 for even parity, 1 for odd parity.

    Returns:
        List of raw-row dicts (columns of the intermediate xlsx).
    """
    ref = os.path.splitext(os.path.basename(filepath))[0]
    rows = []
    current_j = None

    with open(filepath, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()

    n_skipped_j = 0
    for line in lines[1:]:        # first line is the column header
        stripped = line.rstrip()
        if not stripped.strip():
            continue

        # J-value header — update carry-forward value and skip to next line
        m = _J_HEADER_RE.match(stripped)
        if m:
            j_str = m.group(1)
            if '/' in j_str:
                n, d = j_str.split('/')
                current_j = int(n) / int(d)
            else:
                current_j = float(j_str)
            continue

        parsed = _parse_kurucz_line(stripped)
        if parsed is None:
            n_skipped_j += 1
            continue

        if current_j is None:
            # Data row appeared before any J-header — should not happen
            n_skipped_j += 1
            continue

        parsed['J']           = current_j
        parsed['parity_flag'] = parity_flag
        parsed['Reference']   = ref
        rows.append(parsed)

    return rows


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the intermediate Kurucz DataFrame to the NIST-compatible feature schema.

    Key mappings
    ------------
    Level (cm-1)      ← OBS.LEVEL
    lande_g           ← obs.gJ (preferred) or calc.gJ (fallback)
    Configuration     ← kurucz_config_to_nist(Configuration_raw)
    Term              ← Term_raw; '*' appended for odd-parity rows
    parity            ← parity_flag (0=even, 1=odd)
    J, S_qn, L_qn    ← J carry-forward; parse_term(Term)
    orbital columns   ← fill_orbital_occupancies(Configuration, Z=27)
    Uncertainty       ← None  (not available in Kurucz data)
    leading_pct       ← None  (not available)
    term_label        ← ''    (Kurucz uses no letter-prefix convention)
    """
    # Pre-convert all config strings and collect the full orbital column set
    nist_configs = [kurucz_config_to_nist(r) for r in raw_df['Configuration_raw']]
    orbital_cols = collect_orbital_columns(nist_configs, _Z)
    print(f"  Orbital columns ({len(orbital_cols)}): "
          f"{orbital_cols[0]} … {orbital_cols[-1]}")

    parse_errors = 0
    skipped_no_orbitals = 0
    out_rows = []

    for i, row in raw_df.iterrows():
        nist_config = nist_configs[i]
        parity      = int(row['parity_flag'])
        term_raw    = str(row['Term_raw']).strip()

        # Append '*' to odd-parity terms so parse_term() sets parity=1 and the
        # Term column stays consistent with the NIST convention (e.g. '6F*').
        term_str = term_raw + '*' if parity == 1 else term_raw

        S_qn, L_qn, _, term_label = parse_term(term_str)

        # Orbital occupancies
        try:
            occupancies = fill_orbital_occupancies(nist_config, _Z)
        except ValueError as exc:
            print(f"  Warning (config parse): {exc}")
            occupancies = parse_explicit_orbitals(nist_config)
            parse_errors += 1

        # Skip rows whose config yields no recognised orbital segments
        if not occupancies:
            skipped_no_orbitals += 1
            continue

        # Lande g-factor: observed value preferred; fall back to calculated
        obs_gj  = row['obs.gJ']
        calc_gj = row['calc.gJ']
        lande_g_val = (obs_gj  if (obs_gj  is not None and pd.notna(obs_gj))
                       else (calc_gj if (calc_gj is not None and pd.notna(calc_gj))
                             else None))
        lande_str = str(lande_g_val) if lande_g_val is not None else ''

        out_row = {
            'Configuration':      nist_config,
            'Term':               term_str,
            'term_label':         term_label,
            'Prefix':             '',
            'Suffix':             '',
            'Lande':              lande_str,
            'lande_g':            lande_g_val,
            'leading_pct':        None,
            'Reference':          row['Reference'],
            'Z':                  _Z,
            'A':                  _A,
            'proton_number':      _Z,
            'neutron_number':     _A - _Z,
            'J':                  row['J'],
            'S_qn':               S_qn,
            'L_qn':               L_qn,
            'parity':             parity,
            'Level (cm-1)':       row['OBS.LEVEL'],
            'Uncertainty (cm-1)': None,
        }

        for orb in orbital_cols:
            out_row[orb] = occupancies.get(orb, 0)

        out_rows.append(out_row)

    if parse_errors:
        print(f"  {parse_errors} rows had config parse errors "
              f"(saved with partial orbital data)")
    if skipped_no_orbitals:
        print(f"  {skipped_no_orbitals} rows skipped — no valid orbitals in config")

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_kurucz(data_dir: str = 'data') -> pd.DataFrame:
    """
    Full pipeline: txt files → intermediate xlsx → feature CSV.

    Returns the feature DataFrame.
    """
    kurucz_dir = os.path.join(data_dir, 'kurucz')
    even_path  = os.path.join(kurucz_dir, 'Co-even-atom-AI.txt')
    odd_path   = os.path.join(kurucz_dir, 'Coi-odd-atom-AI.txt')

    for p in (even_path, odd_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Kurucz file not found: {p}")

    print(f"\nProcessing Kurucz Co data  (Z={_Z}, A={_A})")
    print(f"  Even parity: {even_path}")
    print(f"  Odd  parity: {odd_path}")

    # Step 1 — parse both txt files
    rows_even = parse_txt_file(even_path, parity_flag=0)
    rows_odd  = parse_txt_file(odd_path,  parity_flag=1)
    print(f"  Observed rows — even: {len(rows_even)}, odd: {len(rows_odd)}")

    raw_df = pd.DataFrame(rows_even + rows_odd)

    # Step 2 — save intermediate xlsx with original Kurucz columns
    xlsx_path = os.path.join(kurucz_dir, 'Co_kurucz_raw.xlsx')
    # Column order for readability
    col_order = ['J', 'parity_flag', 'Configuration_raw', 'Term_raw',
                 'OBS.LEVEL', 'EIGENVALUE', 'T-W', 'calc.gJ', 'obs.gJ', 'Reference']
    raw_df[col_order].to_excel(xlsx_path, index=False)
    print(f"  Intermediate xlsx : {xlsx_path}  ({len(raw_df)} rows)")

    # Step 3 — build NIST-compatible feature DataFrame
    print(f"\nBuilding feature DataFrame...")
    feature_df = build_feature_df(raw_df)
    print(f"  Feature rows      : {len(feature_df)}")

    out_path = os.path.join(data_dir, 'Co_features_kurucz.csv')
    feature_df.to_csv(out_path, index=False)
    print(f"  Feature CSV       : {out_path}")

    return feature_df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Kurucz Co I atomic energy level data'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Root data directory containing the kurucz/ subfolder (default: data)'
    )
    args = parser.parse_args()
    preprocess_kurucz(args.data_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
