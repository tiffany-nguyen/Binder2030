import argparse
import time
import requests
import pandas as pd
from urllib.parse import quote

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SLEEP = 0.15
RETRIES = 3
TIMEOUT = 40

def _get_json(url: str, retries: int = RETRIES, timeout: int = TIMEOUT):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(SLEEP * (i + 1))
    return {"error": last_err or "unknown error"}

def smiles_to_cid_via_name(smiles: str):
    url = f"{PUG}/compound/smiles/{quote(smiles)}/cids/JSON"
    data = _get_json(url)
    if isinstance(data, dict) and "IdentifierList" in data and "CID" in data["IdentifierList"]:
        return data["IdentifierList"]["CID"][0]
    return None

def fetch_assaysummary(cid: int):
    url = f"{PUG}/compound/cid/{cid}/assaysummary/JSON"
    return _get_json(url)

def normalize_assaysummary_rows(table_obj):
    """
    Handles both PubChem shapes:
    A) {"Table":{"Columns":{"Column":[...]}, "Row":[{"Cell":[...]}, ...]}}
    B) {"Table":{"Row":[{key:value,...}, ...]}}
    Returns: list of dict rows with consistent keys.
    """
    if not isinstance(table_obj, dict):
        return []

    # Case A: Columns + Row/Cell
    cols = None
    try:
        cols = table_obj.get("Columns", {}).get("Column")
        rows = table_obj.get("Row", [])
        if cols and rows and isinstance(rows, list) and "Cell" in rows[0]:
            out = []
            for r in rows:
                cells = r.get("Cell", [])
                d = {}
                for i, c in enumerate(cells):
                    key = cols[i] if i < len(cols) else f"col_{i}"
                    d[key] = c
                out.append(d)
            return out
    except Exception:
        pass

    # Case B: already a list of dicts
    try:
        rows = table_obj.get("Row", [])
        if rows and isinstance(rows, list) and isinstance(rows[0], dict):
            return rows
    except Exception:
        pass

    return []

def main(input_csv: str, output_csv: str):
    smiles_df = pd.read_csv(input_csv, header=None, names=["Our SMILES"])
    smiles_df["Our SMILES"] = smiles_df["Our SMILES"].astype(str).str.strip()

    out_rows = []
    all_keys = set()

    for smi in smiles_df["Our SMILES"]:
        if not smi:
            out_rows.append({"Our SMILES": smi, "CID": "N/A", "AID": "N/A", "_note": "Empty input"})
            continue

        cid = smiles_to_cid_via_name(smi)
        time.sleep(SLEEP)
        print(f"SMILES: {smi} -> CID: {cid}")

        if cid is None:
            out_rows.append({"Our SMILES": smi, "CID": "N/A", "AID": "N/A", "_note": "No CID found"})
            continue

        data = fetch_assaysummary(cid)
        time.sleep(SLEEP)

        table = (data or {}).get("Table")
        norm_rows = normalize_assaysummary_rows(table) if table else []

        if not norm_rows:
            out_rows.append({"Our SMILES": smi, "CID": cid, "AID": "N/A", "_note": "No assay summary rows"})
            continue

        # map each assaysummary row to output; include all keys present
        for r in norm_rows:
            row = {"Our SMILES": smi, "CID": cid}
            # Standardize a few common aliases (if present only)
            # If the JSON used "Activity Outcome", "Assay Name", etc. we keep them as-is
            for k, v in r.items():
                row[k] = v
                all_keys.add(k)
            out_rows.append(row)

    # Build columns: Our SMILES, CID, AID (if present), then other keys sorted; add _note if present
    cols = ["Our SMILES", "CID"]
    if "AID" in all_keys:
        cols.append("AID")
        all_keys.discard("AID")
    cols += sorted(all_keys)
    has_note = any("_note" in r for r in out_rows)
    if has_note:
        cols.append("_note")

    # Fill missing with N/A and write
    for r in out_rows:
        for c in cols:
            if c not in r:
                r[c] = "N/A"

    out_df = pd.DataFrame(out_rows, columns=cols)
    out_df.to_csv(output_csv, index=False)
    print(f"Done. Wrote {len(out_df)} rows to {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SMILES -> CID -> assaysummary rows (robust parser).")
    ap.add_argument("input_csv", help="Input CSV with one SMILES per line, no header.")
    ap.add_argument("-o", "--output_csv", default="smiles_assaysummary_rows.csv")
    args = ap.parse_args()
    main(args.input_csv, args.output_csv)
