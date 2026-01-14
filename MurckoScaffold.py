import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import math
 
# ======Setting======
INPUT_CSV = "smiles.csv"   # columns: smiles (required), compound_id (optional)
SMILES_COL = "smiles"
ID_COL = "compound_id"     
 
# ======
df = pd.read_csv(INPUT_CSV)
if ID_COL not in df.columns:
    df[ID_COL] = [f"CMPD_{i+1:04d}" for i in range(len(df))]
 
# ====== SMILES -> Mol ======
mols = []
valid = []
for smi in df[SMILES_COL].astype(str):
    m = Chem.MolFromSmiles(smi)
    mols.append(m)
    valid.append(m is not None)
 
df["is_valid_smiles"] = valid
df_valid = df[df["is_valid_smiles"]].copy()
df_invalid = df[~df["is_valid_smiles"]].copy()
 
# ====== Murcko scaffold ======
def murcko_scaffold_smiles(mol):
    # Murcko scaffold mol -> canonical SMILES
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return None
    return Chem.MolToSmiles(scaf, isomericSmiles=False)
 
scaf_list = []
for mol in [Chem.MolFromSmiles(s) for s in df_valid[SMILES_COL].astype(str)]:
    scaf_list.append(murcko_scaffold_smiles(mol))
 
df_valid["murcko_scaffold"] = scaf_list
 
# scaffold 
df_scaf = df_valid[df_valid["murcko_scaffold"].notna()].copy()
df_no_scaf = df_valid[df_valid["murcko_scaffold"].isna()].copy()
 
# ============
scaf_counts = Counter(df_scaf["murcko_scaffold"])
n_compounds = len(df_scaf)
n_unique_scaf = len(scaf_counts)
 
# top scaffold 
def top_share(k):
    topk = [c for _, c in scaf_counts.most_common(k)]
    return sum(topk) / n_compounds if n_compounds else 0.0
 
top10 = top_share(10)
top50 = top_share(50)
 
# scaffold size distribution
size_counts = Counter(scaf_counts.values())  # key: size, value: how many scaffolds have that size
size_df = pd.DataFrame({"compounds_per_scaffold": list(size_counts.keys()),
                        "num_scaffolds": list(size_counts.values())}).sort_values("compounds_per_scaffold")
 
# Shannon entropy
p = [c / n_compounds for c in scaf_counts.values()]
shannon = -sum(pi * math.log(pi) for pi in p if pi > 0)
 
# ====== Output ======
print("=== Murcko scaffold summary ===")
print(f"Valid SMILES: {len(df_valid)} / {len(df)}")
print(f"Invalid SMILES: {len(df_invalid)}")
print(f"Compounds with scaffold: {len(df_scaf)}")
print(f"Compounds without scaffold (e.g., acyclic): {len(df_no_scaf)}")
print(f"Unique Murcko scaffolds: {n_unique_scaf}")
print(f"Top-10 scaffold share: {top10:.2%}")
print(f"Top-50 scaffold share: {top50:.2%}")
print(f"Shannon entropy (scaffold freq.): {shannon:.3f}")
 
# Supplementary table 
df_scaf_out = df_scaf[[ID_COL, SMILES_COL, "murcko_scaffold"]].copy()
df_scaf_out.to_csv("murcko_scaffold_per_compound.csv", index=False)
 
# Distribution）
size_df.to_csv("murcko_scaffold_size_distribution.csv", index=False)
 
# Top scaffold 
top_scaf_df = pd.DataFrame(scaf_counts.most_common(50), columns=["murcko_scaffold", "num_compounds"])
top_scaf_df.to_csv("top50_murcko_scaffolds.csv", index=False)
