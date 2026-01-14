import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import umap
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load three datasets
# ---------------------------------------
gpcr = pd.read_csv("GPCR_compounds.csv")   # must contain 'smiles'
slc  = pd.read_csv("SLC_compounds.csv")
ion  = pd.read_csv("ION_compounds.csv")

gpcr["dataset"] = "GPCR"
slc["dataset"]  = "SLC"
ion["dataset"]  = "ION"

df = pd.concat([gpcr, slc, ion], ignore_index=True)

# ---------------------------------------
# 2. Convert SMILES → RDKit Mol
# ---------------------------------------
def smiles_to_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None

df["mol"] = df["smiles"].apply(smiles_to_mol)
df_valid = df[df["mol"].notna()].reset_index(drop=True)
mols = df_valid["mol"].tolist()

print(f"Total rows: {len(df)} | Valid molecules: {len(mols)}")

# ---------------------------------------
# 3. Morgan fingerprints (ECFP4-like)
# ---------------------------------------
n_bits = 2048
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)

fps = []
for m in mols:
    bitvect = gen.GetFingerprint(m)
    arr = np.zeros((n_bits,), dtype=np.int8)
    # Convert ExplicitBitVect to numpy array
    Chem.DataStructs.ConvertToNumpyArray(bitvect, arr)
    fps.append(arr)

X = np.asarray(fps)   # shape: (n_mols, 2048)

# ---------------------------------------
# 4. UMAP embedding (joint for all three)
# ---------------------------------------
reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    metric="jaccard",    # good for binary fingerprints
    random_state=42
)

embedding = reducer.fit_transform(X)
df_valid["UMAP_1"] = embedding[:, 0]
df_valid["UMAP_2"] = embedding[:, 1]

# ---------------------------------------
# 5. Plot overlay (GPCR vs SLC vs ION)
# ---------------------------------------
plt.figure(figsize=(7, 6))

palette = {
    "GPCR": "#1b9e77",  # green
    "SLC":  "#d95f02",  # orange
    "ION":  "#7570b3",  # purple
}

for label, group in df_valid.groupby("dataset"):
    plt.scatter(
        group["UMAP_1"],
        group["UMAP_2"],
        s=10,
        alpha=0.6,
        label=label,
        c=palette[label]
    )

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("Combined chemical space of GPCR, SLC, and Ion Channel libraries")
plt.legend(title="Dataset", frameon=True)
plt.tight_layout()
plt.show()

# ---------------------------------------
# 6. Save coordinates for later use
# ---------------------------------------
df_valid.drop(columns=["mol"]).to_csv(
    "GPCR_SLC_ION_UMAP_coordinates.csv",
    index=False
)
print("Saved UMAP coordinates to 'GPCR_SLC_ION_UMAP_coordinates.csv'")
