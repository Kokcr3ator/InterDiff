from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def get_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def compute_qed(smiles, threshold=0.7):
    mol = get_mol(smiles)
    if not mol:
        return 0
    return 1 if QED.qed(mol) >= threshold else 0

def compute_logp(smiles, min_val=1.0, max_val=3.5):
    mol = get_mol(smiles)
    if not mol:
        return 0
    logp = Crippen.MolLogP(mol)
    return 1 if min_val <= logp <= max_val else 0

def compute_molecular_weight(smiles, min_val=200, max_val=450):
    mol = get_mol(smiles)
    if not mol:
        return 0
    mw = Descriptors.MolWt(mol)
    return 1 if min_val <= mw <= max_val else 0

def compute_tpsa(smiles, max_val=90.0):
    mol = get_mol(smiles)
    if not mol:
        return 0
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return 1 if tpsa <= max_val else 0

def compute_sa(smiles, max_val=4.0):
    mol = get_mol(smiles)
    if not mol:
        return 0
    sa = sascorer.calculateScore(mol)
    return 1 if sa <= max_val else 0