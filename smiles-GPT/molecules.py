import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from typing import List, Tuple


def graph_to_molecule(graph: nx.Graph):
    """
    Convert a graph representation to a molecule.
    
    Args:
        graph (nx.Graph): The graph representation of the molecule.
    
    Returns:
        str: The SMILES string of the molecule.
    """
    # Create an RDKit molecule from the graph
    mol = Chem.RWMol()
    
    # Add atoms to the molecule
    atom_map = {}
    for node in graph.nodes():
        atom = Chem.Atom(node)
        atom_idx = mol.AddAtom(atom)
        atom_map[node] = atom_idx
    
    # Add bonds to the molecule
    for u, v, data in graph.edges(data=True):
        bond_type = data.get('bond_type', Chem.rdchem.BondType.SINGLE)
        mol.AddBond(atom_map[u], atom_map[v], bond_type)
    
    # Convert the molecule to a SMILES string
    smiles = Chem.MolToSmiles(mol)
    
    return smiles


def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a graph representation.
    
    Args:
        smiles (str): The SMILES string to convert.
    
    Returns:
        tuple: A tuple containing the graph representation of the molecule.
    """
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    
    # Get bond features
    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append(bond.GetBondTypeAsDouble())
    
    # Create adjacency matrix
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    
    return np.array(atom_features), np.array(bond_features), adj_matrix
