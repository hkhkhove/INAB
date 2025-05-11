import os
import warnings
import time
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser

warnings.filterwarnings("ignore")

from .empirical_feats import run as extract_empirical_feats
from .ESM2 import run as extract_ESM2_feats
from .SaProt import run as extract_SaProt_feats
from .GearNet import run as extract_GearNet_feats


def get_pssm(pssm_file):
    with open(pssm_file, "r") as f:
        text = f.readlines()
    pssm = []
    for line in text[3:]:
        if line == "\n":
            break
        else:
            res_pssm = np.array(list(map(int, line.split()[2:22]))).reshape(1, -1)
            assert res_pssm.shape[1] == 20, f"{pssm_file} is incomplete."
            pssm.append(res_pssm)
    pssm = np.concatenate(pssm, axis=0)
    pssm = 1 / (1 + np.exp(-pssm))

    return pssm


def get_hhm(hhm_file):
    with open(hhm_file, "r") as f:
        text = f.readlines()
    hhm_begin_line = 0
    hhm_end_line = 0
    for i in range(len(text)):
        if "#" in text[i]:
            hhm_begin_line = i + 5
        elif "//" in text[i]:
            hhm_end_line = i
    hhm = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])

    axis_x = 0
    for i in range(hhm_begin_line, hhm_end_line, 3):
        line1 = text[i].split()[2:-1]
        line2 = text[i + 1].split()
        axis_y = 0
        for j in line1:
            if j == "*":
                hhm[axis_x][axis_y] = 9999 / 10000.0
            else:
                hhm[axis_x][axis_y] = float(j) / 10000.0
            axis_y += 1
        for j in line2:
            if j == "*":
                hhm[axis_x][axis_y] = 9999 / 10000.0
            else:
                hhm[axis_x][axis_y] = float(j) / 10000.0
            axis_y += 1
        axis_x += 1
    hhm = (hhm - np.min(hhm)) / (np.max(hhm) - np.min(hhm))

    return hhm


def get_dssp(dssp_file, res_id_list):
    maxASA = {
        "G": 188,
        "A": 198,
        "V": 220,
        "I": 233,
        "L": 304,
        "F": 272,
        "P": 203,
        "M": 262,
        "W": 317,
        "C": 201,
        "S": 234,
        "T": 215,
        "N": 254,
        "Q": 259,
        "Y": 304,
        "H": 258,
        "D": 236,
        "E": 262,
        "K": 317,
        "R": 319,
    }
    map_ss_8 = {
        " ": [1, 0, 0, 0, 0, 0, 0, 0],
        "S": [0, 1, 0, 0, 0, 0, 0, 0],
        "T": [0, 0, 1, 0, 0, 0, 0, 0],
        "H": [0, 0, 0, 1, 0, 0, 0, 0],
        "G": [0, 0, 0, 0, 1, 0, 0, 0],
        "I": [0, 0, 0, 0, 0, 1, 0, 0],
        "E": [0, 0, 0, 0, 0, 0, 1, 0],
        "B": [0, 0, 0, 0, 0, 0, 0, 1],
    }

    with open(dssp_file, "r") as f:
        text = f.readlines()

    start_line = 0
    for i in range(0, len(text)):
        if text[i].startswith("  #  RESIDUE AA STRUCTURE"):
            start_line = i + 1
            break

    norss = {}
    for i in range(start_line, len(text)):
        line = text[i]
        if line[13] not in maxASA.keys() or line[9] == " ":
            continue
        res_id = float(line[5:10])
        res_dssp = np.zeros([14])
        res_dssp[:8] = map_ss_8[line[16]]  # SS
        res_dssp[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
        res_dssp[9] = (float(line[85:91]) + 1) / 2
        res_dssp[10] = min(1, float(line[91:97]) / 180)
        res_dssp[11] = min(1, (float(line[97:103]) + 180) / 360)
        res_dssp[12] = min(1, (float(line[103:109]) + 180) / 360)
        res_dssp[13] = min(1, (float(line[109:115]) + 180) / 360)
        norss[res_id] = res_dssp.reshape((1, -1))

    dssp_ = []
    for res_id_i in res_id_list:
        if res_id_i in norss.keys():
            dssp_.append(norss[res_id_i])
        else:
            dssp_.append(np.zeros(list(norss.values())[0].shape))
    dssp_ = np.concatenate(dssp_, axis=0)

    return dssp_


def get_atom_feats(pdb_file):
    A = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 3, 0],
    }
    V = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 1, 0],
        "CG1": [0, 3, 0],
        "CG2": [0, 3, 0],
    }
    F = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 0, 1],
        "CD1": [0, 1, 1],
        "CD2": [0, 1, 1],
        "CE1": [0, 1, 1],
        "CE2": [0, 1, 1],
        "CZ": [0, 1, 1],
    }
    P = {
        "N": [0, 0, 1],
        "CA": [0, 1, 1],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 1],
        "CG": [0, 2, 1],
        "CD": [0, 2, 1],
    }
    L = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 1, 0],
        "CD1": [0, 3, 0],
        "CD2": [0, 3, 0],
    }
    I = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 1, 0],
        "CG1": [0, 2, 0],
        "CG2": [0, 3, 0],
        "CD1": [0, 3, 0],
    }
    R = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 2, 0],
        "CD": [0, 2, 0],
        "NE": [0, 1, 0],
        "CZ": [1, 0, 0],
        "NH1": [0, 2, 0],
        "NH2": [0, 2, 0],
    }
    D = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [-1, 0, 0],
        "OD1": [-1, 0, 0],
        "OD2": [-1, 0, 0],
    }
    E = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 2, 0],
        "CD": [-1, 0, 0],
        "OE1": [-1, 0, 0],
        "OE2": [-1, 0, 0],
    }
    S = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "OG": [0, 1, 0],
    }
    T = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 1, 0],
        "OG1": [0, 1, 0],
        "CG2": [0, 3, 0],
    }
    C = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "SG": [-1, 1, 0],
    }
    N = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 0, 0],
        "OD1": [0, 0, 0],
        "ND2": [0, 2, 0],
    }
    Q = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 2, 0],
        "CD": [0, 0, 0],
        "OE1": [0, 0, 0],
        "NE2": [0, 2, 0],
    }
    H = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 0, 1],
        "ND1": [-1, 1, 1],
        "CD2": [0, 1, 1],
        "CE1": [0, 1, 1],
        "NE2": [-1, 1, 1],
    }
    K = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 2, 0],
        "CD": [0, 2, 0],
        "CE": [0, 2, 0],
        "NZ": [0, 3, 1],
    }
    Y = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 0, 1],
        "CD1": [0, 1, 1],
        "CD2": [0, 1, 1],
        "CE1": [0, 1, 1],
        "CE2": [0, 1, 1],
        "CZ": [0, 0, 1],
        "OH": [-1, 1, 0],
    }
    M = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 2, 0],
        "SD": [0, 0, 0],
        "CE": [0, 3, 0],
    }
    W = {
        "N": [0, 1, 0],
        "CA": [0, 1, 0],
        "C": [0, 0, 0],
        "O": [0, 0, 0],
        "CB": [0, 2, 0],
        "CG": [0, 0, 1],
        "CD1": [0, 1, 1],
        "CD2": [0, 0, 1],
        "NE1": [0, 1, 1],
        "CE2": [0, 0, 1],
        "CE3": [0, 1, 1],
        "CZ2": [0, 1, 1],
        "CZ3": [0, 1, 1],
        "CH2": [0, 1, 1],
    }
    G = {"N": [0, 1, 0], "CA": [0, 2, 0], "C": [0, 0, 0], "O": [0, 0, 0]}

    atom_fea_dict = {
        "A": A,
        "V": V,
        "F": F,
        "P": P,
        "L": L,
        "I": I,
        "R": R,
        "D": D,
        "E": E,
        "S": S,
        "T": T,
        "C": C,
        "N": N,
        "Q": Q,
        "H": H,
        "K": K,
        "Y": Y,
        "M": M,
        "W": W,
        "G": G,
    }
    for atom_fea in atom_fea_dict.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    atom_count = -1
    res_count = -1
    pdb_file = open(pdb_file, "r")
    rows = []
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {
        "H": 1,
        "C": 12,
        "O": 16,
        "N": 14,
        "S": 32,
        "FE": 56,
        "P": 31,
        "BR": 80,
        "F": 19,
        "CO": 59,
        "V": 51,
        "I": 127,
        "CL": 35.5,
        "CA": 40,
        "B": 10.8,
        "ZN": 65.5,
        "MG": 24.3,
        "NA": 23,
        "HG": 200.6,
        "MN": 55,
        "K": 39.1,
        "AP": 31,
        "AC": 227,
        "AL": 27,
        "W": 183.9,
        "SE": 79,
        "NI": 58.7,
    }
    AA_dic = {
        "GLY": "G",
        "ALA": "A",
        "VAL": "V",
        "LEU": "L",
        "ILE": "I",
        "PHE": "F",
        "TRP": "W",
        "TYR": "Y",
        "ASP": "D",
        "ASN": "N",
        "GLU": "E",
        "LYS": "K",
        "GLN": "Q",
        "MET": "M",
        "SER": "S",
        "THR": "T",
        "CYS": "C",
        "PRO": "P",
        "HIS": "H",
        "ARG": "R",
    }

    while True:
        line = pdb_file.readline()
        if line.startswith("ATOM"):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count += 1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ["N", "CA", "C", "O", "H"]:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = AA_dic[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5, 0.5, 0.5]

            try:
                bfactor = float(line[60:66])
            except ValueError:
                bfactor = 0.5

            row = {
                "atom_type": atom_type,
                "res_id": int(line[22:26]),
                "B_factor": bfactor,
                "mass": Relative_atomic_mass[atom_type],
                "is_sidechain": is_sidechain,
                "charge": atom_fea[0],
                "num_H": atom_fea[1],
                "ring": atom_fea[2],
            }

            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))

            rows.append(row)

        if line.startswith("TER"):
            break

    pdb_af = pd.DataFrame(rows)

    atom_vander_dict = {
        "C": 1.7,
        "O": 1.52,
        "N": 1.55,
        "S": 1.85,
        "H": 1.2,
        "D": 1.2,
        "SE": 1.9,
        "P": 1.8,
        "FE": 2.23,
        "BR": 1.95,
        "F": 1.47,
        "CO": 2.23,
        "V": 2.29,
        "I": 1.98,
        "CL": 1.75,
        "CA": 2.81,
        "B": 2.13,
        "ZN": 2.29,
        "MG": 1.73,
        "NA": 2.27,
        "HG": 1.7,
        "MN": 2.24,
        "K": 2.75,
        "AC": 3.08,
        "AL": 2.51,
        "W": 2.39,
        "NI": 2.22,
    }
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    pdb_af = pdb_af[pdb_af["atom_type"] != "H"]

    mass = np.array(pdb_af["mass"].tolist()).reshape(-1, 1)
    mass = mass / 32
    B_factor = np.array(pdb_af["B_factor"].tolist()).reshape(-1, 1)
    if (max(B_factor) - min(B_factor)) == 0:
        B_factor = np.zeros(B_factor.shape) + 0.5
    else:
        B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
    is_sidechain = np.array(pdb_af["is_sidechain"].tolist()).reshape(-1, 1)
    charge = np.array(pdb_af["charge"].tolist()).reshape(-1, 1)
    num_H = np.array(pdb_af["num_H"].tolist()).reshape(-1, 1)
    ring = np.array(pdb_af["ring"].tolist()).reshape(-1, 1)
    atom_type = pdb_af["atom_type"].tolist()
    atom_vander = np.zeros((len(atom_type), 1))
    for i, type in enumerate(atom_type):
        try:
            atom_vander[i] = atom_vander_dict[type]
        except:
            atom_vander[i] = atom_vander_dict["C"]

    atom_feats = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
    atom_feats = np.concatenate(atom_feats, axis=1)
    res_atom_feats = []
    atom_begin = 0
    for i, res_id in enumerate(res_id_list):
        res_atom_df = pdb_af[pdb_af["res_id"] == res_id]
        atom_num = len(res_atom_df)
        res_atom_feats_i = atom_feats[atom_begin : atom_begin + atom_num]
        res_atom_feats_i = np.average(res_atom_feats_i, axis=0).reshape(1, -1)
        res_atom_feats.append(res_atom_feats_i)
        atom_begin += atom_num

    res_atom_feats = np.concatenate(res_atom_feats, axis=0)

    return res_atom_feats, res_id_list


def get_edges(pdb_file, cutoff=12):
    prot_name = os.path.basename(pdb_file).split(".")[0]
    chain_id = prot_name.split("_")[1]
    parser = PDBParser()
    structure = parser.get_structure(prot_name, pdb_file)
    edges = []
    edge_attr = []
    left = []
    right = []
    coords = []
    for residue in structure[0][chain_id]:
        if residue.id[0] == " ":
            atom_coords = [atom.coord for atom in residue]
            coord = np.mean(atom_coords, axis=0)
            coords.append(coord)
    for i, coord_i in enumerate(coords):
        for j, coord_j in enumerate(coords):
            if j <= i:
                continue
            diff_vector = coord_i - coord_j
            d = np.sqrt(np.sum(diff_vector * diff_vector))
            if d is not None and d <= cutoff:
                left.append(i)
                right.append(j)
                left.append(j)
                right.append(i)
                weight = np.log(abs(i - j)) / d
                edge_attr.append([weight])
                edge_attr.append([weight])

    edges.append(left)
    edges.append(right)
    return np.array(edges), np.array(edge_attr), np.array(coords)


def normalize_np_array(np_array, p=2, axis=0):
    # L2 norm
    norm = np.linalg.norm(np_array, ord=p, axis=axis, keepdims=True)
    normalized_array = np_array / norm
    return normalized_array


def combine(pdb_files):
    success_pdbs = []
    for pdb_file in tqdm(pdb_files, desc="5.Combining features"):
        if os.path.isfile(pdb_file.replace(".pdb", "_input.pkl")):
            success_pdbs.append(pdb_file)
            continue
        try:
            prot_name, _ = os.path.splitext(os.path.basename(pdb_file))
            hhm = get_hhm(pdb_file.replace(".pdb", ".hhm"))
            pssm = get_pssm(pdb_file.replace(".pdb", ".pssm"))
            atom_feats, res_id_list = get_atom_feats(pdb_file)
            dssp = get_dssp(pdb_file.replace(".pdb", ".dssp"), res_id_list)

            assert (
                hhm.shape[0] == pssm.shape[0] == dssp.shape[0] == atom_feats.shape[0]
            ), (
                f"feature dimension mismatch, hhm:{hhm.shape[0]},pssm:{pssm.shape[0]},dssp:{dssp.shape[0]},atom_feats:{atom_feats.shape[0]}"
            )

            esm2_rep = np.load(pdb_file.replace(".pdb", "_esm2.npy"))
            gearnet_rep = np.load(pdb_file.replace(".pdb", "_gearnet.npy"))
            saprot_rep = np.load(pdb_file.replace(".pdb", "_saprot.npy"))

            esm2_rep = normalize_np_array(esm2_rep)
            gearnet_rep = normalize_np_array(gearnet_rep)
            saprot_rep = normalize_np_array(saprot_rep)

            edges, edge_attr, coords = get_edges(pdb_file)

            assert (
                esm2_rep.shape[0]
                == gearnet_rep.shape[0]
                == saprot_rep.shape[0]
                == len(coords)
                == hhm.shape[0]
            ), (
                f"feature dimension mismatch, esm2_rep:{esm2_rep.shape[0]},gearnet_rep:{gearnet_rep.shape[0]},saprot_rep:{saprot_rep.shape[0]},coords:{len(coords)},hhm:{hhm.shape[0]}"
            )

            node_feats = np.concatenate(
                [hhm, pssm, dssp, atom_feats, esm2_rep, gearnet_rep, saprot_rep], axis=1
            )

            with open(pdb_file.replace(".pdb", "_input.pkl"), "wb") as f:
                pickle.dump((prot_name, node_feats, coords, edges, edge_attr), f)

            success_pdbs.append(pdb_file)
        except Exception as e:
            with open("error.log", "a") as f:
                f.write(f">{prot_name}: {e}\n")

    return success_pdbs


def run(pdb_files, path):
    path["log_file"] = path.get("log_file") or os.path.join(
        path["base_dir"],
        "log",
        f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt",
    )
    extract_empirical_feats(pdb_files, path)
    extract_ESM2_feats(pdb_files, path)
    extract_GearNet_feats(pdb_files, path)
    extract_SaProt_feats(pdb_files, path)

    return combine(pdb_files)


if __name__ == "__main__":
    """
    Run as a module to extract features for training:
    > python -m feature.extract --pdb_dir PDB_DIR   
    To specify the GPU device:
    > CUDA_VISIBLE_DEVICES=INDEX python -m feature.extract --pdb_dir PDB_DIR
    """
    import argparse

    argparser = argparse.ArgumentParser(description="Extract features from PDB files")
    argparser.add_argument(
        "--pdb_dir", type=str, required=True, help="Directory containing PDB files"
    )
    args = argparser.parse_args()

    pdb_files = [
        os.path.join(args.pdb_dir, e)
        for e in os.listdir(args.pdb_dir)
        if e.endswith(".pdb") or e.endswith(".cif")
    ]
    pdb_files = [
        e for e in pdb_files if not os.path.isfile(e.replace(".pdb", "_input.pkl"))
    ]

    import config

    success_pdbs = run(pdb_files, config.path)

    print(
        f"Successfully extracted features from {len(success_pdbs)}/{len(pdb_files)} PDB files."
    )
