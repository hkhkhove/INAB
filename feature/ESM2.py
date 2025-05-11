import os
import warnings
import numpy as np
import torch
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
import esm

from .oom_wrapper import oom_wrapper

warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)


def get_seq(pdb_file):
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
    prot_name, ext = os.path.splitext(os.path.basename(pdb_file))
    if ext == ".pdb":
        parser = PDBParser()
    elif ext == ".cif":
        parser = MMCIFParser()
    structure = parser.get_structure("tmp", pdb_file)
    model = structure[0]
    seq = ""
    for residue in model.get_residues():
        if residue.id[0] == " ":
            seq += AA_dic[residue.resname]
    save_path = f"{os.path.join(os.path.dirname(pdb_file), prot_name + '.fasta')}"

    with open(save_path, "w") as f:
        f.write(f">{prot_name}\n{seq}\n")


@oom_wrapper
def inference(model, input, save_path):
    with torch.no_grad():
        results = model(input, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]
    prot_rep = token_representations[:, 1:-1, :].squeeze()

    rep_numpy = prot_rep.detach().cpu().numpy()
    np.save(save_path, rep_numpy)


def run(pdb_files, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    for pdb_file in tqdm(pdb_files, desc="2.ESM-2"):
        prot_name, ext = os.path.splitext(os.path.basename(pdb_file))
        save_path = pdb_file.replace(ext, "_esm2.npy")
        if os.path.isfile(save_path):
            continue

        fasta_file = pdb_file.replace(ext, ".fasta")
        if not os.path.isfile(fasta_file):
            get_seq(pdb_file)

        with open(fasta_file) as f:
            f.readline()
            seq = f.readline().strip()

        data = [(prot_name, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens_device = batch_tokens.to(device)

        oom = inference(model, batch_tokens_device, save_path)
        if oom:
            with open(path["log_file"], "a") as f:
                f.write(
                    f">{prot_name}(length={len(seq)}): ESM2 feature processed using CPU due to CUDA OOM.\n"
                )


if __name__ == "__main__":
    """
    Run as a module to extract ESM2 representations from PDB files:
    > python -m feature.ESM2 --pdb_dir PDB_DIR   
    To specify the GPU device:
    > CUDA_VISIBLE_DEVICES=INDEX python -m feature.extract --pdb_dir PDB_DIR
    """
    import argparse

    argparser = argparse.ArgumentParser(
        description="Extract ESM2 representations from PDB files"
    )
    argparser.add_argument(
        "--pdb_dir", type=str, required=True, help="Directory containing PDB files"
    )
    args = argparser.parse_args()

    pdb_files = [
        os.path.join(args.pdb_dir, e)
        for e in os.listdir(args.pdb_dir)
        if e.endswith(".pdb") or e.endswith(".cif")
    ]

    import config
    import time

    path = config.path

    path["log_file"] = path.get("log_file") or os.path.join(
        path["base_dir"],
        "log",
        f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}_esm2.txt",
    )
    run(pdb_files, config.path)
