import os
import subprocess
import multiprocessing as mp
import warnings
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

warnings.filterwarnings("ignore")

BLAST = ""
HHBLITS = ""
PSSMDB = ""
HHMDB = ""
DSSP = ""
log_file = ""


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
    else:
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


def run_BLAST(pdb_file):
    prot_name = os.path.splitext(os.path.basename(pdb_file))[0]
    outfmt_type = 5
    num_iter = 3
    evalue_threshold = 0.001
    fasta_file = os.path.join(os.path.dirname(pdb_file), prot_name + ".fasta")
    if not os.path.isfile(fasta_file):
        get_seq(pdb_file)
    xml_file = os.path.join(os.path.dirname(pdb_file), prot_name + ".xml")
    pssm_file = os.path.join(os.path.dirname(pdb_file), prot_name + ".pssm")
    if os.path.isfile(pssm_file):
        pass
    else:
        cmd = [
            BLAST,
            "-query",
            fasta_file,
            "-db",
            PSSMDB,
            "-out",
            xml_file,
            "-evalue",
            str(evalue_threshold),
            "-num_iterations",
            str(num_iter),
            "-outfmt",
            str(outfmt_type),
            "-out_ascii_pssm",
            pssm_file,  # Write the pssm file
            "-num_threads",
            "8",
        ]
        subprocess.run(cmd)


def run_HHblits(pdb_file):
    prot_name = os.path.splitext(os.path.basename(pdb_file))[0]
    fasta_file = os.path.join(os.path.dirname(pdb_file), prot_name + ".fasta")
    if not os.path.isfile(fasta_file):
        get_seq(pdb_file)
    hhm_file = os.path.join(os.path.dirname(pdb_file), prot_name + ".hhm")
    if os.path.isfile(hhm_file):
        pass
    else:
        cmd = [
            HHBLITS,
            "-i",
            fasta_file,
            "-d",
            HHMDB,
            "-ohhm",
            hhm_file,
            "-cpu",
            "8",
            "-v",
            "0",
        ]
        subprocess.run(cmd)


def run_dssp(pdb_file):
    prot_name = os.path.splitext(os.path.basename(pdb_file))[0]
    dssp_file = os.path.join(os.path.dirname(pdb_file), prot_name + ".dssp")
    if os.path.isfile(dssp_file):
        pass
    else:
        cmd = [DSSP, "-i", pdb_file, "-o", dssp_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            with open(log_file, "a") as f:
                f.write(f">{prot_name}: DSSP failed\n")


def parallel(num_processes, pdb_files, func):
    with mp.Pool(num_processes) as pool:
        results = pool.imap_unordered(func, pdb_files)
        pbar = tqdm(
            total=len(pdb_files),
            desc=f"1.Empirical features({func.__name__})",
            leave=False,
        )
        for _ in results:
            pbar.update()


def run(pdb_files, path):
    global BLAST, HHBLITS, PSSMDB, HHMDB, DSSP, log_file
    BLAST = path["blast"]
    HHBLITS = path["hhblits"]
    PSSMDB = path["pssmdb"]
    HHMDB = path["hhmdb"]
    DSSP = path["dssp"]
    log_file = path["log_file"]

    for pdb_file in tqdm(pdb_files, desc="1.Empirical features"):
        get_seq(pdb_file)
        run_dssp(pdb_file)
        # run_BLAST(pdb_file)
        # run_HHblits(pdb_file)

    parallel(8, pdb_files, run_BLAST)
    parallel(8, pdb_files, run_HHblits)


if __name__ == "__main__":
    """
    Run as a module to extract empirical features:
    > python -m feature.empirical_feats --pdb_dir PDB_DIR   
    """
    import argparse

    argparser = argparse.ArgumentParser(
        description="Extract empirical features from PDB files"
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
        f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}_empfeats.txt",
    )

    run(pdb_files, config.path)
