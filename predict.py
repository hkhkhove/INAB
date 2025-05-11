import pickle
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import tyro
from tqdm import tqdm

import config
from model.INAB import INAB

torch.set_grad_enabled(False)


@dataclass
class Args:
    gpu: str = "3"
    model_path: str = "model_parameters/INAB_DNA.pth"
    input_dir: Optional[str] = None
    """
    A directory that includes pdb files.
    Features extracted from the pdb files will be saved in this directory.
    Output files will also be saved in this directory.
    """
    pdb_path: Optional[str] = None
    """Path to the pdb file to be predicted"""
    prot_names: Optional[str] = None
    """
    A text file that includes the names of the proteins to be predicted.
    The names should be separated by new lines.
    If this argument is not specified, all proteins in the input_dir will be predicted.
    If pdb_path is specified, this argument is ignored.
    """


def predict(model, device, input_data):
    prot_name, node_feats, coords, edges, edge_attr = input_data

    coords = torch.from_numpy(coords).to(device).float()
    edges = torch.from_numpy(edges).to(device).long()
    edge_attr = torch.from_numpy(edge_attr).to(device).float()
    node_feats = torch.from_numpy(node_feats).to(device).float()

    output = model(node_feats, coords, edges, edge_attr)

    output = output.squeeze().cpu().tolist()

    return output


if "__main__" == __name__:
    args = tyro.cli(Args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config.path["log_file"] = os.path.join(
        config.path["base_dir"],
        "log",
        f"predict_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt",
    )
    # Import after setting CUDA_VISIBLE_DEVICES, otherwise it will not work
    from feature.extract import run as extract_feats

    if args.input_dir is None == args.pdb_path is None:
        raise ValueError("Exactly one of --pdb_path or --input_dir must be specified.")

    if args.input_dir is not None:
        pdbs = [
            os.path.join(args.input_dir, e)
            for e in os.listdir(args.input_dir)
            if e.endswith(".pdb") or e.endswith(".cif")
        ]
        if args.prot_names is not None:
            with open(args.prot_names, "r") as f:
                prot_list = f.read().splitlines()
            pdbs = [e for e in pdbs if os.path.basename(e).split(".")[0] in prot_list]
    elif args.pdb_path is not None:
        pdbs = [args.pdb_path]

    # Mamba requires cuda
    device = torch.device(f"cuda")

    model = INAB(config.model)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()
    model.to(device)

    feats_start = time.time()

    success_pdbs = extract_feats(pdbs, config.path)

    feats_end = time.time()

    for pdb in tqdm(success_pdbs, desc="6.Predicting"):
        with open(pdb.replace(".pdb", "_input.pkl"), "rb") as f:
            input_data = pickle.load(f)

        with torch.cuda.device(device):
            output = predict(model, device, input_data)

        with open(pdb.replace(".pdb", "_output.txt"), "w") as f:
            f.write("\n".join([str(e) for e in output]))

        predict_end = time.time()

        with open(pdb.replace(".pdb", "_time.txt"), "w") as f:
            f.write(
                f"feats:{feats_end - feats_start:.2f}s\npredict:{predict_end - feats_end:.2f}s\n"
            )
