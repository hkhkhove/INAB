import os,warnings
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM

from .oom_wrapper import oom_wrapper

warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

@oom_wrapper
def inference(model,input,save_path):
    with torch.no_grad():
        output = model(**input)
    output = output.logits.squeeze(0)
    output = output.detach().cpu().numpy()[1:-1,:]
    np.save(save_path, output)    

def run(pdb_files,path):
    assert os.path.exists(path['foldseek']), f"Foldseek not found."
    assert os.path.exists(path['saprot_model']), f"SaProt model not found"

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get structural seqs from pdb file
    def get_struc_seq(foldseek,
                    pdb_path,
                    chains: list = None,
                    process_id: int = 0,
                    plddt_path: str = None,
                    plddt_threshold: float = 70.) -> dict:
        """
        
        Args:
            foldseek: Binary executable file of foldseek
            path: Path to pdb file
            chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
            process_id: Process ID for temporary files. This is used for parallel processing.
            plddt_path: Path to plddt file. If None, plddt will not be used.
            plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        Returns:
            seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
            (seq, struc_seq, combined_seq).
        """
        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        assert os.path.exists(pdb_path), f"Pdb file not found: {pdb_path}"
        assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
        prot_name=os.path.basename(pdb_path).split('.')[0]

        tmp_save_path = f"get_struc_seq_{prot_name}.tsv"
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {pdb_path} {tmp_save_path}"
        os.system(cmd)

        seq_dict = {}
        name = os.path.basename(pdb_path)
        with open(tmp_save_path, "r") as r:
            for i, line in enumerate(r):
                desc, seq, struc_seq = line.split("\t")[:3]
                
                # Mask low plddt
                if plddt_path is not None:
                    with open(plddt_path, "r") as r:
                        plddts = np.array(json.load(r)["confidenceScore"])
                        
                        # Mask regions with plddt < threshold
                        indices = np.where(plddts < plddt_threshold)[0]
                        np_seq = np.array(list(struc_seq))
                        np_seq[indices] = "#"
                        struc_seq = "".join(np_seq)
                
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]

                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                        seq_dict[chain] = (seq, struc_seq, combined_seq)
            
        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return seq_dict

    model_path = path['saprot_model']
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval() 
    for pdb_file in tqdm(pdb_files,desc="4.SaProt"):
        save_path=pdb_file.replace('.pdb','_saprot.npy')
        if os.path.isfile(save_path):
            continue

        prot_name,_=os.path.splitext(os.path.basename(pdb_file))
        chain_id=prot_name.split('_')[1]
        chains=[]
        chains.append(chain_id)
        parsed_seqs = get_struc_seq(path["foldseek"], pdb_file, chains)[chain_id]
        seq, foldseek_seq, combined_seq = parsed_seqs
        input = tokenizer(combined_seq, return_tensors="pt")

        oom=inference(model, input.to(device), save_path)
        
        if oom:
            with open(path['log_file'],'a') as f:
                f.write(f">{prot_name}(length={len(seq)}): SaProt feature processed using CPU due to CUDA OOM.\n")


if __name__=="__main__":
    """
    Run as a module to extract SaProt representations from PDB files:
    > python -m feature.SaProt --pdb_dir PDB_DIR   
    To specify the GPU device:
    > CUDA_VISIBLE_DEVICES=INDEX python -m feature.extract --pdb_dir PDB_DIR
    """
    import argparse
    argparser=argparse.ArgumentParser(description="Extract SaProt representations from PDB files")
    argparser.add_argument('--pdb_dir',type=str,required=True,help='Directory containing PDB files')
    args=argparser.parse_args()

    pdb_files=[os.path.join(args.pdb_dir,e) for e in os.listdir(args.pdb_dir) if e.endswith('.pdb') or e.endswith('.cif')]
    pdb_files=[e for e in pdb_files if not os.path.isfile(e.replace('.pdb','_saprot.npy'))]

    import config

    run(pdb_files,config.path)