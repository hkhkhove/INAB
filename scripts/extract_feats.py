import json
import multiprocessing as mp
import os
import subprocess
import argparse
import warnings
import traceback
import numpy as np
import torch
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
import esm
from transformers import EsmTokenizer, EsmForMaskedLM
from rdkit import Chem
from rdkit.Chem import rdmolops
from torchdrug import data, layers, models, transforms
from torchdrug.layers import geometry

warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

def get_seq(prot):
    AA_dic = {'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','TYR':'Y','ASP':'D','ASN':'N',
          'GLU':'E','LYS':'K','GLN':'Q','MET':'M','SER':'S','THR':'T','CYS':'C','PRO':'P','HIS':'H','ARG':'R'}
    pdb_file=f"{HOME_DIR}/features/{prot}.pdb"

    parser=PDBParser()
    structure = parser.get_structure("tmp", pdb_file)
    model=structure[0]
    seq=""
    for residue in model.get_residues():
        if residue.id[0]==' ':
            seq+=AA_dic[residue.resname]
    save_path=f"{HOME_DIR}/features/{prot}.fasta"

    with open(save_path,'w') as f:
        f.write(f'>{prot}\n{seq}\n')
        
def run_BLAST(prot):
    outfmt_type = 5
    num_iter = 3
    evalue_threshold = 0.001
    fasta_file=f"{HOME_DIR}/features/{prot}.fasta"
    xml_file = f"{HOME_DIR}/features/{prot}.xml"
    pssm_file = f"{HOME_DIR}/features/{prot}.pssm"
    if os.path.isfile(pssm_file):
        pass
    else:
        cmd = [BLAST,
                '-query', fasta_file,
                '-db',BLAST_DB,
                '-out',xml_file,
                '-evalue',str(evalue_threshold),
                '-num_iterations',str(num_iter),
                '-outfmt',str(outfmt_type),
                '-out_ascii_pssm',pssm_file,  # Write the pssm file
                '-num_threads','8']                       
        subprocess.run(cmd)

def run_HHblits(prot):
    fasta_file=f"{HOME_DIR}/features/{prot}.fasta"
    hhm_file = f"{HOME_DIR}/features/{prot}.hhm"#ohhm

    if os.path.isfile(hhm_file):
        pass
    else:
        cmd=[HHBLITS,
             '-i',fasta_file,
             '-d',HH_DB,
             '-ohhm',hhm_file,
             '-cpu','8',
             '-v','0']
        subprocess.run(cmd)

def run_dssp(prot):
    chain_file=f"{HOME_DIR}/features/{prot}.pdb"
    dssp_file=f'{HOME_DIR}/features/{prot}.dssp'
    if os.path.isfile(dssp_file):
        pass
    else:
        cmd=[DSSP,
                '-i',chain_file,
                '-o',dssp_file]
        result=subprocess.run(cmd)

        if result.returncode!=0:
            print(f"{prot} DSSP Failed")

def run_esm2(prot_list):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    for prot in tqdm(prot_list,desc="ESM-2"):
        if os.path.isfile(f'{HOME_DIR}/features/{prot}.esm2.npy'):
            continue
        try:
            with open(f'{HOME_DIR}/features/{prot}.fasta','r') as f:
                f.readline()
                seq=f.readline().strip()
            data=[(prot,seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens=batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        except Exception as e:
            print(f"{prot} error: {e}")
            traceback.print_exc()
            continue
        token_representations = results["representations"][33]
        prot_rep=token_representations[:,1:-1,:]
        for tuple, rep in zip(data,prot_rep):
            prot_name=tuple[0]
            prot_length=len(tuple[1])
            assert prot_length==rep.shape[0]
            rep_numpy = rep.detach().cpu().numpy()
            np.save(f'{HOME_DIR}/features/{prot}.esm2.npy', rep_numpy)

def run_saprot(prot_list):
    
    # Get structural seqs from pdb file
    def get_struc_seq(foldseek,
                    path,
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
        assert os.path.exists(path), f"Pdb file not found: {path}"
        assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
        prot_name=os.path.basename(path).split('.')[0]

        tmp_save_path = f"get_struc_seq_{prot_name}.tsv"
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
        os.system(cmd)

        seq_dict = {}
        name = os.path.basename(path)
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

    model_path = "../pretrained_models/SaProt_650M_PDB"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval() 
    for prot in tqdm(prot_list,desc="SaProt"):
        if os.path.isfile(f'{HOME_DIR}/features/{prot}.saprot.npy'):
            continue
        pdb_path=f'{HOME_DIR}/features/{prot}.pdb'
        chain_id=prot.split('_')[1]
        chains=[]
        chains.append(chain_id)
        try:
            # Extract the "A" chain from the pdb file and encode it into a struc_seq
            # pLDDT is used to mask low-confidence regions if "plddt_path" is provided
            parsed_seqs = get_struc_seq(FOLDSEEK, pdb_path, chains)[chain_id]
            seq, foldseek_seq, combined_seq = parsed_seqs
            
            tokens = tokenizer.tokenize(combined_seq)
            #print(len(tokens))

            inputs = tokenizer(combined_seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            outputs = outputs.logits.squeeze(0)
            outputs = outputs.detach().cpu().numpy()[1:-1,:]
            np.save(f'{HOME_DIR}/features/{prot}.saprot.npy', outputs)
        except Exception as e:
            print(f"{prot} error: {e}")
            traceback.print_exc()
            continue

class GearNetProteinDataset(data.ProteinDataset):
        def __init__(self, pdb_files,transform=None, lazy=False):
            super().__init__()
            self.transform = transform
            self.lazy=lazy
            self.targets={}
            self.data = []
            self.pdb_files = []
            self.sequences = []
            self.load_pdbs(pdb_files)

        def load_pdbs(self, pdb_files):
            
            for pdb_file in pdb_files:
                mol = Chem.MolFromPDBFile(pdb_file,sanitize=False) 
                if not mol:
                    print(f"Can't construct molecule from pdb file `{pdb_file}`. Ignore this sample.")
                    return None
                # try:
                #     rdmolops.SanitizeMol(mol) 
                # except Exception as e:
                #     logger.debug("Can't sanitize molecule from pdb file `%s`. Ignore this sample. Exception: %s" % (pdb_file, e))
                #     #continue
                protein = data.Protein.from_molecule(mol,atom_feature=None, bond_feature=None)
                if not protein:
                    print(f"Can't construct protein from pdb file `{pdb_file}`. Ignore this sample.")
                    continue
                if hasattr(protein, "residue_feature"):
                    with protein.residue():
                        protein.residue_feature = protein.residue_feature.to_dense()       
                self.data.append(protein)
                self.pdb_files.append(pdb_file)
                self.sequences.append(protein.to_sequence() if protein else None)

def run_gearnet(prot_list):

    model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512,512,512,512], 
                            num_relation=7, edge_input_dim=59, num_angle_bin=8,
                            batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
    state_dict = torch.load("../pretrained_models/mc_gearnet_edge.pth", map_location=device)  
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                    geometry.KNNEdge(k=10, min_distance=5),
                                                                    geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet"             )

    #truncuate_transform = transforms.TruncateProtein(max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view='residue')
    transform = transforms.Compose([protein_view_transform])

    pdb_files=[f"{HOME_DIR}/features/{prot}.pdb" for prot in prot_list]
    dataset=GearNetProteinDataset(pdb_files,transform=transform, lazy=False)

    filenames = dataset.pdb_files
    dataset = tqdm(dataset, f"GearNet")
    for i,e in enumerate(dataset):
        prot_name=os.path.basename(filenames[i]).split(".")[0]
        if os.path.isfile(f"{HOME_DIR}/features/{prot_name}.gearnet.npy"):
            continue
        try:
            protein = e['graph']
            _protein = data.Protein.pack([protein]).to(device)
            protein_ = graph_construction_model(_protein).to(device)
            rep = model.forward(protein_, protein_.node_feature.float())
            rep=rep['node_feature'].detach().cpu().numpy()
            np.save(f"{HOME_DIR}/features/{prot_name}.gearnet.npy",rep)
        except Exception as e:
            print(f"{prot_name} error: {e}")
            traceback.print_exc()
            continue

def parallel(num_processes, prot_list, func):
    with mp.Pool(num_processes) as pool:
        results = pool.imap_unordered(func, prot_list)
        pbar=tqdm(total=len(prot_list),desc=func.__name__)
        for _ in results:
            pbar.update()

if __name__ == "__main__":

    argparser=argparse.ArgumentParser()
    argparser.add_argument('--dir',type=str ,default="../dataset/INAB")
    args=argparser.parse_args()

    BLAST = '../lib/psiblast'
    BLAST_DB = ''
    HHBLITS = '../lib/hhblits'
    HH_DB = ''
    FOLDSEEK = '../lib/foldseek'
    DSSP="../lib/dssp"

    HOME_DIR=args.dir

    if not os.path.exists(HOME_DIR):
        print("Please download the dataset and extract it to the dataset directory")
        exit(1)
    if not os.path.exists(f'{HOME_DIR}/features'):
        os.makedirs(f'{HOME_DIR}/features')

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    prot='6cf2_F'
    get_seq(prot)
    # parallel(8,prot_list,run_BLAST(prot))
    # parallel(8,prot_list,run_HHblits(prot))
    run_dssp(prot)
    run_esm2([prot])
    run_saprot([prot])
    run_gearnet([prot])
