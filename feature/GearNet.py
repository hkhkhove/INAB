import os,warnings
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops
from torchdrug import data, layers, models, transforms
from torchdrug.layers import geometry

from .oom_wrapper import oom_wrapper

warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

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
            
            for pdb_file in tqdm(pdb_files,desc="3.GearNet(Loading PDBs)",leave=False):
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

@oom_wrapper
def inference(model,input,save_path):
    with torch.no_grad():
        rep = model.forward(input, input.node_feature.float())
    rep=rep['node_feature'].detach().cpu().numpy()
    np.save(save_path,rep)   

def run(pdb_files,path):
    
    assert os.path.exists(path['gearnet_model']), f"GearNet model not found"

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512,512,512,512], 
                            num_relation=7, edge_input_dim=59, num_angle_bin=8,
                            batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
    state_dict = torch.load(path['gearnet_model'], map_location=device)  
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

    dataset=GearNetProteinDataset(pdb_files,transform=transform, lazy=False)

    pdb_files_ = dataset.pdb_files
    dataset = tqdm(dataset, desc="3.GearNet")
    for i,e in enumerate(dataset):
        save_path=pdb_files_[i].replace('.pdb','_gearnet.npy')
        if os.path.isfile(save_path):
            # continue
            pass

        prot_name,_=os.path.splitext(os.path.basename(pdb_files_[i]))
        protein = e['graph']
        _protein = data.Protein.pack([protein]).to(device)
        protein_ = graph_construction_model(_protein).to(device)

        oom=inference(model, protein_, save_path)
        
        if oom:
            with open(path['log_file'],'a') as f:
                f.write(f">{prot_name}(length={len(protein)}): GearNet feature processed using CPU due to CUDA OOM.\n")

if __name__=="__main__":
    """
    Run as a module to extract GearNet representations from PDB files:
    > python -m feature.GearNet --pdb_dir PDB_DIR   
    To specify the GPU device:
    > CUDA_VISIBLE_DEVICES=INDEX python -m feature.GearNet --pdb_dir PDB_DIR
    """
    import argparse
    argparser=argparse.ArgumentParser(description="Extract GearNet representations from PDB files")
    argparser.add_argument('--pdb_dir',type=str,required=True,help='Directory containing PDB files')
    args=argparser.parse_args()

    pdb_files=[os.path.join(args.pdb_dir,e) for e in os.listdir(args.pdb_dir) if e.endswith('.pdb') or e.endswith('.cif')]
    # pdb_files=[e for e in pdb_files if not os.path.isfile(e.replace('.pdb','_gearnet.npy'))]

    import config

    run(pdb_files,config.path)