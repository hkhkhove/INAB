from model.INAB import INAB
from dataclasses import dataclass
import torch
import pickle
from tqdm import tqdm
import tyro

torch.set_grad_enabled(False)

@dataclass
class Args:
    gpu:int=0
    mode:str="regression"
    model_path:str="pretrained_models/INAB_DNA_model.pth"
    input:str="demo/5f7q_E/5f7q_E_features.pkl"
    d_model:int=512
    num_seq_model_layers:int=3
    num_egnn_layers:int=4
    seq_model:str="mamba"
    feats:str="all"
    order:str="ME"

def evaluate(model,device,input_data):
    
    results={}
    
    for features in tqdm(input_data,desc="Predicting"):
        
        prot_name,node_feats,coords,edges,edge_attr=features

        coords=torch.from_numpy(coords).to(device).float()
        edges=torch.from_numpy(edges).to(device).long()
        edge_attr=torch.from_numpy(edge_attr).to(device).float()
        node_feats=torch.from_numpy(node_feats).to(device).float()

        outputs = model(node_feats,coords,edges,edge_attr)
        results[prot_name]=outputs.cpu().numpy()

    return results
    
if "__main__"==__name__:
    args=tyro.cli(Args)
   
    device = torch.device(f"cuda:{args.gpu}")
    
    with open(args.input,'rb') as f:
        input_data=pickle.load(f)

    model=INAB(args)
    model.load_state_dict(torch.load(args.model_path,weights_only=True))
    model.eval()
    model.to(device)

    with torch.cuda.device(device):
        results=evaluate(model,device,input_data)
    
    print(results)
    