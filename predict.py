from model.INAB import INAB
from model.egnn_clean import EGNN
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,matthews_corrcoef,average_precision_score,explained_variance_score
from dataclasses import dataclass,field
import numpy as np
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
    input_features:str="demo/5f7q_E/5f7q_E_features.pkl"
    d_model:int=512
    num_seq_model_layers:int=3
    num_egnn_layers:int=4
    seq_model:str="mamba"
    feats:str="all"
    order:str="ME"

def measure(outputs,labels,name):
    if args.mode=="regression":
        mse=mean_squared_error(labels,outputs)
        mae=mean_absolute_error(labels,outputs)
        # ev_score=explained_variance_score(labels,outputs)
        r2=r2_score(labels,outputs)
        return {
            f"{name}_mse":mse,
            f"{name}_mae":mae,
            f"{name}_r2":r2
        }
    elif args.mode=="classification":
        auroc=roc_auc_score(labels,outputs)
        outputs_binary=np.where(outputs < 0.5, 0, 1)
        mcc=matthews_corrcoef(labels,outputs_binary)
        auprc=average_precision_score(labels, outputs)
        return {
            f"{name}_auroc":auroc,
            f"{name}_auprc":auprc,
            f"{name}_mcc":mcc
        }
    else:
        raise ValueError("Invalid mode. Expected 'regression' or 'classification'.")

def evaluate(model,device,input_data):
    for features in input_data:
        prot_name,node_feats,coords,edges,edge_attr=features

        coords=torch.from_numpy(coords).to(device).float()
        edges=torch.from_numpy(edges).to(device).long()
        edge_attr=torch.from_numpy(edge_attr).to(device).float()
        node_feats=torch.from_numpy(node_feats).to(device).float()


        outputs = model(node_feats,coords,edges,edge_attr)


    return outputs
    
if "__main__"==__name__:
    args=tyro.cli(Args)
   
    device = torch.device(f"cuda:{args.gpu}")
    
    with open(args.input_features,'rb') as f:
        input_data=pickle.load(f)

    model=INAB(args)
    model.load_state_dict(torch.load(args.model_path,weights_only=True))
    model.eval()
    model.to(device)

    with torch.cuda.device(device):
        results=evaluate(model,device,input_data)
    
    print(results)
    