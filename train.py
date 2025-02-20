from copy import deepcopy
from dataclasses import dataclass, field
import os
import pickle
import random
import warnings
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    explained_variance_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
import torch
from torch import nn
from torch.optim.adam import Adam
from tqdm import tqdm
import tyro
from model.INAB import INAB

warnings.filterwarnings("ignore")

@dataclass
class Args:
    lr:float=1e-4
    exp_name:str=""
    gpu:int=2
    num_epochs:int=100
    task:str="dna"
    L1:float=0
    L2:float=0
    mode:str="regression"
    d_model:int=512
    dataset:str="ours"
    num_seq_model_layers:int=3
    num_egnn_layers:int=4
    seq_model:str="mamba"
    gpus:list[int]= field(default_factory=list)
    criterion:str="huber"
    huber_threshold:float=0.5
    cutoff:str='12'
    feats:str="all"
    """all,no_hmm,no_pssm,no_pssm_hmm,no_ss,no_af,no_ss_af,no_esm2,no_gearnet,no_saprot,no_handcrafted,no_plm"""
    order:str="ME"
    lr_step:int=7
    lr_gamma:float=0.1
    cross_validate:bool=False

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

def k_fold(data,split=0.2):
    for i in range(0,len(data),int(len(data)*split)):
        yield data[:i]+data[i+int(len(data)*split):],data[i:i+int(len(data)*split)]

def filter_feats(feats,type):
    """
    feat_dim
    hmm(30): 0-29
    pssm(20): 30-49
    ss(14): 50-63
    af(7): 64-70
    esm2_rep(1280): 71-1350
    gearnet_rep(512): 1351-1862
    saprot_rep(446): 1863-2308
    """
    if type=="all":
        return feats
    elif type=="no_hmm":
        return feats[:,30:]
    elif type=="no_pssm":
        return np.concatenate((feats[:,:30],feats[:,50:]),axis=1)
    elif type=="no_hmm_pssm":
        return feats[:,50:]
    elif type=="no_ss":
        return np.concatenate((feats[:,:50],feats[:,64:]),axis=1)
    elif type=="no_af":
        return np.concatenate((feats[:,:64],feats[:,71:]),axis=1)
    elif type=="no_ss_af":
        return np.concatenate((feats[:,:50],feats[:,71:]),axis=1)
    elif type=="no_esm2":
        return np.concatenate((feats[:,:71],feats[:,1351:]),axis=1)
    elif type=="no_gearnet":
        return np.concatenate((feats[:,:1351],feats[:,1863:]),axis=1)
    elif type=="no_saprot":
        return feats[:,:1863]
    elif type=="no_plm":
        return feats[:,:71]
    elif type=="no_handcrafted":
        return feats[:,71:]
    else:
        raise ValueError("Invalid feats type. Expected 'no_hmm', 'no_pssm', 'no_hmm_pssm', 'no_ss', 'no_af', 'no_ss_af', 'no_esm2', 'no_gearnet', 'no_saprot', 'no_plm' or 'no_handcrafted'.")

def train(model,device,criterion,optimizer,data,result_name):
    total_loss = 0
    total_batches = 0
    model.train()  
    all_outputs=[]
    all_labels=[]
    
    for inputs, labels in data:
        prot_name,node_feats,coords,edges,edge_attr=inputs
        coords=torch.from_numpy(coords).to(device).float()
        edges=torch.from_numpy(edges).to(device).long()
        edge_attr=torch.from_numpy(edge_attr).to(device).float()
        node_feats=filter_feats(node_feats,args.feats)
        node_feats=torch.from_numpy(node_feats).to(device).float()

        labels=torch.from_numpy(labels).to(device).float()

        if args.mode=="classification":
            labels = torch.where(labels > 0.5, torch.tensor(1.0, dtype=torch.float).to(device), torch.tensor(0.0, dtype=torch.float).to(device))

        outputs = model(node_feats,coords,edges,edge_attr)
        loss = criterion(outputs, labels.unsqueeze(0).unsqueeze(2))
        #loss=F.l1_loss(outputs,labels)

        # loss,_,_=dilate_loss(outputs,labels.unsqueeze(0).unsqueeze(2),0.5,0.01,device)

        total_loss += loss.item()
        total_batches += 1

        # l1_regularization = torch.tensor(0.).to(device)
        # l2_regularization = torch.tensor(0.).to(device)
        # for param in model.parameters():
        #     l1_regularization += torch.norm(param, 1)
        #     #l2_regularization += torch.norm(param, 2)
        #     l2_regularization += torch.sum(param ** 2)

        # loss += wandb.config['L1'] * l1_regularization + wandb.config['L2'] * l2_regularization


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_outputs.append(outputs.detach().float().cpu().numpy().ravel()) 
        all_labels.append(labels.detach().float().cpu().numpy().ravel())  
    
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / total_batches
    train_results=measure(all_outputs,all_labels,result_name)
    train_results['train_loss']=avg_loss

    return train_results

def evaluate(model,device,data,result_name):
    model.eval() 
    with torch.no_grad():
        all_outputs=[]
        all_labels=[]
        for inputs, labels in data:
            prot_name,node_feats,coords,edges,edge_attr=inputs

            coords=torch.from_numpy(coords).to(device).float()
            edges=torch.from_numpy(edges).to(device).long()
            edge_attr=torch.from_numpy(edge_attr).to(device).float()
            node_feats=filter_feats(node_feats,args.feats)
            node_feats=torch.from_numpy(node_feats).to(device).float()

            labels=torch.from_numpy(labels).to(device).float()

            if args.mode=="classification":
                labels = torch.where(labels > 0.5, torch.tensor(1.0, dtype=torch.float).to(device), torch.tensor(0.0, dtype=torch.float).to(device))

            outputs = model(node_feats,coords,edges,edge_attr)

            all_outputs.append(outputs.detach().float().cpu().numpy().ravel()) 
            all_labels.append(labels.detach().float().cpu().numpy().ravel())  

        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)

        results=measure(all_outputs,all_labels,result_name)

        return results

def main(args,tr_data,val_data=None,te_data=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(f'./dataset/{args.dataset}'):
        raise FileNotFoundError(f'Dataset {args.dataset} not exists.')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=INAB(args).to(device)

    if args.mode=="regression":
        if args.criterion=="mse":
            criterion = nn.MSELoss()
        elif args.criterion=="huber":
            criterion = nn.SmoothL1Loss(beta=args.huber_threshold)
        else:
            raise ValueError("Invalid criterion. Expected 'mse', 'huber', 'weightedMse'.")
    elif args.mode=="classification": 
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid mode. Expected 'regression' or 'classification'.")
    
    optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=args.L2)
    patience=10
    best_val_results={
        'val_mse':float('inf'),
        'val_mae':float('inf'),
        'val_r2':-float('inf')
    }

    best_weights = None
    epochs_without_improvement=0
    #scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)  

    for epoch in tqdm(range(args.num_epochs),desc="Training"):
            if args.cross_validate:
                train_results=train(model,device,criterion,optimizer,tr_data,'train')
                val_results=evaluate(model,device,val_data,'val')
                
                results=train_results | val_results

                if results['val_r2'] > best_val_results['val_r2']:
                    best_val_results=val_results
                    best_weights = deepcopy(model.state_dict())
                    epochs_without_improvement=0
                else:
                    epochs_without_improvement+=1

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping at epoch {epoch}, best epoch is {epoch-patience}")
                    print(f"Best val results: {best_val_results}")
                    return best_val_results,epoch-patience 
            else:
                train_results=train(model,device,criterion,optimizer,tr_data,'train')
                test_results=evaluate(model,device,te_data,'test')
                results=train_results | test_results

                print(results)

            #scheduler.step()
            
if __name__ == "__main__":

    args=tyro.cli(Args)
    with open(f'dataset/INAB/dna_train.pkl','rb') as f:
        tr_data=pickle.load(f)
    with open(f'dataset/INAB/dna_test.pkl','rb') as f:
        te_data=pickle.load(f)

    random.seed(76)
    random.shuffle(tr_data)
    random.shuffle(te_data)
    
    if args.cross_validate:
        results=[]
        for tr,val in k_fold(tr_data):
            results.append(main(args,tr,val,te_data))
        print(results)
    else:
        main(args,tr_data,None,te_data)
    

    



    







