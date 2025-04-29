from copy import deepcopy
from dataclasses import dataclass, field
import os
import time
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
import tyro

from model.INAB import INAB
import config


warnings.filterwarnings("ignore")

@dataclass
class Args:
    exp_name:str=""
    log_file:str=""
    dataset:str="INAB"
    task:str="dna"
    lr:float=1e-4
    criterion:str="huber"
    huber_threshold:float=0.5
    d_model:int=512
    seq_model:str="mamba"
    num_seq_model_layers:int=3
    num_egnn_layers:int=4
    feats:str="all"
    order:str="ME"
    mode:str="regression"
    gpu:int=0
    cross_validate:bool=False
    num_epochs:int=100
    L1:float=0
    L2:float=0
    gpus:list[int]= field(default_factory=list)
    lr_step:int=7
    lr_gamma:float=0.1

def load_data(dataset,task):
    print(f"Loading {dataset} {task} data...")
    def load(prot_list):
        data=[]
        for prot in prot_list:
            input_file=f'./dataset/{dataset}/features/{prot}_input.pkl'
            if os.path.isfile(input_file):
                with open(input_file,'rb') as f:
                    input_data=pickle.load(f)
                with open(f'./dataset/{dataset}/labels/{prot}.txt','r') as f:
                    lines=f.read().splitlines()
                dna_label=np.array([float(e.strip().split()[0]) for e in lines])
                rna_label=np.array([float(e.strip().split()[1]) for e in lines])
                if task=="dna":
                    label_data=dna_label
                elif task=="rna":
                    label_data=rna_label
                else:
                    raise ValueError("Invalid task. Expected 'dna' or 'rna'.")

                if len(input_data[1])==len(label_data):
                    data.append((input_data,label_data))
                else:
                    with open(args.log_file,'a') as f:
                        f.write(f">{prot}: Length mismatch, feature {len(input_data[1])} vs label {len(label_data)}\n")

        return data
    
    with open(f'./dataset/{dataset}/{task}_train.txt','r') as f:
        tr_list=f.read().splitlines()
    with open(f'./dataset/{dataset}/{task}_test.txt','r') as f:
        te_list=f.read().splitlines()
    tr_data,te_data=load(tr_list),load(te_list)

    print("Data loaded: ")
    print(f"- train: {len(tr_data)}/{len(tr_list)}")
    print(f"- test: {len(te_data)}/{len(te_list)}")

    return tr_data,te_data
    
def measure(outputs,labels,name):
    if config.model["mode"]=="regression":
        mse=mean_squared_error(labels,outputs)
        mae=mean_absolute_error(labels,outputs)
        # ev_score=explained_variance_score(labels,outputs)
        r2=r2_score(labels,outputs)
        return {
            f"{name}_mse":mse,
            f"{name}_mae":mae,
            f"{name}_r2":r2
        }
    elif config.model["mode"]=="classification":
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

def print_results(epoch,results):
    str=f'|epoch:{epoch}|'
    for k,v in results.items():
        str+=f"{k}:{v:.4f}|"
        
    with open(args.log_file,'a') as f:
        f.write(str+'\n')

    print(str)

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
    elif type=="no_gearnet_saprot":
        return feats[:,:1351]
    elif type=="no_esm2_saprot":
        return np.concatenate((feats[:,:71],feats[:,1351:1863]),axis=1)
    elif type=="no_esm2_gearnet":
        return np.concatenate((feats[:,:71],feats[:,1863:]),axis=1)

    else:
        raise ValueError("Invalid feats type.")

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
        node_feats=filter_feats(node_feats,config.model["feats"])
        node_feats=torch.from_numpy(node_feats).to(device).float()

        labels=torch.from_numpy(labels).to(device).float()

        if config.model["mode"]=="classification":
            labels = torch.where(labels > 0.5, torch.tensor(1.0, dtype=torch.float).to(device), torch.tensor(0.0, dtype=torch.float).to(device))

        outputs = model(node_feats,coords,edges,edge_attr)

        outputs=outputs.squeeze()
        labels=labels.squeeze()
        
        # print(outputs.shape,labels.shape)
        loss = criterion(outputs, labels)
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
            node_feats=filter_feats(node_feats,config.model["feats"])
            node_feats=torch.from_numpy(node_feats).to(device).float()

            labels=torch.from_numpy(labels).to(device).float()

            if config.model["mode"]=="classification":
                labels = torch.where(labels > 0.5, torch.tensor(1.0, dtype=torch.float).to(device), torch.tensor(0.0, dtype=torch.float).to(device))

            outputs = model(node_feats,coords,edges,edge_attr)
            
            outputs=outputs.squeeze()
            labels=labels.squeeze()
            
            all_outputs.append(outputs.detach().float().cpu().numpy().ravel()) 
            all_labels.append(labels.detach().float().cpu().numpy().ravel())  

        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)

        results=measure(all_outputs,all_labels,result_name)

        return results

def main(tr_data,val_data=None,te_data=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=INAB(config.model).to(device)

    if config.model["mode"]=="regression":
        if args.criterion=="mse":
            criterion = nn.MSELoss()
        elif args.criterion=="huber":
            criterion = nn.SmoothL1Loss(beta=args.huber_threshold)
        else:
            raise ValueError("Invalid criterion. Expected 'mse', 'huber', 'weightedMse'.")
    elif config.model["mode"]=="classification": 
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid mode. Expected 'regression' or 'classification'.")
    
    optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=args.L2)
    patience=7
    best_val_results={
        'val_mse':float('inf'),
        'val_mae':float('inf'),
        'val_r2':-float('inf')
    }
    if config.model['mode']=="regression":
        best_test_results={
            'test_mse':float('inf'),
            'test_mae':float('inf'),
            'test_r2':-float('inf')
        }
    elif config.model['mode']=="classification":
        best_test_results={
            'test_auroc':-float('inf'),
            'test_auprc':-float('inf'),
            'test_mcc':-float('inf')
        }
    best_weights = None
    epochs_without_improvement=0
    #scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)  

    print("Training...")
    for epoch in range(args.num_epochs):
        if args.cross_validate:
            train_results=train(model,device,criterion,optimizer,tr_data,'train')
            val_results=evaluate(model,device,val_data,'val')
            
            results=train_results | val_results
            print_results(epoch,results)

            if results['val_r2'] > best_val_results['val_r2']:
                best_val_results=val_results
                best_weights = deepcopy(model.state_dict())
                epochs_without_improvement=0
            else:
                epochs_without_improvement+=1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}, best epoch is {epoch-patience}")
                print(f"Best val results: {best_val_results}")
                return best_val_results
        else:
            train_results=train(model,device,criterion,optimizer,tr_data,'train')
            test_results=evaluate(model,device,te_data,'test')
            results=train_results | test_results

            print_results(epoch,results)

            if config.model["mode"]=="regression":
                if results['test_r2'] > best_test_results['test_r2']:
                    best_test_results=test_results
                    # best_weights = deepcopy(model.state_dict())
                    epochs_without_improvement=0
                else:
                    epochs_without_improvement+=1
            elif config.model["mode"]=="classification":
                if results['test_auroc'] > best_test_results['test_auroc']:
                    best_test_results=test_results
                    # best_weights = deepcopy(model.state_dict())
                    epochs_without_improvement=0
                else:
                    epochs_without_improvement+=1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}, best epoch is {epoch-patience}")
                print(f"Best test results: {best_test_results}")
                return best_test_results

        #scheduler.step()
            
if __name__ == "__main__":

    args=tyro.cli(Args)
    config.model["d_model"]=args.d_model
    config.model["seq_model"]=args.seq_model
    config.model["num_seq_model_layers"]=args.num_seq_model_layers
    config.model["num_egnn_layers"]=args.num_egnn_layers
    config.model["feats"]=args.feats
    config.model["order"]=args.order
    config.model["mode"]=args.mode

    args.exp_name+=f"_{args.dataset}_{args.task}"
    args.log_file=os.path.join(config.base_dir,'log',args.exp_name+f"_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt")

    tr_data,te_data=load_data(args.dataset,args.task)

    random.seed(76)
    random.shuffle(tr_data)
    random.shuffle(te_data)
    
    if args.cross_validate:
        results=[]
        for tr,val in k_fold(tr_data):
            results.append(main(tr,val,te_data))
        print(results)
    else:
        main(tr_data,te_data=te_data)
    

    



    







