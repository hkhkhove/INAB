from torch import nn
from model.egnn_clean import EGNN
from mamba_ssm import Mamba2

# feat_dim
# hmm(30): 0-29
# pssm(20): 30-49
# ss(14): 50-63
# af(7): 64-70
# esm2_rep(1280): 71-1350
# gearnet_rep(512): 1351-1862
# saprot_rep(446): 1863-2308
feats_dim={
    "all":2309,
    "no_hmm":2279,
    "no_pssm":2289,
    "no_hmm_pssm":2259,
    "no_ss":2295,
    "no_af":2302,
    "no_ss_af":2288,
    "no_esm2":1029,
    "no_gearnet":1797,
    "no_saprot":1863,
    "no_plm":71
}

class INAB(nn.Module):
    def __init__(self,args):
        super(INAB, self).__init__()
        self.args=args        

        self.emmbedding=nn.Linear(feats_dim[args.feats],args.d_model)
 
        if args.seq_model=="mamba":
            mamba_modules=[
                    Mamba2(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=args.d_model, # Model dimension d_model
                    d_state=64,  # SSM state expansion factor, typically 64 or 128
                    d_conv=4,    # Local convolution width
                    expand=2,    # Block expansion factor
                    headdim=64   #ensure (d_model*expand)%headdim==0 and ((d_model*expand)/headdim)%8==0, default 64
                ) for _ in range(args.num_seq_model_layers)
            ]
            self.seq_model=nn.Sequential(*mamba_modules)
        elif args.seq_model=="transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=1,dropout=0.1,batch_first=True)
            self.seq_model=nn.TransformerEncoder(encoder_layer, num_layers=args.num_seq_model_layers)
        else:
            raise ValueError("seq_model should be either mamba or transformer")      

        self.egnn=EGNN(in_node_nf=args.d_model, hidden_nf=args.d_model, out_node_nf=args.d_model, in_edge_nf=1, n_layers=args.num_egnn_layers,attention=True)

        if args.mode=="regression":
            self.mlp=nn.Sequential(
                nn.Linear(args.d_model,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,1),
                nn.Sigmoid()
            )
        elif args.mode=="classification":
            self.mlp=nn.Sequential(
                nn.Linear(args.d_model,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,1),
            )

        
    def forward(self,node_feats,coords,edges,edge_attr):
        if self.args.order=="ME":
            h=self.emmbedding(node_feats).unsqueeze(dim=0) #(batch,seq_len,feat_dim)
            
            h1=self.seq_model(h).squeeze(dim=0) #(seq_len,feat_dim)
            
            h2,x=self.egnn(h1,coords,edges,edge_attr)
            
            y=self.mlp(h2)

        elif self.args.order=="EM":
            h=self.emmbedding(node_feats)

            h1,x=self.egnn(h,coords,edges,edge_attr)
            h1=h1.unsqueeze(dim=0)
                        
            h2=self.seq_model(h1)

            y=self.mlp(h2)

        return y
    