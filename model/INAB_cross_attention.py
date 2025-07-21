from torch import nn
from model.egnn_clean import EGNN
from mamba_ssm import Mamba2


# feat_dim
# hhm(30): 0-29
# pssm(20): 30-49
# ss(14): 50-63
# af(7): 64-70
# esm2_rep(1280): 71-1350
# gearnet_rep(512): 1351-1862
# saprot_rep(446): 1863-2308
feats_dim = {
    "all": 2309,
    "no_hhm": 2279,
    "no_pssm": 2289,
    "no_hhm_pssm": 2259,
    "no_ss": 2295,
    "no_af": 2302,
    "no_ss_af": 2288,
    "no_esm2": 1029,
    "no_gearnet": 1797,
    "no_saprot": 1863,
    "no_plm": 71,
    "no_gearnet_saprot": 1351,
    "no_esm2_saprot": 583,
    "no_esm2_gearnet": 517,
    "no_hhm_pssm_esm2_gearnet": 467,
}


class INAB(nn.Module):
    def __init__(self, config):
        super(INAB, self).__init__()
        self.config = config
        self.emmbedding = nn.Linear(feats_dim[config["feats"]], config["d_model"])
        self.cross_attention = nn.MultiheadAttention(embed_dim=config["d_model"], num_heads=1)
        self.W_Q = nn.Linear(config["d_model"], config["d_model"])
        self.W_K = nn.Linear(config["d_model"], config["d_model"])
        self.W_V = nn.Linear(config["d_model"], config["d_model"])

        if config["seq_model"] == "mamba":
            mamba_modules = [
                Mamba2(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=config["d_model"],  # Model dimension d_model
                    d_state=64,  # SSM state expansion factor, typically 64 or 128
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    headdim=64,  # ensure (d_model*expand)%headdim==0 and ((d_model*expand)/headdim)%8==0, default 64
                )
                for _ in range(config["num_seq_model_layers"])
            ]
            self.seq_model = nn.Sequential(*mamba_modules)
        elif config["seq_model"] == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=config["d_model"], nhead=1, dropout=0.1, batch_first=True)
            self.seq_model = nn.TransformerEncoder(encoder_layer, num_layers=config["num_seq_model_layers"])
        else:
            raise ValueError("seq_model should be either mamba or transformer")

        self.struc_model = EGNN(
            in_node_nf=config["d_model"],
            hidden_nf=config["d_model"],
            out_node_nf=config["d_model"],
            in_edge_nf=1,
            n_layers=config["num_egnn_layers"],
            attention=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(config["d_model"], 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, node_feats, coords, edges, edge_attr):
        if self.config["order"] == "seq_struc":
            h = self.emmbedding(node_feats).unsqueeze(dim=0)  # (batch,seq_len,feat_dim), batch=1

            seq_h = self.seq_model(h)

            struct_h, x = self.struc_model(h.squeeze(0), coords, edges, edge_attr)  # (batch,seq_len,feat_dim)
            struct_h = struct_h.unsqueeze(dim=0)  # (batch,seq_len,feat_dim), batch=1
            Q = self.W_Q(seq_h)
            K = self.W_K(struct_h)
            V = self.W_V(struct_h)

            fused_h, _ = self.cross_attention(Q, K, V)

            y = self.mlp(fused_h)

        elif self.config["order"] == "struc_seq":
            h = self.emmbedding(node_feats)

            h1, x = self.struc_model(h, coords, edges, edge_attr)
            h1 = h1.unsqueeze(dim=0)

            h2 = self.seq_model(h1)

            y = self.mlp(h2)

        else:
            raise ValueError("order should be either seq_struc or struc_seq")

        return y
