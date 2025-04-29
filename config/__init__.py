import os

base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model={
    "d_model":512,
    "seq_model":"mamba",
    "num_seq_model_layers":3,
    "num_egnn_layers":4,
    "feats":"all",
    "order":"ME",
    "mode":"regression",
}

path={
    "base_dir":base_dir,
    #By default, these tools are placed in the project's specified directory, but you can manually set their paths.
    "blast":os.path.join(base_dir,'lib','psiblast'),
    "hhblits":os.path.join(base_dir,'lib','hhblits'),
    "foldseek":os.path.join(base_dir,'lib','foldseek'),
    "dssp":os.path.join(base_dir,'lib','dssp'),
    "saprot_model":os.path.join(base_dir,'model_parameters','SaProt_650M_PDB'),
    "gearnet_model":os.path.join(base_dir,'model_parameters','mc_gearnet_edge.pth'),

    #Paths below must be manually set
    "pssmdb":"/home2/public/database/uniref90/uniref90",
    "hhmdb":"/home2/public/database/UniRef30_2023_02/UniRef30_2023_02",
}
