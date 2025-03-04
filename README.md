# INAB: Identify Nucleic Acid Binding Domain via Cross-modal Protein Language Models and Multiscale Computation

![Pipeline](./images/pipeline.png)
## Dependencies
> Python Version: 3.10.16
```
troch==2.5.1
biopython==1.84
mamba-ssm==2.2.4
pandas==1.5.3
scikit-learn==1.6.0
tqdm==4.67.1
transformers==4.28.1
tyro==0.9.6
torchdrug==0.2.1
fair-esm==2.0.0
causal-conv1d==1.5.0.post8
numpy==1.24.4
```

## Third-party Tools

[PSI-BLAST](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
[HH-suite](https://github.com/soedinglab/hh-suite)
[DSSP](https://swift.cmbi.umcn.nl/gv/dssp/DSSP_5.html)
[ESM-2](https://github.com/facebookresearch/esmn)
[GearNet](https://github.com/DeepGraphLearning/GearNet)
[SaProt](https://github.com/westlake-repl/SaProt)
[Foldseek](https://github.com/steineggerlab/foldseek)

## Dataset and pretrained models

- Download here [Google Drive](https://drive.google.com/drive/folders/1KLv127DwIMTm308UcSMp-UsKhIjPhhyH?usp=sharing). After downloading, unzip the dataset to the ```dataset``` directory and put the model weights in the ```pretrained_models``` directory. 
- The dataset comprises features and hierarchical labels. The provided features exclusively include PDB files and PSSM/HHM profiles, while additional features must be extracted using the ```extract_feat.py``` script. 
- Pretrained models with the suffix "GraphBind" denote those trained on the GraphBind training set, whereas all other models were trained using the INAB training set.

## Demo
**1. Extract and combine features**
```
python extract_feats.py --dir demo/6cf2_F
python combine_feats.py --dir demo/6cf2_F --prot_list demo/6cf2_F/demo.txt --save_name 6cf2_F
```
**2. Predict using the pretrained model**
```
python predict.py --model_path pretrained_models/INAB_RNA_model.pth --input demo/6cf2_F/6cf2_F_features.pkl
```
