# RaptorX-Single

RaptorX-Single: exploring the advantage of single sequence based protein structure prediction

## Environment
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
conda env create -f environment.yml
conda activate RaptorXSingle
```

## Parameter
Download the following parameter files as needed, you can save these files at `params/` folder.

### PLM (protein language model) model parameters

* esm1b_t33_650M_UR50S.pt
    ```
    wget -P params/ https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ```

* esm1v_t33_650M_UR90S_1.pt
    ```
    wget -P params/ https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt
    ```

* prot_t5_xl_uniref50
    ```
    wget -P params/ https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip?download=1
    ```
### Parameters of RaptorX-Single
Please find the parameters at: <a href="https://doi.org/10.5281/zenodo.7351378"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7351378.svg" alt="DOI"></a>

### Parameters for general protein structure prediction
* RaptorX-Single-ESM1b.pt
* RaptorX-Single-ESM1v.pt
* RaptorX-Single-ProtTrans.pt
* RaptorX-Single-ESM1b-ESM1v-ProtTrans.pt

### Parameters for antibody structure prediction
* RaptorX-Single-ESM1b-Ab.pt
* RaptorX-Single-ESM1v-Ab.pt
* RaptorX-Single-ProtTrans-Ab.pt
* RaptorX-Single-ESM1b-ESM1v-ProtTrans-Ab.pt


## Usage
```
pred.py [-h] [--out_dir OUT_DIR] [--plm_param_dir PLM_PARAM_DIR]
               [--device_id DEVICE_ID] [--n_cycle N_CYCLE]
               [--n_worker N_WORKER]
               fasta_path param

positional arguments:
  fasta_path            fasta file or dir.
  param                 param file path.

optional arguments:
  --out_dir             output dir. (default: 'output/')
  --plm_param_dir       param path for PLM models (ESM1b, ESM1v and ProtTrans). (default: 'params/')
  --device_id           device id (-1 for CPU, >=0 for GPU). (default: -1)
  --n_cycle             cycle time. (default: 4)
  --n_worker            DataLoader num_workers. (default: 0)
```
## Runnning

### CPU
```
python pred.py example/seq/ params/RaptorX-Single-ESM1b.pt --out_dir=example/out/
```

### GPU (GPU 0 as example)
```
python pred.py example/seq/ params/RaptorX-Single-ESM1b.pt --out_dir=example/out/ --device_id=0
```

## Benchmark
The benchmark target lists are saved at `benchmark/`.


## Aknowledgements
* https://github.com/facebookresearch/esm
* https://github.com/agemagician/ProtTrans


## Reference

Jing, Xiaoyang, Fandi Wu, and Jinbo Xu. "RaptorX-Single: single-sequence protein structure prediction by integrating protein language models." bioRxiv (2023): 2023-04.

