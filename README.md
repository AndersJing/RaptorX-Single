# RaptorX-Single

Single-sequence protein structure prediction by integrating protein language models

## Environment
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
conda env create -f environment.yml
conda activate RaptorXSingle
```

## Code
```
git clone https://github.com/AndersJing/RaptorX-Single.git
cd RaptorX-Single/
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
    wget -P params/ https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip
    ```

### Parameters of RaptorX-Single
Please find the parameters at: <a href="https://doi.org/10.5281/zenodo.7351378"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7351378.svg" alt="DOI"></a>

### Parameters for general protein structure prediction
* RaptorX-Single-ESM1b.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b.pt`
* RaptorX-Single-ESM1v.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1v.pt`
* RaptorX-Single-ProtTrans.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ProtTrans.pt`
* RaptorX-Single-ESM1b-ESM1v-ProtTrans.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b-ESM1v-ProtTrans.pt`

### Parameters for antibody structure prediction
* RaptorX-Single-ESM1b-Ab.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b-Ab.pt`
* RaptorX-Single-ESM1v-Ab.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1v-Ab.pt`
* RaptorX-Single-ProtTrans-Ab.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ProtTrans-Ab.pt`
* RaptorX-Single-ESM1b-ESM1v-ProtTrans-Ab.pt: `wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b-ESM1v-ProtTrans-Ab.pt`


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
Jing, X., Wu, F., Luo, X., & Xu, J. (2024). Single-sequence protein structure prediction by integrating protein language models. Proceedings of the National Academy of Sciences, 121(13), e2308788121. <a href="https://www.pnas.org/doi/10.1073/pnas.2308788121"> Link </a>

Jing, X., Wu, F., Luo, X., & Xu, J. (2023). RaptorX-Single: single-sequence protein structure prediction by integrating protein language models. bioRxiv, 2023-04. <a href="https://www.biorxiv.org/content/10.1101/2023.04.24.538081v2"> Link </a>

## Contact
Xiaoyang Jing: xyjing@ttic.edu


