#!/usr/bin/env python3
# encoding: utf-8


import os
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset


RESIDUE_TYPES = "ARNDCQEGHILKMFPSTWYVX-BZUOJ"
UNK_RESIDUE_INDEX = 20  # unknown residue (X and BZUOJ)
GAP_RESIDUE_INDEX = 21  # gap


class SeqDataset(Dataset):
    def __init__(self, fasta_path, plm_models=[]):
        self.plm_models = plm_models

        fasta_files = []
        if os.path.isfile(fasta_path):
            fasta_files = [fasta_path]
        elif os.path.isdir(fasta_path):
            fasta_files = glob(f"{fasta_path}/*.fasta")
        self.fasta_files = fasta_files
        self.data_len = len(fasta_files)
        print("fasta num:", len(fasta_files))

        if "ESM1v" in self.plm_models or "ESM1b" in self.plm_models:
            import esm

            self.esm_batch_converter = esm.pretrained.load_model_and_alphabet_core(
                torch.load(os.environ["ESM_param"], map_location="cpu"), None
            )[1].get_batch_converter()

        if "ProtTrans" in self.plm_models:
            from transformers import T5Tokenizer

            self.ProtTrans_tokenizer = T5Tokenizer.from_pretrained(
                os.environ["ProtTrans_param"], do_lower_case=False
            )

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fasta_file = self.fasta_files[idx]

        # target
        target = os.path.basename(fasta_file).split(".")[0]

        # seq
        seq = "".join(
            [_.strip() for _ in open(fasta_file).readlines() if not _.startswith(">")]
        )

        # seq_encoding
        seq_encoding = np.array(
            [RESIDUE_TYPES.index(_) for _ in seq if _ in RESIDUE_TYPES]
        )
        seq_encoding[seq_encoding > GAP_RESIDUE_INDEX] = UNK_RESIDUE_INDEX

        # ESM feature
        ESM_token = 0
        if "ESM1v" in self.plm_models or "ESM1b" in self.plm_models:
            ESM_data = [("", seq)]
            _, _, ESM_token = self.esm_batch_converter(ESM_data)
            ESM_token = ESM_token[0]

        # ProtTrans feature
        ProtTrans_token = 0
        if "ProtTrans" in self.plm_models:
            _seq = [" ".join([_ for _ in seq])]
            ProtTrans_token = self.ProtTrans_tokenizer.batch_encode_plus(
                _seq, add_special_tokens=True, padding=True
            )["input_ids"][0]
            ProtTrans_token = np.array(ProtTrans_token).astype(np.int)

        feature = {
            "seq_encoding": seq_encoding,
            "ESM_token": ESM_token,
            "ProtTrans_token": ProtTrans_token,
        }
        target_info = {"target": target, "sequence": seq}

        return feature, target_info
