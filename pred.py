#!/usr/bin/env python3
# encoding: utf-8


import os
import time
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

from Config import CONFIGS
from Dataset import SeqDataset
from models.MainModel import MainModel
from utils.Utils import outputs_to_pdb

import logging

FORMAT = "[%(filename)s:%(lineno)s %(funcName)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]


class Attn4StrucPred:
    def __init__(self, configs, args):
        super().__init__()

        self.args = args
        self.device = args.device
        self.pred_keys = configs["PREDICTION"]["pred_keys"]
        configs["EMBEDDING_MODULE"]["plm_config"]["plm_models"] = args.plm_models
        configs["EMBEDDING_MODULE"]["plm_config"]["model_config"][
            "device"
        ] = args.device

        # model
        self.model = MainModel(configs).to(self.device)

        # load params
        state_dict = torch.load(args.param, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x, recycling, recycle_data, *args, **kwargs):
        x["recycling"] = recycling
        x["recycle_data"] = recycle_data

        preds, recycle_data = self.model(x, *args, **kwargs)

        return preds, recycle_data

    def move_to_device(self, batch_data):
        for k in batch_data[0]:
            if isinstance(batch_data[0][k], torch.Tensor):
                batch_data[0][k] = batch_data[0][k].to(self.device)

        return batch_data

    def pred_step(self, batch_data, batch_idx):
        feature, sample_info = self.move_to_device(batch_data)

        start_time = time.time()

        target = sample_info["target"][0]

        recycle_data = None
        for i in range(self.args.n_cycle):
            preds, recycle_data = self.forward(
                feature, i < self.args.n_cycle, recycle_data
            )

        # save pdb
        outputs_to_pdb(
            preds, sample_info, self.pred_keys, self.args.out_dir, self.args.outfile_tag
        )

        print(f"running time: {target}", time.time() - start_time)

    def pred(self, data_loader):
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                self.pred_step(sample, i)

        print("total running time:", time.time() - start_time)


def update_args(args):
    assert args.n_cycle > 0, "n_cycle must >=1."

    args.device = (
        torch.device("cuda:%s" % (args.device_id))
        if args.device_id >= 0
        else torch.device("cpu")
    )

    # plm_models
    plm_models = []
    if args.param.find("ESM1b") > -1:
        plm_models += ["ESM1b"]
        os.environ["ESM1b_param"] = args.plm_param_dir + "esm1b_t33_650M_UR50S.pt"
        os.environ["ESM_param"] = os.environ["ESM1b_param"]
    if args.param.find("ESM1v") > -1:
        plm_models += ["ESM1v"]
        os.environ["ESM1v_param"] = args.plm_param_dir + "esm1v_t33_650M_UR90S_1.pt"
        os.environ["ESM_param"] = os.environ["ESM1v_param"]
    if args.param.find("ProtTrans") > -1:
        plm_models += ["ProtTrans"]
        os.environ["ProtTrans_param"] = args.plm_param_dir + "prot_t5_xl_uniref50"

    print(plm_models)
    args.plm_models = plm_models
    args.outfile_tag = os.path.basename(args.param).split(".")[0]

    os.makedirs(args.out_dir, exist_ok=True)

    return args


def main(args):
    args = update_args(args)

    data_set = SeqDataset(args.fasta_path, args.plm_models)
    data_loader = DataLoader(data_set, pin_memory=False, num_workers=args.n_worker)

    mAttn4StrucPred = Attn4StrucPred(CONFIGS, args)
    mAttn4StrucPred.pred(data_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("fasta_path", type=str, help="fasta file or dir")
    parser.add_argument("param", type=str, help="param file path")

    parser.add_argument("--out_dir", type=str, default="output/", help="output dir.")
    parser.add_argument(
        "--plm_param_dir",
        type=str,
        default="params/",
        help="param path for PLM models (ESM1b, ESM1v and ProtTrans)",
    )
    parser.add_argument(
        "--device_id", type=int, default=-1, help="device id (-1 for CPU, >=0 for GPU)."
    )
    parser.add_argument("--n_cycle", type=int, default=4, help="cycle time.")
    parser.add_argument(
        "--n_worker", type=int, default=0, help="DataLoader num_workers."
    )
    args = parser.parse_args()
    print(args)

    main(args)
