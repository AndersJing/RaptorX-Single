#!/usr/bin/env python3
# encoding: utf-8

import os
import torch
import numpy as np
import warnings
from . import all_atom, constants

warnings.filterwarnings("ignore")

res_types = constants.restypes + ["X"]


def gen_BackBoneAtomCoords(
    aatype, preds_list, pred_names, atom_name="CB", default_atom_name="CA"
):
    """

    return the predicted coordinates of backbone atom,
    the coordinates of backbone atom is determined by the predicted Ca coord and predicted orientation

    Returns:
        preds_pos: [num_batch, num_res, 3]
        # has_tom:   [num_batch, num_res]
    """
    assert atom_name in [
        "C",
        "CB",
        "N",
    ], f"atom name {atom_name} should be backbone atom C, CB or N"

    preds_trans = preds_list[pred_names.index("Ca_coord")]
    preds_rots = preds_list[pred_names.index("orient")]

    def to_tensor(x):
        return torch.from_numpy(x).type_as(preds_rots)

    atom37_mask = all_atom.constants.restype_atom37_mask[aatype.int().cpu().numpy()]
    atom37_mask = to_tensor(atom37_mask)

    ref_coord37 = all_atom.constants.ref_coord7_positions[aatype.int().cpu().numpy()]
    ref_coord37 = to_tensor(ref_coord37)
    rot_coord37 = torch.einsum(
        "b n x y, b n a y -> b n a x", preds_rots, ref_coord37
    ) + preds_trans.unsqueeze(-2)

    atom_idx = all_atom.constants.atom_order[atom_name]

    preds_atom = rot_coord37[..., atom_idx, :]
    has_atom = atom37_mask[..., atom_idx][..., None]

    if default_atom_name is not None:
        default_atom_idx = all_atom.constants.atom_order[default_atom_name]
        preds_default_atom = rot_coord37[..., default_atom_idx, :]
        has_default_atom = atom37_mask[..., default_atom_idx][..., None]

        preds_pos = torch.where(
            has_atom > 0, preds_atom, preds_default_atom
        )  # [num_batch, num_res, 3]
        has_pos = torch.where(has_atom > 0, has_atom, has_default_atom).squeeze(
            -1
        )  # [num_batch, num_res]
    else:
        preds_pos = preds_atom
        has_pos = has_atom.squeeze(-1)

    return preds_pos, has_pos


def pred_coords_to_distmap(
    aatype, res_mask, struc_out, struc_out_names, atom_name="CB", default_atom_name="CA"
):
    """convert pred coordinates to distance map for recycling

    Args:
        aatype (tensor: [n_batch, seq_len]): amino acid type (index)
        res_mask (tensor: [n_batch, seq_len]): residue mask
        struc_out (dict): out of structure module
        struc_out_names (list): structure out names
        atom_name (str): target atom
        default_atom_name (str, optional): default atom name if atom_name not exists. Defaults to "CA".
    Returns:
        dist_map: distance map
    """

    aatype[aatype > 20] = 20

    coords, has_pos = gen_BackBoneAtomCoords(
        aatype, struc_out, struc_out_names, atom_name, default_atom_name
    )
    coords = coords.detach()
    res_mask = res_mask * has_pos

    # calc dist map
    dist_map = torch.cdist(coords, coords, p=2)

    # mask
    res_mask = res_mask[:, :, None] * res_mask[:, None, :]
    dist_map.masked_fill_(res_mask == 0, torch.finfo(dist_map.dtype).max)

    return dist_map


def to_array(x):
    return x.detach().cpu().numpy()


def res_1to3(r):
    return constants.restype_1to3.get(res_types[r], "UNK")


def get_pred_LDDT(pLDDT_logit):
    """calc pred LDDT from logit

    Args:
        pLDDT_logit (array): pred LDDT logit

    Returns:
        pLDDT: pred LDDT
    """
    pLDDT_prob = torch.softmax(pLDDT_logit.float(), dim=-1)

    num_res, num_bins = pLDDT_prob.shape
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    pLDDT = (pLDDT_prob * bin_centers).sum(axis=-1)

    return pLDDT


def get_aatype(seq):
    aatype = np.vectorize(
        lambda x: constants.restype_order_with_x.get(x, constants.restype_num)
    )(np.asarray(list(seq)))
    return aatype


def coord14_to_coord37(aa_type, coord14, atom14_mask):
    """
    Args:
        aa_type: [N]
        coord14: [N, 14, 3]
        atom14_mask: [N, 14]
    Returns:
        coord37: [N, 37, 3]
        atom37_mask: [N, 37]
    """

    per_res_idx = constants.restype_atom37_to_atom14[aa_type]
    res_idx = np.tile(
        np.arange(per_res_idx.shape[0])[..., None], (1, per_res_idx.shape[1])
    )

    atom37_pos_mask = constants.restype_atom37_mask[aa_type]
    atom37_mask = atom14_mask[res_idx, per_res_idx]
    atom37_mask = atom37_mask * atom37_pos_mask

    coord37 = coord14[res_idx, per_res_idx]
    coord37 = coord37 * atom37_mask[..., None]

    return coord37, atom37_mask


def to_pdb(aa_type, atom_positions, pred_LDDT, atom_mask):
    """Converts a `Protein` instance to a PDB string.

    Args:
      aa_type: [N]
      atom_positions: [N, 14, 3]/ [N, 37, 3]
      atom_mask: [N, 14]/ [N, 37]

    Returns:
      PDB string.
    """

    assert aa_type.shape[0] == atom_positions.shape[0]
    assert aa_type.shape[0] == atom_mask.shape[0]
    assert atom_positions.shape[1] == atom_mask.shape[1]

    # convert 14 atom representation to 37 atom representation
    if atom_positions.shape[1] == 14:
        atom_positions, atom_mask = coord14_to_coord37(
            aa_type, atom_positions, atom_mask
        )

    # write masked residue:
    atom_mask[
        :,
        [
            constants.atom_order["N"],
            constants.atom_order["CA"],
            constants.atom_order["C"],
            constants.atom_order["O"],
        ],
    ] = 1

    pdb_lines = []
    residue_index = np.arange(len(aa_type)) + 1
    b_factors = pred_LDDT

    if np.any(aa_type > constants.restype_num):
        raise ValueError("Invalid aatypes.")

    pdb_lines.append("MODEL     1")
    atom_index = 1
    chain_id = "A"
    # Add all atom sites.
    for i in range(aa_type.shape[0]):
        res_name_3 = res_1to3(aa_type[i])
        b_factor = b_factors[i]
        for atom_name, pos, mask in zip(
            constants.atom_types, atom_positions[i], atom_mask[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            # Protein supports only C, N, O, S, this works.
            element = atom_name[0]
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_id:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = "TER"
    chain_termination_line = (
        f"{chain_end:<6}{atom_index:>5}      {res_1to3(aa_type[-1]):>3} "
        f"{chain_id:>1}{residue_index[-1]:>4}"
    )
    pdb_lines.append(chain_termination_line)
    pdb_lines.append("ENDMDL")

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def outputs_to_pdb(preds, infos, pred_names, out_dir, outfile_tag):
    preds_trans = preds[pred_names.index("Ca_coord")]
    preds_rots = preds[pred_names.index("orient")]
    preds_angles = preds[pred_names.index("norm_angles")]

    # init as gap
    aa_types = (
        torch.zeros(preds_trans.shape[:2]).type_as(preds_trans).long()
        + constants.restype_num
    )
    for i in range(len(infos["sequence"])):
        aa_type_i = torch.from_numpy(get_aatype(infos["sequence"][i])).type_as(aa_types)
        aa_types[i][: len(aa_type_i)] = aa_type_i

    preds_rg_rots, preds_rg_trans = all_atom.torsions_to_rotran(
        aa_types, preds_rots, preds_trans, preds_angles
    )
    preds_coord14, preds_atom14_mask = all_atom.rg_rotran_to_coord14(
        aa_types, preds_rg_rots, preds_rg_trans
    )

    os.makedirs(out_dir, exist_ok=True)

    num_batch = len(preds_trans)
    pred_LDDT_logits = preds[pred_names.index("pLDDT_logit")].cpu()
    for i in range(num_batch):
        target = infos["target"][i]
        aa_type = aa_types[i]
        pred_coord14 = preds_coord14[i]
        pred_atom14_mask = preds_atom14_mask[i]
        pred_LDDT = get_pred_LDDT(pred_LDDT_logits[i]) * 100

        aa_type, pred_coord14, pred_atom14_mask = map(
            to_array, [aa_type, pred_coord14, pred_atom14_mask]
        )
        pred_coord37, pred_atom37_mask = coord14_to_coord37(
            aa_type, pred_coord14, pred_atom14_mask
        )

        pdb_str = to_pdb(aa_type, pred_coord37, pred_LDDT, pred_atom37_mask)
        pdb_file = os.path.join(out_dir, target + f".{outfile_tag}.pdb")
        with open(pdb_file, "w") as f:
            f.write(pdb_str)
        print(pdb_file)
