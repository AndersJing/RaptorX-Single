import torch
from torch import einsum
from einops import repeat
import numpy as np
from . import constants


def torsions_to_rotran(aatype, res_rots, res_trans, sincos):
    """

    Args:
        aatype: torch.Tensor [..., N]
        res_rots: torch.Tensor [..., N, 3, 3]
        res_trans: torch.Tensor [..., N, 1, 3]
        sincos: torch.Tensor [..., N, 7, 2]

    Returns:
        rg_rots: torch.Tensor [..., N, 8, 3, 3]
        rg_trans: torch.Tensor [..., N, 8, 1, 3]
    """

    data_in_shape = aatype.shape

    aatype = aatype.long().reshape(-1)
    np_aatype = aatype.cpu().numpy()
    (num_res,) = aatype.shape
    res_rots = res_rots.reshape((num_res, 3, 3))
    res_trans = res_trans.reshape((num_res, 1, 3))
    sincos = sincos.reshape((num_res, 7, 2))

    (num_residues,) = aatype.shape

    m = constants.restype_rigid_group_default_frame[np_aatype]
    m = torch.from_numpy(m).type_as(res_rots)

    # rotatino ans transition for reference rigid group
    ref_rots = m[..., :3, :3]
    ref_trans = m[..., :3, 3].unsqueeze(-2)

    # insert zero rotation for backbone group.
    sin = torch.cat(
        [torch.zeros([num_residues, 1]).type_as(sincos), sincos[..., 0]], dim=-1
    )
    cos = torch.cat(
        [torch.ones([num_residues, 1]).type_as(sincos), sincos[..., 1]], dim=-1
    )

    ones = torch.ones_like(sin)
    zeros = torch.zeros_like(cos)

    # rotation along x axis
    pred_rg_rots = torch.zeros((num_residues, 8, 3, 3)).type_as(res_rots)
    pred_rg_rots[..., 0, 0], pred_rg_rots[..., 0, 1], pred_rg_rots[..., 0, 2] = (
        ones,
        zeros,
        zeros,
    )
    pred_rg_rots[..., 1, 0], pred_rg_rots[..., 1, 1], pred_rg_rots[..., 1, 2] = (
        zeros,
        cos,
        -sin,
    )
    pred_rg_rots[..., 2, 0], pred_rg_rots[..., 2, 1], pred_rg_rots[..., 2, 2] = (
        zeros,
        sin,
        cos,
    )

    rg_chi_rots = einsum("n g x y, n g y z -> n g x z", ref_rots, pred_rg_rots)

    chi1_rots = rg_chi_rots[:, 4]
    chi2_rots = rg_chi_rots[:, 5]
    chi3_rots = rg_chi_rots[:, 6]
    chi4_rots = rg_chi_rots[:, 7]

    chi1_trans = ref_trans[:, 4]
    chi2_trans = ref_trans[:, 5]
    chi3_trans = ref_trans[:, 6]
    chi4_trans = ref_trans[:, 7]

    chi1_rots_to_backb = chi1_rots
    chi1_trans_to_backb = chi1_trans

    chi2_rots_to_backb = einsum(" n x y, n y z -> n x z", chi1_rots_to_backb, chi2_rots)
    chi2_trans_to_backb = (
        einsum(" n x y, n z y -> n z x", chi1_rots_to_backb, chi2_trans)
        + chi1_trans_to_backb
    )

    chi3_rots_to_backb = einsum(" n x y, n y z -> n x z", chi2_rots_to_backb, chi3_rots)
    chi3_trans_to_backb = (
        einsum(" n x y, n z y -> n z x", chi2_rots_to_backb, chi3_trans)
        + chi2_trans_to_backb
    )

    chi4_rots_to_backb = einsum(" n x y, n y z -> n x z", chi3_rots_to_backb, chi4_rots)
    chi4_trans_to_backb = (
        einsum(" n x y, n z y -> n z x", chi3_rots_to_backb, chi4_trans)
        + chi3_trans_to_backb
    )

    _cat = lambda a, b, c, d: torch.cat(
        [a[:, :5], b[:, None], c[:, None], d[:, None]], dim=1
    )

    rg_rots_to_backb = _cat(
        *(rg_chi_rots, chi2_rots_to_backb, chi3_rots_to_backb, chi4_rots_to_backb)
    )

    rg_trans_to_backb = _cat(
        *(ref_trans, chi2_trans_to_backb, chi3_trans_to_backb, chi4_trans_to_backb)
    )

    # print('res rots ', res_rots.shape)
    # print('rg_rots_to_backb ', rg_rots_to_backb.shape)
    rg_rots = einsum("n x y , n g y z -> n g x z", res_rots, rg_rots_to_backb)
    rg_trans = einsum(
        "n x y, n g z y -> n g z x", res_rots, rg_trans_to_backb
    ) + repeat(res_trans, "n d c -> n g d c", g=8)

    rg_rots = rg_rots.reshape(data_in_shape + (8, 3, 3))
    rg_trans = rg_trans.reshape(data_in_shape + (8, 1, 3))

    data_dict = {
        "rg_rots": rg_rots,
        "rg_trans": rg_trans,
        "sin_angles": sin,
        "cos_angles": cos,
        "rg_chi_rots": rg_chi_rots,
        "rg_chi_trans": ref_trans,
        "ref_rots": ref_rots,
        "ref_trans": ref_trans,
    }

    return rg_rots, rg_trans  # , data_dict


def rg_rotran_to_coord14(aatype, rg_rots, rg_trans):

    """

    aatype: torch.Tensor [...,N]
    rg_rots: torch.Tensor [..., N, 8, 3, 3]
    rg_trans: torch.Tensor [...,N, 8, 1, 3]

    return:
        pred_position: [..., N,  14, 3]
    """

    data_in_shape = aatype.shape
    aatype = aatype.long().reshape(-1)
    np_aatype = aatype.cpu().numpy()

    (num_res,) = aatype.shape
    rg_rots = rg_rots.reshape((num_res, 8, 3, 3))
    rg_trans = rg_trans.reshape((num_res, 8, 1, 3))

    residx_to_group_idx = constants.restype_atom14_to_rigid_group[np_aatype]
    group_mask = np.eye(8)[residx_to_group_idx]
    group_mask = torch.from_numpy(group_mask).type_as(rg_rots)

    atom14_rots = (rg_rots[:, None, :] * group_mask[..., None, None]).sum(dim=2)
    atom14_trans = (rg_trans[:, None, :] * group_mask[..., None, None]).sum(dim=2)

    ref_position = constants.restype_atom14_rigid_group_positions[np_aatype]
    ref_position = torch.from_numpy(ref_position).type_as(atom14_rots)
    pred_position = (
        einsum("n g x y, n g y -> n g x", atom14_rots, ref_position).unsqueeze(-2)
        + atom14_trans
    )

    atom14_mask = constants.restype_atom14_mask[np_aatype]
    atom14_mask = torch.from_numpy(atom14_mask).type_as(pred_position)

    pred_position = pred_position * atom14_mask[..., None, None]
    pred_position = pred_position.reshape(data_in_shape + (14, 3))

    atom14_mask = atom14_mask.reshape(data_in_shape + (14,))
    return pred_position, atom14_mask
