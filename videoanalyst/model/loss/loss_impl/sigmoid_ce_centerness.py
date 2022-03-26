# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F

from ...module_base import ModuleBase
# from videoanalyst.model.module_base import ModuleBase

from ..loss_base import TRACK_LOSSES
# from videoanalyst.model.loss.loss_base import TRACK_LOSSES

from .utils import SafeLog
# from videoanalyst.model.loss.loss_impl.utils import SafeLog

eps = np.finfo(np.float32).tiny


@TRACK_LOSSES.register
class SigmoidCrossEntropyCenterness(ModuleBase):

    default_hyper_params = dict(
        name="centerness",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self, background=0, ignore_label=-1):
        super(SigmoidCrossEntropyCenterness, self).__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def update_params(self, ):
        self.background = self._hyper_params["background"]
        self.ignore_label = self._hyper_params["ignore_label"]
        self.weight = self._hyper_params["weight"]

    def forward(self, pred_data, target_data):
        r"""
        Center-ness loss
        Computation technique originated from this implementation:
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        
        P.S. previous implementation can be found at the commit 232141cdc5ac94602c28765c9cf173789da7415e

        Arguments
        ---------
        pred: torch.Tensor
            center-ness logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        pred = pred_data["ctr_pred"]
        label = target_data["ctr_gt"]
        mask = (~(label == self.background)).type(torch.Tensor).to(pred.device)     # 这行代码是在过滤背景，因为有些label确实是0，特征像素点里gt-box太远了，负值直接降截变0
        loss = F.binary_cross_entropy_with_logits(pred, label,
                                                  reduction="none") * mask      # 实乃我之偏见，并非分类才可以用binary 交叉熵损失，不过要求label之值between[0,1]
        # suppress loss residual (original vers.)
        loss_residual = F.binary_cross_entropy(label, label,
                                               reduction="none") * mask         # 消除损失残差，这一步切不可丢弃
        loss = loss - loss_residual.detach()

        loss = loss.sum() / torch.max(mask.sum(),
                                      self.t_one) * self._hyper_params["weight"]
        extra = dict()

        return loss, extra


if __name__ == '__main__':
    B = 16
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.int8)
    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    # criterion_cls = SigmoidCrossEntropyRetina()
    # loss_cls = criterion_cls(pred_cls, gt_cls)

    criterion_ctr = SigmoidCrossEntropyCenterness()
    criterion_ctr._hyper_params = SigmoidCrossEntropyCenterness.default_hyper_params
    criterion_ctr.update_params()
    # loss_ctr = criterion_ctr(pred_ctr, gt_ctr, gt_cls)
    loss_ctr = criterion_ctr(pred_ctr, gt_ctr)

    # criterion_reg = IOULoss()
    # loss_reg = criterion_reg(pred_reg, gt_reg, gt_cls)

    from IPython import embed
    embed()
