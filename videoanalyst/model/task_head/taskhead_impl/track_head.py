# -*- coding: utf-8 -*

# 该文件已品阅完毕

import numpy as np

import torch
import torch.nn as nn

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS, VOS_HEADS

torch.set_printoptions(precision=8)


def get_xy_ctr(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in torch)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = torch.linspace(0., fm_height - 1., fm_height).reshape(
        1, fm_height, 1, 1).repeat(1, 1, fm_width,
                                   1)  # .broadcast([1, fm_height, fm_width, 1])
    x_list = torch.linspace(0., fm_width - 1., fm_width).reshape(
        1, 1, fm_width, 1).repeat(1, fm_height, 1,
                                  1)  # .broadcast([1, fm_height, fm_width, 1])
    xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
    xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = xy_ctr.type(torch.Tensor)
    return xy_ctr


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1.,
                         fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])     # 直接预测左上角离中心的位置
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])     # 直接预测右下角离中心的位置
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


@TRACK_HEADS.register
@VOS_HEADS.register
class DenseboxHead(ModuleBase):
    r"""
    Densebox Head for siamfcpp

    Hyper-parameter
    ---------------
    total_stride: int
        stride in backbone
    score_size: int
        final feature map
    x_size: int
        search image size
    num_conv3x3: int
        number of conv3x3 tiled in head
    head_conv_bn: list
        has_bn flag of conv3x3 in head, list with length of num_conv3x3
    head_width: int
        feature width in head structure
    conv_weight_std: float
        std for conv init
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        x_size=303,
        num_conv3x3=3,
        head_conv_bn=[False, False, True],
        head_width=256,
        conv_weight_std=0.0001,
        input_size_adapt=False,
    )

    def __init__(self):
        super(DenseboxHead, self).__init__()

        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

        self.cls_convs = []
        self.bbox_convs = []

    def forward(self, c_out, r_out, x_size=0, raw_output=False):
        # classification head
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        bbox = r_out

        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)     # 反射机制  注意：cls将会被保存到 self._state["corr_fea"] (self指代这个运作架构的pipeline), 此时变化过程为：shape(cls)=(256, 21, 21) => (256, 19, 19) => (256, 17, 17)
            bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)

        # classification score
        cls_score = self.cls_score_p5(cls)  #todo
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)
        # center-ness score
        ctr_score = self.ctr_score_p5(cls)  #todo
        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)
        # regression
        offsets = self.bbox_offsets_p5(bbox)
        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride     # 这一步又是什么依据？答案：这一步的意义是保证 预测的偏移值 都是正数；并且公式中附带的超参数是可学习可训练的！！！
        if raw_output:      # 原生输出，目的何在？
            return [cls_score, ctr_score, offsets]
        # bbox decoding
        if self._hyper_params["input_size_adapt"] and x_size > 0:   # 如果是自适应尺寸，就做单独计算，其实计算逻辑和固定尺寸完全一样
            score_offset = (x_size - 1 -
                            (offsets.size(-1) - 1) * self.total_stride) // 2
            fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset,
                                   self.total_stride)
            fm_ctr = fm_ctr.to(offsets.device)
        else:
            fm_ctr = self.fm_ctr.to(offsets.device)     # 表示得分图的中心位置
        bbox = get_box(fm_ctr, offsets)

        return [cls_score, ctr_score, bbox, cls]    # 看论文，即可知道，架构图中被红色矩形框框起来的部分就是cls，shape=(256, 17, 17)

    def update_params(self):
        x_size = self._hyper_params["x_size"]
        score_size = self._hyper_params["score_size"]
        total_stride = self._hyper_params["total_stride"]
        score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2      # 得分偏移量：—1是中心对称计算所要求的
        self._hyper_params["score_offset"] = score_offset

        self.score_size = self._hyper_params["score_size"]
        self.total_stride = self._hyper_params["total_stride"]
        self.score_offset = self._hyper_params["score_offset"]
        ctr = get_xy_ctr_np(self.score_size, self.score_offset,
                            self.total_stride)
        self.fm_ctr = ctr
        self.fm_ctr.require_grad = False        # 这是为何？ 默认是test模式？

        # 可见， 该函数被调用后，头部网络的构建，参数的初始化均已完成；而该函数的调用是在builder.py中完成的
        self._make_conv3x3()
        self._make_conv_output()
        self._initialize_conv()

    def _make_conv3x3(self):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        head_conv_bn = self._hyper_params['head_conv_bn']
        head_width = self._hyper_params['head_width']
        self.cls_conv3x3_list = []
        self.bbox_conv3x3_list = []
        for i in range(num_conv3x3):
            cls_conv3x3 = conv_bn_relu(head_width,
                                       head_width,
                                       stride=1,
                                       kszie=3,
                                       pad=0,       # 由于padding=0，可知feature map每次卷积后都减小2个像素
                                       has_bn=head_conv_bn[i])

            bbox_conv3x3 = conv_bn_relu(head_width,
                                        head_width,
                                        stride=1,
                                        kszie=3,
                                        pad=0,
                                        has_bn=head_conv_bn[i])
            setattr(self, 'cls_p5_conv%d' % (i + 1), cls_conv3x3)
            setattr(self, 'bbox_p5_conv%d' % (i + 1), bbox_conv3x3)
            self.cls_conv3x3_list.append(cls_conv3x3)
            self.bbox_conv3x3_list.append(bbox_conv3x3)

    def _make_conv_output(self):
        head_width = self._hyper_params['head_width']
        self.cls_score_p5 = conv_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.ctr_score_p5 = conv_bn_relu(head_width,        # 中心度得分
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.bbox_offsets_p5 = conv_bn_relu(head_width,
                                            4,
                                            stride=1,
                                            kszie=1,
                                            pad=0,
                                            has_relu=False)

    def _initialize_conv(self, ):
        # Please refer to: https://ai.plainenglish.io/weight-initialization-in-neural-network-9b3935192f6
        num_conv3x3 = self._hyper_params['num_conv3x3']
        conv_weight_std = self._hyper_params['conv_weight_std']

        # initialze head
        conv_list = []
        for i in range(num_conv3x3):
            conv_list.append(getattr(self, 'cls_p5_conv%d' % (i + 1)).conv)
            conv_list.append(getattr(self, 'bbox_p5_conv%d' % (i + 1)).conv)

        conv_list.append(self.cls_score_p5.conv)
        conv_list.append(self.ctr_score_p5.conv)
        conv_list.append(self.bbox_offsets_p5.conv)
        conv_classifier = [self.cls_score_p5.conv]
        assert all(elem in conv_list for elem in conv_classifier)

        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(
                conv.weight, std=conv_weight_std)  # conv_weight_std = 0.0001
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            if conv in conv_classifier:
                torch.nn.init.constant_(conv.bias, torch.tensor(bv))    # 这是为何？ 什么初始化逻辑？
            else:
                # torch.nn.init.constant_(conv.bias, 0)  # from PyTorch default implementation
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)
