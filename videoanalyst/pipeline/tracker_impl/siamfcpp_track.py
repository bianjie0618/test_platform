# -*- coding: utf-8 -*

from copy import deepcopy

import numpy as np

import torch
from torch import cosine_similarity

from videoanalyst.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from videoanalyst.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppTracker(PipelineBase):
    r"""
    Basic SiamFC++ tracker

    Hyper-parameters
    ----------------
        total_stride: int
            stride in backbone     # 主干网总步长
        context_amount: float
            factor controlling the image patch cropping range. Set to 0.5 by convention.
        test_lr: float
            factor controlling target size updating speed
        penalty_k: float
            factor controlling the penalization on target size (scale/ratio) change
        window_influence: float
            factor controlling spatial windowing on scores
        windowing: str
            windowing type. Currently support: "cosine"
        z_size: int
            template image size
        x_size: int
            search image size
        num_conv3x3: int
            number of conv3x3 tiled in head
        min_w: float
            minimum width
        min_h: float
            minimum height
        phase_init: str 
            phase name for template feature extraction      # 阶段名称
        phase_track: str
            phase name for target search
        corr_fea_output: bool
            whether output corr feature

    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        score_offset=87,        # 这一步是可以计算出来的
        context_amount=0.5,
        initial_context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.04,
        window_influence=0.21,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",   # 模板特征提取阶段
        phase_track="track",    # 后续帧跟踪阶段
        corr_fea_output=False,
    )

    def __init__(self, *args, **kwargs):
        super(SiamFCppTracker, self).__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def feature(self, im: np.array, target_pos, target_sz, avg_chans=None):
        """Extract feature

        Parameters
        ----------
        im : np.array
            initial frame
        target_pos : 
            target position (x, y)
        target_sz : [type]
            target size (w, h)
        avg_chans : [type], optional
            channel mean values, (B, G, R), by default None
        
        Returns
        -------
        [type]
            [description]
        """
        if avg_chans is None:
            avg_chans = np.mean(im, axis=(0, 1))        # 在联合维度上求取平均值

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            data = imarray_to_tensor(im_z_crop).to(self.device)
            features = self._model(data, phase=phase)
        # 新增代码
        self._state['shared_feature'] = features[-1]
        features = features[:2]

        return features, im_z_crop, avg_chans

    def similarity_score(self, res_pos, res_sz, avg_chans=None):
        
        tracked_img = self._state['filtered_img']
        if avg_chans is None:
            avg_chans = np.mean(tracked_img, axis=(0, 1))        # 在联合维度上求取平均值

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, _ = get_crop(
            tracked_img,
            res_pos,
            res_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        phase = 'shared_feature'
        with torch.no_grad():
            data = imarray_to_tensor(im_z_crop).to(self.device)
            new_feature = self._model(data, phase=phase)
            self._state['new_feature'] = new_feature
        cosine_sim = cosine_similarity(torch.flatten(new_feature), torch.flatten(self._state['shared_feature']), dim=0)
        return cosine_sim
    
    def put_filtered_img(self, filtered_img):
        self._state['filtered_img'] = filtered_img

    def template_fusing(self):
        new_c_z_k = self._model.c_z_k(self._state['shared_feature']) * 0.2 + self._state['features'][0] * 0.8
        new_r_z_k = self._model.r_z_k(self._state['shared_feature']) * 0.2 + self._state['features'][1] * 0.8
        self._state['features'] = [new_c_z_k, new_r_z_k]

    def set_context_amount(self, num):
        self._hyper_params['context_amount'] = num
    
    def get_initial_context_amount(self):
        return self._hyper_params['initial_context_amount']

    def reset_context_amount(self):
        self._hyper_params['context_amount'] = self._hyper_params['initial_context_amount']


    def init(self, im, state):
        r"""Initialize tracker      # 初始化追踪器
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        
        Arguments
        ---------
        im : np.array
            initial frame image
        state
            target state on initial frame (bbox in case of SOT), format: xywh
        """
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        # 为什么特征会出现两个元素，分别有什么用处？？？上下两个分支么？？？
        features, im_z_crop, avg_chans = self.feature(im, target_pos, target_sz)

        score_size = self._hyper_params['score_size']
        if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
        elif self._hyper_params['windowing'] == 'uniform':
            window = np.ones((score_size, score_size))
        else:
            window = np.ones((score_size, score_size))

        self._state['z_crop'] = im_z_crop
        self._state['avg_chans'] = avg_chans
        self._state['features'] = features
        self._state['window'] = window      # 加窗矩阵
        # self.state['target_pos'] = target_pos
        # self.state['target_sz'] = target_sz
        self._state['state'] = (target_pos, target_sz)

    def get_avg_chans(self):
        return self._state['avg_chans']
    
    def scale(self, target_sz, z_size, context_amount):
        wc = target_sz[0] + context_amount * sum(target_sz)
        hc = target_sz[1] + context_amount * sum(target_sz)
        s_crop = np.sqrt(wc * hc)
        scale = z_size / s_crop
        return scale

    def track(self,
              im_x,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        phase_track = self._hyper_params['phase_track']
        im_x_crop, scale_x = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        self._state["scale_x"] = deepcopy(scale_x)      # 有必要deepcopy么？
        with torch.no_grad():
            score, box, cls, ctr, extra = self._model(      # shape(score) = ([1, 289, 1]), shape(box) = ([1, 289, 4]), shape(cls) = ([1, 289, 1]), shape(ctr) = ([1, 289, 1])
                imarray_to_tensor(im_x_crop).to(self.device),   
                *features,
                phase=phase_track)      # 这里返回的score是综合了分类和状态预测得分后的综合性分类得分
        if self._hyper_params["corr_fea_output"]:
            self._state["corr_fea"] = extra["corr_fea"]

        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
        box_wh = xyxy2cxywh(box)

        # score post-processing ####################################后处理阶段品读待续##############
        if self._hyper_params['initial_context_amount'] != self._hyper_params['context_amount']:
            scale_x = self.scale(target_sz, z_size, self._hyper_params['initial_context_amount'])
        
        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, target_sz, scale_x)
        # box post-processing
        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, target_pos, target_sz, scale_x,
            x_size, penalty)

        if self.debug:
            box = self._cvt_box_crop2frame(box_wh, target_pos, x_size, scale_x)

        # restrict new_target_pos & new_target_sz
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)

        # record basic mid-level info
        self._state['x_crop'] = im_x_crop       # Affine Transformation + Cropping 后的结果就是im_x_crop
        bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop
        # record optional mid-level info
        if update_state:
            self._state['score'] = score    # 综合得分
            self._state['pscore'] = pscore[best_pscore_id]
            self._state['all_box'] = box
            self._state['cls'] = cls    # 分类得分
            self._state['ctr'] = ctr    # 预测框质量评估得分

        return new_target_pos, new_target_sz

    def set_state(self, state):
        self._state["state"] = state

    def get_track_score(self):
        return float(self._state["pscore"])

    def update(self, im, state=None):
        """ Perform tracking on current frame
            Accept provided target state prior on current frame
            e.g. search the target in another video sequence simultanously

        Arguments
        ---------
        im : np.array
            current frame image
        state
            provided target state prior (bbox in case of SOT), format: xywh
        """
        # use prediction on the last frame as target state prior
        if state is None:
            target_pos_prior, target_sz_prior = self._state['state']
        # use provided bbox as target state prior
        else:
            rect = state  # bbox in xywh format is given for initialization in case of tracking
            box = xywh2cxywh(rect).reshape(4)
            target_pos_prior, target_sz_prior = box[:2], box[2:]
        features = self._state['features']

        # forward inference to estimate new state
        target_pos, target_sz = self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        # self.state['target_pos'], self.state['target_sz'] = target_pos, target_sz
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        if self._hyper_params["corr_fea_output"]:
            return target_pos, target_sz, self._state["corr_fea"]
        return track_rect

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)    # 之前的状态
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = self._hyper_params['penalty_k']
        target_sz_in_crop = target_sz * scale_x         # 计算在裁切并缩放后得到的patch上对应的尺度(上一次的目标尺度)
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty    尺度变化惩罚
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty    比例变化惩罚
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)      # 惩罚公式(综合了scale和ratio)
        pscore = penalty * score

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = self._hyper_params['window_influence']       # 加窗矩阵的影响比例
        pscore = pscore * (
            1 - window_influence) + self._state['window'] * window_influence    # 最终得分综合了"加窗影响"和"尺度横纵比变化"影响
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)      # 计算在裁切的图像块中所做预测(缩放到x_size(303)前)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        # 下面是关于bounding box 如何更新的实现细节
        test_lr = self._hyper_params['test_lr']
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x   # 这里的res可以理解为restored，即被恢复的原图上的中心坐标和长宽
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self._state['im_w'], target_pos[0]))     # 不超过图片的左右边界
        target_pos[1] = max(0, min(self._state['im_h'], target_pos[1]))     # 不超过图像的上下边界
        target_sz[0] = max(self._hyper_params['min_w'],
                           min(self._state['im_w'], target_sz[0]))          # bbox的w不超过原始图像的宽，其w不小于bbox所要求的最小w
        target_sz[1] = max(self._hyper_params['min_h'],
                           min(self._state['im_h'], target_sz[1]))

        return target_pos, target_sz

    def _cvt_box_crop2frame(self, box_in_crop, target_pos, scale_x, x_size):
        r"""
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        # 当前帧的搜索图像块是以上一帧跟踪结果bbox的中心为中心，进行放射变换和裁切得到的
        x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                               2) / scale_x
        y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                               2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame
