# -*- coding: utf-8 -*

from copy import deepcopy

import cv2
import numpy as np

import torch
from torch import cosine_similarity

from videoanalyst.pipeline.pipeline_base import VOS_PIPELINES, PipelineBase
from videoanalyst.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh)


# ============================== Tracker definition ============================== #
@VOS_PIPELINES.register
class StateAwareTracker_with_siamfc(PipelineBase):
    r"""
    Basic State-Aware Tracker for vos

    Hyper-parameters
    ----------------
        z_size: int
            template image size
        save_patch: bool
            save and visualize the predicted mask for saliency image patch
        mask_pred_thresh: float
            threshold to binarize predicted mask for final decision
        mask_filter_thresh: float
            threshold to binarize predicted mask for filter the patch of global modeling loop
        GMP_image_size: int
            image size of the input of global modeling loop
        saliency_image_size: int
            image size of saliency image
        saliency_image_field: int
            corresponding fields of saliency image
        cropping_strategy: bool
            use cropping strategy or not
        state_score_thresh: float
            threshhold for state score
        global_modeling: bool
            use global modeling loop or not
        seg_ema_u: float
            hyper-parameter u for global feature updating
        seg_ema_s: float
            hyper-parameter s for global feature updating
        track_failed_score_th: float
            if tracker score < th, then the mask will be ignored
        update_global_fea_th: float
            if state score > th, the global fea will be updated 

    """
    default_hyper_params = dict(
        z_size=127,
        save_patch=True,
        mask_pred_thresh=0.6,
        mask_filter_thresh=0.6,
        GMP_image_size=129,
        saliency_image_size=257,
        saliency_image_field=129,
        cropping_strategy=True,        # 尝试下不用裁切转换策略---------------
        conf_score_thresh=0.85,         # 分割置信度阈值
        state_score_thresh=0.85,         # 注意该值要小于 siamfc_template_update_thresh_1_state_score
        global_modeling=True,
        seg_ema_u=0.5,
        seg_ema_s=0.6,
        context_amount=0.5,
        mask_rect_lr=0.65,
        track_failed_score_th=0.20,              # 服务于 siamfc 相似度判定阈值
        siamfc_template_update_thresh_1_state_score = 0.95,   # 更新siamfc 模板阈值
        siamfc_template_update_thresh_1_cosine_sim = 0.80,
        track_failed_cosine_similarity_th = 0.20,
        update_global_fea_th=0.85,
    )

    def __init__(self, segmenter, tracker):

        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._segmenter = segmenter
        self._tracker = tracker     # 这里追加tracker，我们只需要在这里添加SiamFC即可

        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._segmenter, self._tracker)

    def set_model(self, segmenter, tracker):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._segmenter = segmenter.to(self.device)
        self._segmenter.eval()
        self._tracker.set_device(self.device)

    def set_device(self, device):
        self.device = device
        self._segmenter = self._segmenter.to(device)
        self._tracker.set_device(self.device)
    
    def init(self, im, state, benchmark_test = True, init_mask = None):
        if benchmark_test:
            pass
        else:
            assert init_mask is not None, 'init_mask error'
            self._init(im, state, init_mask)

    def _init(self, im, state, init_mask):
        """
        initialize the whole pipeline :
        tracker init => global modeling loop init

        :param im: init frame
        :param state: bbox in xywh format
        :param init_mask: binary mask of target object in shape (h,w)
        """

        #========== SiamFC++ init ==============
        self._tracker.init(im, state)
        avg_chans = self._tracker.get_avg_chans()
        self._state['avg_chans'] = avg_chans

        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]
        self._state['state'] = (target_pos, target_sz)
        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # ========== Global Modeling Loop init ==============
        init_image, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],
            x_size=self._hyper_params["GMP_image_size"],
            avg_chans=avg_chans,
            context_amount=self._hyper_params["context_amount"],
            func_get_subwindow=get_subwindow_tracking,
        )
        init_mask_c3 = np.stack([init_mask, init_mask, init_mask],
                                -1).astype(np.uint8)
        init_mask_crop_c3, _ = get_crop(
            init_mask_c3,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],
            x_size=self._hyper_params["GMP_image_size"],
            avg_chans=avg_chans * 0,
            context_amount=self._hyper_params["context_amount"],
            func_get_subwindow=get_subwindow_tracking,
        )
        init_mask_crop = init_mask_crop_c3[:, :, 0]
        init_mask_crop = (init_mask_crop >
                          self._hyper_params['mask_filter_thresh']).astype(
                              np.uint8)
        init_mask_crop = np.expand_dims(init_mask_crop,
                                        axis=-1)  #shape: (129,129,1)
        filtered_image = init_mask_crop * init_image
        self._state['filtered_image'] = filtered_image  #shape: (129,129,3)

        with torch.no_grad():
            deep_feature = self._segmenter(imarray_to_tensor(filtered_image).to(
                self.device),
                                           phase='global_feature')[0]

        self._state['seg_init_feature'] = deep_feature  #shape : (1,256,5,5)
        self._state['seg_global_feature'] = deep_feature    #   对应论文中的G_t
        self._state['gml_feature'] = deep_feature
        self._state['conf_score'] = 1       # 初始的置信度得分是1，这是必然的

    # 全局建模回路
    def global_modeling(self):
        """
        always runs after seg4vos, takes newly predicted filtered image,
        extracts high-level feature and updates the global feature based on confidence score

        """
        filtered_image = self._state['filtered_image']  # shape: (129,129,3)
        with torch.no_grad():
            deep_feature = self._segmenter(imarray_to_tensor(filtered_image).to(
                self.device),
                                           phase='global_feature')[0]

        seg_global_feature = self._state['seg_global_feature']
        seg_init_feature = self._state['seg_init_feature']
        u = self._hyper_params['seg_ema_u']
        s = self._hyper_params['seg_ema_s']
        conf_score = self._state['conf_score']

        u = u * conf_score
        seg_global_feature = seg_global_feature * (1 - u) + deep_feature * u
        gml_feature = seg_global_feature * s + seg_init_feature * (1 - s)       # 全局建模回路所用的抽象特征始终都给初始特征留有一席之地，这是合理的，防止跟丢或跟踪退化情况的出现

        self._state['seg_global_feature'] = seg_global_feature  # 后面的帧没有初始特征的影子
        self._state['gml_feature'] = gml_feature                # 始终带有初始特征

    def joint_segmentation(self, im_x, target_pos, target_sz, corr_feature,
                           gml_feature, **kwargs):
        r"""
        segment the current frame for VOS
        crop image => segmentation =>  params updation

        :param im_x: current image
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :param corr_feature: correlated feature produced by siamese encoder
        :param gml_feature: global feature produced by gloabl modeling loop
        :return: pred_mask  mask prediction in the patch of saliency image
        :return: pred_mask_b binary mask prediction in the patch of saliency image
        """

        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        # crop image for saliency encoder
        saliency_image, scale_seg = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],        # 127
            output_size=self._hyper_params["saliency_image_size"],  # 257
            x_size=self._hyper_params["saliency_image_field"],      #  129
            avg_chans=avg_chans,
            context_amount=self._hyper_params["context_amount"],
            func_get_subwindow=get_subwindow_tracking,
        )
        self._state["scale_x"] = scale_seg
        # mask prediction
        pred_mask = self._segmenter(imarray_to_tensor(saliency_image).to(
            self.device),
                                    corr_feature,
                                    gml_feature,
                                    phase='segment')[0]  #tensor(1,1,257,257)

        pred_mask = tensor_to_numpy(pred_mask[0]).transpose(
            (1, 2, 0))  #np (257,257,1)

        # post processing
        mask_filter = (pred_mask >
                       self._hyper_params['mask_filter_thresh']).astype(
                           np.uint8)
        pred_mask_b = (pred_mask >
                       self._hyper_params['mask_pred_thresh']).astype(np.uint8)

        if self._hyper_params['save_patch']:
            mask_red = np.zeros_like(saliency_image)
            mask_red[:, :, 0] = mask_filter[:, :, 0] * 255
            masked_image = saliency_image * 0.5 + mask_red * 0.5
            self._state['patch_prediction'] = masked_image          # save_patch的作用是啥呢？

        filtered_image = saliency_image * mask_filter
        filtered_image = cv2.resize(filtered_image,
                                    (self._hyper_params["GMP_image_size"],
                                     self._hyper_params["GMP_image_size"]))     # 257 是被放大的尺寸，现在要缩放回去到129
        self._state['filtered_image'] = filtered_image      # 用于全局建模回路提取特征

        if pred_mask_b.sum() > 0:
            conf_score = (pred_mask * pred_mask_b).sum() / pred_mask_b.sum()    # 置信度得分
        else:
            conf_score = 0
        self._state['conf_score'] = conf_score
        mask_in_full_image = self._mask_back(
            pred_mask,
            size=self._hyper_params["saliency_image_size"],
            region=self._hyper_params["saliency_image_field"])      # 反推出原图上的mask
        self._state['mask_in_full_image'] = mask_in_full_image

        # 提取整张背景干净的图片
        full_image_mask_filter = (mask_in_full_image >
                       self._hyper_params['mask_filter_thresh']).astype(
                           np.uint8)[:, :, np.newaxis]
        # self._state['full_image_mask_filter'] = full_image_mask_filter
        full_filtered_image = im_x * full_image_mask_filter
        self._tracker.state['filtered_image'] = full_filtered_image

        if self._tracker.get_track_score(                       # 我也需要在SiamFC上增益一个相似度得分接口！！！
        ) < self._hyper_params["track_failed_score_th"]:
            self._state['mask_in_full_image'] *= 0          # 传统的跟踪器也有判断，也要参合进来，将自己的最高预测得分作为分割器分割结果是否有效的依据
        return pred_mask, pred_mask_b

    def get_global_box_from_masks(self, cnts):
        boxes = np.zeros((len(cnts), 4))
        for i, cnt in enumerate(cnts):
            rect = cv2.boundingRect(cnt.reshape(-1, 2)) # x1,y1, w, h
            boxes[i] = rect
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        global_box = [
            np.amin(boxes[:, 0]),       # 这几步操作完全是为了找到能覆盖所有mask的最大矩形框，故而称为global box，仅有一个
            np.amin(boxes[:, 1]),
            np.amax(boxes[:, 2]),
            np.amax(boxes[:, 3])
        ]
        global_box = np.array(global_box)
        global_box[2:] = global_box[2:] - global_box[:2]        # 
        return global_box

    def cropping_strategy(self, im, p_mask_b):
        r"""
        swithes the bbox prediction strategy based on the estimation of predicted mask.
        returns newly predicted target position and size

        :param p_mask_b: binary mask prediction in the patch of saliency image
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :return: new_target_pos, new_target_sz
        """

        # **********************
        # track_pos, track_sz = self._tracker.update(im)
        similarity_score = self._tracker.get_track_score()  # 余弦相似度，[-1, 1]
        # #***********************

        # new_target_pos, new_target_sz = self._state["state"]        # 这一句貌似有些多余
        conf_score = self._state['conf_score']
        contours, _ = cv2.findContours(p_mask_b, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        self._state["track_score"] = similarity_score

        # similarity_score > self._hyper_params['track_failed_score_th'] and
        if conf_score >= self._hyper_params['conf_score_thresh'] and len(contours) != 0 and np.max(cnt_area) >= 800 and \
            similarity_score >= self._hyper_params['track_failed_cosine_similarity_th']:      # and np.max(cnt_area) > 3000

            # if len(contours) != 0 and np.max(cnt_area) > 999:
            pbox = self.get_global_box_from_masks(contours)
            rect_full, cxywh_full = self._coord_back(
                pbox,
                size=self._hyper_params["saliency_image_size"],
                region=self._hyper_params["saliency_image_field"])
            mask_pos, mask_sz = cxywh_full[:2], cxywh_full[2:]      # 目标位置的中心和大小

            conc_score = np.max(cnt_area) / sum(cnt_area)       # concentrate score
            state_score = conf_score * conc_score               # overall score
            self._state['conc_score'] = conc_score
            self._state['state_score'] = state_score

            if state_score > self._hyper_params['state_score_thresh']:
                new_target_pos = mask_pos
                lr = self._hyper_params["mask_rect_lr"]
                new_target_sz = self._state["state"][1] * (             # 貌似有多个mask 对应的矩形框；后确认，仅有一个矩形框
                    1 - lr) + mask_sz * lr
                # ***********************
                # 更新模板      # 余弦距离通常0.3以上就算正常水平
                self._tracker.reset_x_sz_ratio()
                if state_score > self._hyper_params['siamfc_template_update_thresh_1_state_score'] and \
                        similarity_score > self._hyper_params['siamfc_template_update_thresh_1_cosine_sim']:
                    self._tracker.set_state((new_target_pos, new_target_sz))
                    self._tracker.update_template()
                # ***********************
            else:
                # self._tracker.enlarge_size()
                new_target_pos, new_target_sz = self._tracker.update(im, coarse_locating=True)

            self._state['mask_rect'] = rect_full

            # # else:  # empty mask 跟踪范围过小
            # self._state['mask_rect'] = [-1, -1, -1, -1]
            # self._state['state_score'] = 0
            # # 触发粗定位
            # new_target_pos, new_target_sz = self._tracker.update(im, coarse_locating=True)

        else:  # empty mask
            self._state['mask_rect'] = [-1, -1, -1, -1]
            self._state['state_score'] = 0          
            # 触发siamfc定位
            # self._tracker.enlarge_size()
            new_target_pos, new_target_sz = self._tracker.update(im, coarse_locating=True)

        # return new_target_pos, new_target_sz        # 返回的是中心位置以及长宽
        return new_target_pos, new_target_sz
    
    def aux_seg_with_track(self, im):
        pass

    def update(self, im):

        # get track
        target_pos_prior, target_sz_prior = self._state['state']
        self._state['current_state'] = deepcopy(self._state['state'])

        # forward inference to estimate new state
        # tracking for VOS returns regressed box and correlation feature
        self._tracker.set_state(self._state["state"])
        self._tracker.unset_track_score()
        corr_feature = self._tracker.short_distance_update(         # 显然要走一遍传统的SiamFc++跟踪网络，这样必然耗时
                im)     # _tracker.update 返回来的是当前帧目标的中心位置和长宽
        

        # 要求的目标状态的格式：cxywh
        # segmentation returnd predicted masks
        gml_feature = self._state['gml_feature']
        pred_mask, pred_mask_b = self.joint_segmentation(
            im, target_pos_prior, target_sz_prior, corr_feature, gml_feature)   # gml_feature是带有第一帧filtered image的信息

        # cropping strategy loop swtiches the coordinate prediction method
        if self._hyper_params['cropping_strategy']:
            target_pos, target_sz = self.cropping_strategy(im, 
                pred_mask_b)     # 到这里其实已经将目标的bounding box计算出来了，到时候可以直接从这里返回目标的B-box
        else:
            target_pos, target_sz = self._tracker.update(im, coarse_locating=True)
            self._state["state_score"] = 0

        # global modeling loop updates global feature for next frame's segmentation
        if self._hyper_params['global_modeling']:
            if self._state["state_score"] > self._hyper_params["update_global_fea_th"]:
                self.global_modeling()
        # save underlying state
        self._state['state'] = target_pos, target_sz
        track_rect = cxywh2xywh(
            np.concatenate([target_pos, target_sz], axis=-1))
        self._state['track_box'] = track_rect
        return self._state['mask_in_full_image'], track_rect    # 返回的是full_image的mask！！！

    # ======== vos processes ======== #

    def _mask_back(self, p_mask, size=257, region=129):
        """
        Warp the predicted mask from cropped patch back to original image.

        :param p_mask: predicted_mask (h,w)
        :param size: image size of cropped patch
        :param region: region size with template = 127
        :return: mask in full image
        """

        target_pos, target_sz = self._state['current_state']
        scale_x = self._state['scale_x']

        zoom_ratio = size / region
        scale = scale_x * zoom_ratio        # 这里遗漏了一项，就是从127到129的尺度变换，所以region的值改变为127更为准确****&&**…………&&&……%……
        cx_f, cy_f = target_pos[0], target_pos[1]
        cx_c, cy_c = (size - 1) / 2, (size - 1) / 2

        a, b = 1 / (scale), 1 / (scale)

        c = cx_f - a * cx_c     # 减号后面是相对距离，减号前面是搜索图像块的绝对位置(pos的记录点是图像块的中心)
        d = cy_f - b * cy_c

        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        mask_in_full_image = cv2.warpAffine(
            p_mask,
            mapping, (self._state['im_w'], self._state['im_h']),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        return mask_in_full_image

    def _coord_back(self, rect, size=257, region=129):
        """
        Warp the predicted coordinates from cropped patch back to original image.

        :param rect: rect with coords in cropped patch
        :param size: image size of cropped patch
        :param region: region size with template = 127
        :return: rect(xywh) and cxywh in full image
        """

        target_pos, _ = self._state['current_state']
        scale_x = self._state['scale_x']

        zoom_ratio = size / region
        scale = scale_x * zoom_ratio
        cx_f, cy_f = target_pos[0], target_pos[1]       # center x coordinate in full image
        cx_c, cy_c = (size - 1) / 2, (size - 1) / 2     # center x in cropped image(saliency patch)

        a, b = 1 / (scale), 1 / (scale)

        c = cx_f - a * cx_c
        d = cy_f - b * cy_c

        x1, y1, w, h = rect[0], rect[1], rect[2], rect[3]

        x1_t = a * x1 + c
        y1_t = b * y1 + d
        w_t, h_t = w * a, h * b
        return [x1_t, y1_t, w_t, h_t], xywh2cxywh([x1_t, y1_t, w_t, h_t])