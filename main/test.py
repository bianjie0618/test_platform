# -*- coding: utf-8 -*-

import argparse
import os.path as osp

from loguru import logger

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml',
                        type=str,
                        help='experiment configuration')
    parser.add_argument("-d",
                        "--device",
                        default="cuda",
                        type=str,
                        help="torch.device, cuda or cpu")
    # parser.add_argument(
    #     "-v",
    #     "--video",
    #     type=str,
    #     default="webcam",
    #     help=
    #     r"video input mode. \"webcam\" for webcamera, \"path/*.<extension>\" for image files, \"path/file.<extension>\". Default is webcam. "
    # )
    # parser.add_argument("-o",
    #                     "--output",
    #                     type=str,
    #                     default="",
    #                     help="path to dump the track video")
    # parser.add_argument("-s",
    #                     "--start-index",
    #                     type=int,
    #                     default=0,
    #                     help="start index / #frames to skip")
    # parser.add_argument(
    #     "-r",
    #     "--resize",
    #     type=float,
    #     default=1.0,
    #     help="resize result image to anothor ratio (for saving bandwidth)")
    # parser.add_argument(
    #     "-do",
    #     "--dump-only",
    #     action="store_true",
    #     help=
    #     "only dump, do not show image (in cases where cv2.imshow inccurs errors)"
    # )
    # parser.add_argument('--heatmap', dest='heat_map', action='store_true')
    # parser.add_argument('--no-heatmap', dest='heat_map', action='store_false')
    # parser.set_defaults(heat_map=False)
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="raw",
                        help="choose from \"raw\" or \"self-defined: my_track\"")

    return parser


# def build_siamfcpp_tester(task_cfg):
#     # build model
#     model = model_builder.build("track", task_cfg.model)
#     # build pipeline
#     pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
#     # build tester
#     testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
#     return testers


# def build_sat_tester(task_cfg):
#     # build model
#     tracker_model = model_builder.build("track", task_cfg.tracker_model)
#     tracker = pipeline_builder.build("track",
#                                      task_cfg.tracker_pipeline,
#                                      model=tracker_model)
#     segmenter = model_builder.build('vos', task_cfg.segmenter)
#     # build pipeline
#     pipeline = pipeline_builder.build('vos',
#                                       task_cfg.pipeline,
#                                       segmenter=segmenter,
#                                       tracker=tracker)
#     # build tester
#     testers = tester_builder('vos', task_cfg.tester, "tester", pipeline)
#     return testers

# def build_siamseg_tester(task_cfg):
#     # build model
#     tracker_model = model_builder.build("track", task_cfg.tracker_model)
#     tracker = pipeline_builder.build("track",
#                                      task_cfg.tracker_pipeline,
#                                      model=tracker_model)
#     segmenter = model_builder.build('vos', task_cfg.segmenter)
#     # build pipeline
#     pipeline = pipeline_builder.build('vos',
#                                       task_cfg.pipeline,
#                                       segmenter=segmenter,
#                                       tracker=tracker)
#     # build tester
#     testers = tester_builder('track', task_cfg.tester, "tester", pipeline)
#     return testers

def build_tracking_tester(task_cfg):
    # build model

    # build pipeline
    pipeline = "instantiate a tracker"
    # build tester
    testers = tester_builder('track', task_cfg.tester, "tester", pipeline)
    return testers


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    # if task == 'track':
    #     testers = build_siamfcpp_tester(task_cfg)
    # elif task == 'vos':
    #     testers = build_sat_tester(task_cfg)
    # if task == 'track':
    #     testers = build_siamfcpp_tester(task_cfg)
    # else:
    #     testers = build_siamseg_tester(task_cfg)

    testers = build_tracking_tester(task_cfg)
    for tester in testers:
        tester.test()
