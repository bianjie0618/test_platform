# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from videoanalyst.model.task_head.taskhead_base import TASK_HEADS
from videoanalyst.utils import merge_cfg_into_hps


def build(task: str, cfg: CfgNode):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task in TASK_HEADS:
        head_modules = TASK_HEADS[task]
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    name = cfg.name     # 头部网络的name，其实是一个字符串，在这里可能是DenseboxHead
    head_module = head_modules[name]()          # 实例化头部网络模型
    hps = head_module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)    # 利用配置中的参数覆盖网络模型中的默认参数，并在下一行将之设定回去
    head_module.set_hps(hps)
    head_module.update_params()                 # 利用配置参数构建网络，并进行初始化工作

    return head_module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}
    for cfg_name, module in TASK_HEADS.items(): # cfg_name = "vos" or "track"   须知，module这个模块继承的父类是字典dict
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            task_model = module[name]
            hps = task_model.default_hyper_params   # 从类的静态变量中获取到默认的超参数，并将之注入到配置中
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
