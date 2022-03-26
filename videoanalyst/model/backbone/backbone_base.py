# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

# 这里是注册地

TRACK_BACKBONES = Registry('TRACK_BACKBONES')
VOS_BACKBONES = Registry('VOS_BACKBONES')

TASK_BACKBONES = dict(
    track=TRACK_BACKBONES,
    vos=VOS_BACKBONES,
)
