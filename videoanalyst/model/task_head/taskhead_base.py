# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

# 写在这里即可，属于被动调用文件，这里的两个对象就是装饰器对象
TRACK_HEADS = Registry('TRACK_HEADS')
VOS_HEADS = Registry('VOS_HEADS')
TASK_HEADS = dict(
    track=TRACK_HEADS,
    vos=VOS_HEADS,
)
