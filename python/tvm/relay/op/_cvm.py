
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import topi
from .op import register_compute, register_schedule
from .op import schedule_injective

schedule_injective = schedule_injective
schedule_elemwise = schedule_injective

register_schedule("cvm_lut", schedule_injective)
register_schedule("cvm_clip", schedule_elemwise)
register_schedule("cvm_left_shift", schedule_elemwise)
register_schedule("cvm_right_shift", schedule_elemwise)

