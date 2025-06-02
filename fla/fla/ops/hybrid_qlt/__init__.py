# -*- coding: utf-8 -*-

from .chunk import chunk_hybrid_softmax_delta_rule, pos_chunk_hybrid_softmax_delta_rule
from .fused_chunk import fused_chunk_delta_rule

__all__ = [
    'fused_chunk_delta_rule',
    'chunk_hybrid_softmax_delta_rule',
    'pos_chunk_hybrid_softmax_delta_rule'
]
