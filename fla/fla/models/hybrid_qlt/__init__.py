# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hybrid_qlt.configuration_hybrid_qlt import HybridQLTConfig
from fla.models.hybrid_qlt.modeling_hybrid_qlt import HybridQLTForCausalLM, HybridQLTModel

AutoConfig.register(HybridQLTConfig.model_type, HybridQLTConfig)
AutoModel.register(HybridQLTConfig, HybridQLTModel)
AutoModelForCausalLM.register(HybridQLTConfig, HybridQLTForCausalLM)

__all__ = ['HybridQLTConfig', 'HybridQLTForCausalLM', 'HybridQLTModel']
