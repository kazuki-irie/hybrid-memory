from typing import Optional, Tuple, List

import torch
from fla.models import HybridQLTForCausalLM, HybridQLTConfig

class HybridQLTForCausalLMMod(HybridQLTForCausalLM):
    def __init__(self, config: HybridQLTConfig):
        super().__init__(config)
        self.config = config
        # self.config.attn_mode = "gla_mod_chunk"

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        causal_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return causal_output.logits
