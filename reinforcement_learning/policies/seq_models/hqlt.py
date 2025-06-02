import torch.nn as nn
import torchkit.pytorch_utils as ptu
from fla.models import HybridQLTModel, HybridQLTConfig
from fla.models.utils import Cache
import torch


class HQLT(nn.Module):
    name = "hqlt"

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layer,
        n_head,
        pdrop,
        max_seq_length,
        **kwargs
    ):
        super().__init__()
        # config = transformers.GPT2Config(
        #     vocab_size=1,  # doesn't matter, we don't use word embeddings
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     n_embd=hidden_size,
        #     attn_pdrop=pdrop,
        #     resid_pdrop=pdrop,
        #     embd_pdrop=pdrop,
        #     # Maximum length sequence the transformer will see; default 1024 might be not long
        #     n_positions=max_seq_length + 2,
        # )  # needs to be divisible by n_head

        if 'chunk_size' in kwargs.keys():
            chunk_size = kwargs['chunk_size']
        else:
            chunk_size = 64

        # NB: dropout is ignored
        config = HybridQLTConfig(
            vocab_size=1,  # doesn't matter, we don't use word embeddings
            num_hidden_layers=n_layer,
            num_heads=n_head,
            hidden_size=hidden_size,
            head_dim=hidden_size // n_head,
            use_local_pos_encoding=True,
            use_short_conv=False,
            use_gate=False,
            use_memory_gate=True,
            qk_activation='identity',
            attn_mode='sync_chunk',
            chunk_size=chunk_size
        )

        self.transformer = HybridQLTModel(config)

        assert input_size == hidden_size
        self.hidden_size = hidden_size
        self.max_history_length = max_seq_length - 1
        # print({k: v.shape for k, v in self.transformer.named_parameters()})

    def forward(self, input_embeds, pkv):
        """
        input_embeds:
            training -- (max_seq_length, B, input_dim)
            eval -- (1, 1, input_dim)
        """
        input_embeds = torch.transpose(input_embeds, 0, 1)  # (T,B) --> (B, T)

        outputs = self.transformer(
            inputs_embeds=input_embeds,
            output_attentions=False,
            past_key_values=pkv
        )
        # return outputs[0], outputs.past_key_values
        return outputs[0].transpose(0, 1), outputs.past_key_values

    def get_zero_internal_state(self, batch_size=None):
        return Cache()
        if batch_size is None:  # inference, batch_size=1
            pkv = None
            initial_timestep = ptu.arange(0, 1)
            return (pkv, initial_timestep, ptu.zeros((0, 1, self.hidden_size)).float())
        else:  # training, not used
            return None
