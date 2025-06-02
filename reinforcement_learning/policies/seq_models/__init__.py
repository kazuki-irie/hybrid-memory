from .rnn_vanilla import RNN, LSTM, GRU
from .gpt2_vanilla import GPT2
from .hqlt import HQLT
from .deltanet import DeltaNet

SEQ_MODELS = {RNN.name: RNN, LSTM.name: LSTM, GRU.name: GRU, GPT2.name: GPT2,
              HQLT.name: HQLT, DeltaNet.name: DeltaNet}
