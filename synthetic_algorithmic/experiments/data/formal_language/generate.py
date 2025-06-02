# Copyright (c) NXAI GmbH and its affiliates 2024
# Andreas Auer
import inspect

import numpy as np

from .tasks.modular_arithmetic_with_brackets import modular_arithmetic_with_brackets
from .tasks.cycle_navigation import cycle_navigation
from .tasks.even_pairs import even_pairs
from .tasks.modular_arithmetic import modular_arithmetic
from .tasks.parity import parity

GEN_FUNCS = {
    "parity": parity,
    "cycle_navigation": cycle_navigation,
    "even_pairs": even_pairs,
    "modular_arithmetic": modular_arithmetic,
    "modular_arithmetic_with_brackets": modular_arithmetic_with_brackets,
}

TOKEN_SYNTH = [
    "parity",
    "cycle_navigation",
    "even_pairs",
    "modular_arithmetic",
    "modular_arithmetic_with_brackets",
]

GEN_FUNCS_RES_DTYPE = {synth_lang_type: np.int32 for synth_lang_type in TOKEN_SYNTH}
GEN_ARGS = {key: inspect.signature(func).parameters for key, func in GEN_FUNCS.items()}

ALL_ARGNAMES = set(sum([list(args) for args in GEN_ARGS.values()], start=[]))
ALL_ARGS = {}
for argname in ALL_ARGNAMES:
    for val in GEN_ARGS.values():
        if argname in val:
            ALL_ARGS[argname] = val[argname]
            break
