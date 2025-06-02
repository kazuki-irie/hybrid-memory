## HQLT Model implementations

This folder contains HQLT model implementations used in all our experiments.

This repository was originally forked from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention). We provide our code as-is; for more up-to-date implementations of linear transformer models, we recommend checking the original repository.

* The HQLT model implementations can be found under `fla/fla/models/hybrid_qlt`.
* HQLT layer can be found under `fla/fla/layers/hybrid_qlt.py` in which different HQLT variants are specified by `attn_mode`. Naming is somewhat confusing in our code; the corresponding mapping is as follows:

  * `sync_chunk`: Synchronous
  * `chunk`: Delayed-Chunk
  * `delayed`: Delayed-Streaming
  * `sync_chunk_linear`: Synchronous with linear attention
 
  The model without positional encodings in KV-memory (as in our ablation) can be obtained by setting `use_local_pos_encoding` to `False`.
