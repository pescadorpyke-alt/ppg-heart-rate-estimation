# `data/` — pointer only

The raw PPG-DaLiA dataset (~2.7 GB) and all processed artifacts do **not** live
here, to keep this OneDrive-synced folder light. They live on:

```
D:\ppg-dalia-hr\
  raw\          # extracted PPG-DaLiA pickles: S1.pkl ... S15.pkl
  processed\    # windowed + filtered arrays (created in Phase 1)
  checkpoints\  # trained model weights (created in Phase 2)
```

Paths are defined in [`src/config.py`](../src/config.py) and can be overridden
with the `PPG_HEAVY_ROOT` environment variable.

Dataset: [PPG-DaLiA, UCI ML Repository #495](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
(CC BY 4.0). Download instructions are in the project root `README.md`.
