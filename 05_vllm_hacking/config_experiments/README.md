# Config Experiments

This lab tests one small KV cache configuration question:

```text
If block_size changes, how many blocks are needed and how much token space is unused?
```

## Run

```powershell
python 05_vllm_hacking\config_experiments\config_experiments.py
```

The experiment keeps `num_requests` and `tokens_per_request` fixed, then compares
`block_size = 4, 8, 16`.

