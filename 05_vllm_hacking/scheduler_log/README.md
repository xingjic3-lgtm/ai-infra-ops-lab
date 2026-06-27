# Scheduler Log

This lab logs a tiny vLLM-like scheduler loop.

Each scheduler step:

```text
1. move waiting requests into running slots
2. decode one token for each running request
3. move completed requests to finished
```

## Run

```powershell
python 05_vllm_hacking\scheduler_log\scheduler_log.py
```

Read the output as:

```text
waiting queue -> running batch -> one decode step -> finished queue
```

