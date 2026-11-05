# Environment

Recorded by `scripts/setup_lambda.sh` on 2026-11-04T21:23:19Z.

## Instance

- Instance type: 1x A100 40GB SXM4
- Hourly price (USD): 1.99
- Repo commit: dcb7f56a1f3a4735f7224dd2443da204b77f6b6d

## GPU

```
name, driver_version, memory.total [MiB]
NVIDIA A100-SXM4-40GB, 570.148.08, 40960 MiB
```

```
Wed Nov  4 21:23:19 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  |   00000000:06:00.0 Off |                    0 |
| N/A   31C    P0             48W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## CUDA toolkit

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

## OS and Python

```
PRETTY_NAME="Ubuntu 22.04.5 LTS"
VERSION_ID="22.04"
Python 3.10.12
```

## Disk

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda1       497G   25G  472G   6% /
```

## Model

- HF model: `mistralai/Mistral-7B-v0.1` @ `27d67f1b5f57dc0953326b2601d68371d40ea8da`
- GGUF repo: `TheBloke/Mistral-7B-v0.1-GGUF` @ `d4ae605152c8de0d6570cf624c083fa57dd0d551`
- GGUF files are separately quantized artifacts, not the same weights as the HF runs.

| GGUF file | SHA256 |
|---|---|
| `mistral-7b-v0.1.Q4_K_M.gguf` | `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c` |
| `mistral-7b-v0.1.Q8_0.gguf` | `b342a437813a742535e78c0251ced535f21cac9adabad7b467bdc11b9d5a77fd` |

## Python packages

Full lists: `results/freeze-venv-{hf,vllm,llamacpp}.txt`.

| venv | torch | key package |
|---|---|---|
| .venv-hf | 2.2.2+cu121 | transformers 4.40.2, bitsandbytes 0.43.1, lm-eval 0.4.3 |
| .venv-vllm | 2.3.0+cu121 | vllm 0.4.2 |
| .venv-llamacpp | 2.2.2+cu121 | llama-cpp-python 0.2.77 |

## Version deviations from requirements.txt

| Package | Pinned | Used | Reason |
|---|---|---|---|
| lm-eval (.venv-vllm only) | 0.4.3 | 0.4.3, one line patched | `detokenize=True` in `vllm_causallms.py`: vLLM 0.4.2 drops prompt_logprobs otherwise |
| peft | not pinned | 0.10.0 | newest peft needs newer accelerate/transformers than this stack; lm-eval failed to import |
| transformers (.venv-vllm) | not pinned for vLLM | 4.40.2 | newest requires torch >= 2.6; vLLM ships torch 2.3.0 |
| accelerate, datasets (.venv-vllm) | not pinned for vLLM | 0.29.3, 2.19.1 | match the HF venv |
| torch (.venv-vllm) | 2.2.2 | 2.3.0 | vLLM 0.4.2 pins its own torch |
