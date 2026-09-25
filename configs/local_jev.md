# Local Jev decision classifier with Vulkan GPU offload

`CLASSIFIER_MODE=local_jev` moves every routing verdict onto a local,
purpose-built 2B decision model:

[`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF`](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF)

The gateway sends the Jev single-pass template, asks for exactly one token
(`max_tokens=1`), and reads `A` / `B` straight from the top log probabilities:

```
P(A) = e^{logprob_A} / (e^{logprob_A} + e^{logprob_B})
```

No key, no network, no spend. A verdict that misses the budget fails **open** —
the tools are kept — so a slow or missing classifier never breaks a turn.

---

## 1. Prerequisites

`llama-server` must be a build **with the Vulkan backend**. Confirm it can see a
GPU:

```bash
llama-server --list-devices
# Available devices:
#   Vulkan0: AMD BC-250 (RADV GFX1013) (15849 MiB, 14784 MiB free)
```

If no device is listed, install the userspace Vulkan driver:

```bash
# Debian / Ubuntu
sudo apt install vulkan-tools mesa-vulkan-drivers

# Arch
sudo pacman -S vulkan-icd-loader mesa-vulkan-drivers   # or vulkan-intel / vulkan-radeon

# Fedora
sudo dnf install vulkan-tools mesa-vulkan-drivers

# Verify
vulkaninfo --summary
```

If the device still does not appear, `llama-server` was compiled without Vulkan.
Rebuild llama.cpp with:

```bash
cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release
```

## 2. Serve the model, fully offloaded

`-ngl 99` offloads all layers to VRAM through Vulkan:

```bash
llama-server -hf chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF:Q8_0 \
  --port 11435 \
  -ngl 99 \
  -c 2048 \
  --threads 4
```

A correct startup logs (raise verbosity with `-lv 4` if you don't see these):

```
llama_prepare_model_devices: using device Vulkan0 (AMD BC-250 (RADV GFX1013))
load_tensors: offloading output layer to GPU
load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloaded 26/26 layers to GPU
load_tensors:      Vulkan0 model buffer size =  1970.02 MiB
```

Available quants on the same repo:

| Quant | VRAM (Vulkan0 buffer) | Notes |
| --- | --- | --- |
| `Q8_0` | ~1970 MiB | default here; best-calibrated at the shipped threshold |
| `Q4_K_M` | ~1241 MiB | smaller; needs a higher threshold to avoid false positives |
| `BF16` | ~4 GB | full precision, rarely needed for a binary decision |

**Measured** on a Radeon BC-250 (RADV, `-ngl 99`), 22 labeled prompts at the
shipped `CLASSIFIER_NEEDS_TOOLS_THRESHOLD=0.15`:

| Quant | Accuracy @0.15 | Best threshold | Median verdict | VRAM |
| --- | --- | --- | --- | --- |
| `Q8_0` | **22/22 (100%)** | 0.16 → 100% | 400 ms | 1970 MiB |
| `Q4_K_M` | 15/22 (68%) | 0.24 → 95% | 391 ms | 1241 MiB |

Speed is a wash (~2%, within run-to-run noise). Q4_K_M's error mode is false
positives: its *no-tool* probabilities cluster at 0.16–0.40, just over the
0.15 cut, so it keeps tool schemas it does not need. Re-tuning to ~0.24 recovers
most of it but still trails Q8_0 — which is why `Q8_0` is the default.

## 3. Point the gateway at it

```bash
agent-gateway start --profile local_jev
# or pick it from the interactive launcher: agent-gateway  ->  question 2
```

Relevant `.env` keys:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLASSIFIER_MODE` | — | set to `local_jev` |
| `LOCAL_JEV_URL` | `http://127.0.0.1:11435/v1/chat/completions` | llama-server endpoint |
| `CLASSIFIER_MODEL` | `jev-style-qwen3.5-2b-q8_0` | label sent to the server |
| `LOCAL_JEV_TIMEOUT_SECONDS` | `0.4` | verify budget before failing open |

## 4. Tuning the budget

`LOCAL_JEV_TIMEOUT_SECONDS` is the only knob that matters for reliability:

* **GPU (`-ngl 99`)**: `0.4` is comfortable; a discrete GPU answers in a few ms.
* **CPU only**: raise it (e.g. `2.0`). Verdicts are single-token, but the
  ~65-token decision template dominates prompt-eval time, so a CPU host can
  exceed 0.4s and lose verdicts to the timeout — which is safe (tools are kept)
  but means the classifier is effectively bypassed.

## 5. Quick health check

```bash
agent-gateway doctor        # classifier line reports local_jev reachability
curl -s http://127.0.0.1:11435/health
```
