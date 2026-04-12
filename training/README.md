# AfriqueLLM Training — Reproduction Setup

End-to-end pipeline for continued pre-training (CPT) following the AfriqueLLM paper.

---

## 1. Install dependencies

```bash
pip install llamafactory deepspeed flash-attn liger-kernel wandb
# or, from the LLaMA-Factory repo:
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
```

---

## 2. Download data

```bash
python download_fineweb2.py \
    --out_dir data/raw/fineweb2 \
    --streaming \
    --max_rows_per_lang 1000   # remove this cap for the full dataset
```

---

## 3. Register data with LLaMA-Factory

```bash
# Option A — one merged file (simplest, good for testing)
python prepare_data.py \
    --data_dir data/raw/fineweb2 \
    --out_dir data/lf \
    --merge

# Option B — one dataset per language (matches the paper, allows per-language upsampling)
python prepare_data.py \
    --data_dir data/raw/fineweb2 \
    --out_dir data/lf
```

Then update `dataset` in `train_cpt.yaml`:
- Option A: `dataset: fineweb2`
- Option B: `dataset: fineweb2_fra_Latn,fineweb2_swh_Latn,...`

---

## 4. Choose your model

Edit `model_name_or_path` in `train_cpt.yaml`:

| Paper model       | HuggingFace ID              |
|-------------------|-----------------------------|
| AfriqueLlama-8B   | meta-llama/Llama-3.1-8B     |
| AfriqueGemma-4B   | google/gemma-3-4b-pt        |
| AfriqueGemma-12B  | google/gemma-3-12b-pt       |
| AfriqueQwen-8B    | Qwen/Qwen3-8B               |
| AfriqueQwen-14B   | Qwen/Qwen3-14B              |

---

## 5. Choose DeepSpeed stage

The paper uses ZeRO-1 for smaller models and ZeRO-2 for 12B+.

- 4B  → `deepspeed: ds_zero1.json`
- 8B  → `deepspeed: ds_zero1.json`
- 12B → `deepspeed: ds_zero2.json` (rename ds_zero1.json and change `"stage": 2`)
- 14B → `deepspeed: ds_zero2.json`

---

## 6. Launch training

```bash
# Single node, all available GPUs
bash run_train.sh

# Specify GPU count explicitly
NUM_GPUS=4 bash run_train.sh

# Multi-node (run on each node)
MASTER_ADDR=192.168.1.1 NNODES=4 NODE_RANK=0 bash run_train.sh  # node 0
MASTER_ADDR=192.168.1.1 NNODES=4 NODE_RANK=1 bash run_train.sh  # node 1
# etc.
```

---

## Key hyperparameters from the paper

| Parameter              | Value    | Source          |
|------------------------|----------|-----------------|
| Learning rate          | 5e-5     | Table 9 sweep   |
| Context length         | 16k      | Table 10 sweep  |
| LR scheduler           | cosine   | Appendix B.1    |
| min_lr_rate            | 0.01     | Table 11        |
| warmup_ratio           | 0.001    | Table 11        |
| Global batch size      | 4M tokens| Section 3.2     |
| weight_decay           | 0.1      | Appendix B.2    |
| adam_beta1/beta2       | 0.9/0.95 | Appendix B.2    |

The launch script automatically computes `gradient_accumulation_steps` to
hit the 4M token global batch size given however many GPUs you have.

---

## Notes

- **Flash Attention 3** (`flash_attn: fa3`) requires H100 GPUs.
  On A100 use `fa2`; on other hardware omit the flag entirely.
- **Liger Kernel** is optional but gives a meaningful throughput boost.
- The paper's full run is 1 epoch over 25B tokens, which took 18–31 hours
  on 64 H100s depending on model size (Table 12).
- For testing with FineWeb2 only (no code/math/synthetic), expect slightly
  weaker reasoning scores than Table 2 (the `+M` monolingual-only row).
