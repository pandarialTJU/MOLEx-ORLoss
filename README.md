# MoLEx Anti-Spoofing

MoLEx (Mixture-of-LoRA-Experts) fine-tunes WavLM with sparse LoRA experts and an LSTM classifier for anti-spoofing. This repository contains the minimal training/evaluation code used in our experiments and is ready for public release.

## Repository Layout

```
molex_github/
├── configs/           # JSON configs (model + optimizer)
├── src/               # Training + model sources
│   ├── main.py        # Distributed training entry point (torchrun)
│   ├── model_MOE.py   # Model implementation
│   └── ...            # Data utilities, WavLM wrapper, etc.
├── environment.yml    # Conda environment specification
├── run_training.sh    # Helper launcher script
└── README.md
```

## Setup

1. **Create the environment**
   ```bash
   conda env create -f environment.yml
   conda activate molex
   ```
2. **Download WavLM weights**  
   Obtain `WavLM-Large.pt` from Microsoft (or another compatible checkpoint) and either:
   - set `model_config.wavlm_checkpoint` in `configs/molex_ssl.conf`, or
   - export `MOLEX_WAVLM_CHECKPOINT=/path/to/WavLM-Large.pt`.
3. **Prepare metadata**  
   Dataloaders expect `fold{K}_{train,validation,evaluation}.tsv` files plus a `wav.scp`-style file mapping utterance IDs to waveform paths. This matches the ASVspoof/CyberSecurity format.

## Training and Evaluation

1. Edit `configs/molex_ssl.conf` to reflect your dataset paths, batch sizes, epochs, etc.
2. Provide the required paths via environment variables or by editing `run_training.sh`:
   ```bash
   export META_DIR=/path/to/meta
   export FEAT_FILE=/path/to/wav.scp
   export OUTPUT_DIR=/path/to/output
   export NUM_GPUS=2
   ```
3. Launch training:
   ```bash
   bash run_training.sh
   ```
   The script wraps `torchrun`, points `PYTHONPATH` at `src/`, and stores checkpoints/logs under `${OUTPUT_DIR}/Exp_<idx>/`.

You can also invoke the trainer directly:
```bash
torchrun --standalone --nproc_per_node=2 src/main.py \
  --config configs/molex_ssl.conf \
  --meta_dir $META_DIR \
  --feat_file $FEAT_FILE \
  --output_dir $OUTPUT_DIR \
  --exp_idx 0
```

