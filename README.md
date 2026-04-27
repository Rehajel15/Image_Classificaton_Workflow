# Image Classification Workflow

> My own framework to train an image classification model.

A small PyTorch training pipeline for image classification using transfer learning on
EfficientNet backbones. Designed to be readable end-to-end: one script (`whole_process.py`)
wires together model creation, dataloaders, training with validation, and a final
confusion matrix.

## Features

- Transfer learning on EfficientNet B0 / B3 / B7 / V2-L (frozen backbone, custom classifier head)
- Mixed-precision training (`torch.amp.autocast` + `GradScaler`)
- Per-epoch validation, optional LR scheduler, early stopping
- Best + last checkpoint saved each epoch (`<name>_best.pth`, `<name>_last.pth`)
- Optional remote control of long training runs from your phone via [ntfy.sh](https://ntfy.sh)
  (push notification after each cycle; reply `stopp` to end, anything else to keep training)
- Optional TensorBoard logging via `create_writer(...)`

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 13 wheels for `torch` / `torchvision` which are not on PyPI.
Install them from the PyTorch index:

```bash
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

If you don't have a CUDA 13 setup, install whatever `torch` / `torchvision` versions match
your hardware — the rest of `requirements.txt` is platform-independent.

## Dataset layout

The training script expects an `ImageFolder` layout:

```
data/<dataset_name>/
├── train/
│   ├── class_a/*.png
│   └── class_b/*.png
├── test/
│   ├── class_a/*.png
│   └── class_b/*.png
└── val/                  # optional — falls back to test/ if missing
    ├── class_a/*.png
    └── class_b/*.png
```

The placeholder folder `data (insert folder with data)/` shows where to put data. The
default config in `whole_process.py` points at `data/Intel-Image-Classification` with
six classes (Kaggle: *Intel Image Classification*).

For the [Koryakinp/Fingers](https://www.kaggle.com/datasets/koryakinp/fingers) dataset
(filenames like `..._3L.png`), run `organize_dataset(...)` from `data_setup.py` once to
move files into per-class subfolders before training.

## Quick start

1. Drop your dataset into `data/<your_dataset>/{train,test,val}/<class>/`.
2. Edit `whole_process.py`:
   - `data_path` and `out_features` (number of classes)
   - `model_name` for checkpoint filenames
   - `ntfy_topic` — set to `None` to disable remote notifications, or use a private topic
3. Run:

```bash
python whole_process.py
```

Checkpoints land in `models/`. Training metrics print per epoch.

## Project structure

| File | Role |
|------|------|
| `whole_process.py`   | Orchestrator — entry point |
| `model_builder.py`   | EfficientNet variants with frozen backbone + custom head |
| `data_setup.py`      | `ImageFolder` datasets and dataloaders, optional augmentation |
| `engine.py`          | `train_step`, `eval_step`, and the main `train` loop |
| `helper_functions.py`| Seeding, model summary, plots, ntfy helpers, TensorBoard writer |

## Remote control via ntfy

If `ntfy_topic` is set, after each `epochs`-block the script sends a push notification
with the current losses/accuracies and waits for a reply on the same topic.

- Reply `stopp` → training ends, model is saved.
- Reply anything else → another `epochs`-block runs.

ntfy connection errors are retried with exponential backoff; if the topic is unreachable
after several attempts, training stops gracefully and the model is preserved.

> **Note:** ntfy.sh topics are public by default. Use a long, hard-to-guess topic name or
> run a self-hosted ntfy server.
