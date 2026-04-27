# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
pip install -r requirements.txt
python whole_process.py
```

`whole_process.py` is the orchestrator — it wires every module together (seed → model creation → dataloaders → training with val/early-stop/checkpoints → confusion matrix on test) and is the main entry point. There is no test suite, lint config, or build step.

`requirements.txt` pins `torch==2.11.0+cu130` / `torchvision==0.26.0+cu130`, which are CUDA 13 wheels not on PyPI. Install from the PyTorch index:

```bash
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

## Dataset layout

`create_image_dataloaders` (data_setup.py) expects an `ImageFolder` layout: `<data_path>/train/<class>/*.png`, `<data_path>/test/...`, optionally `<data_path>/val/...`. It always returns the 4-tuple `(train_dl, test_dl, val_dl, class_names)`; **`val_dl` is `None` when no `val/` directory exists** — callers must handle that. `whole_process.py` falls back to the test loader for per-epoch evaluation in that case.

The folder `data (insert folder with data)/` is a placeholder. The dataset path used in `whole_process.py` is `data/Intel-Image-Classification` with `out_features=6`; both `data_path` and `out_features` must be changed together when switching datasets.

`organize_dataset` (data_setup.py:10) is specific to the Koryakinp/Fingers dataset — it parses the digit before `L.png`/`R.png` from filenames and moves files into class subfolders. It is not invoked by `whole_process.py` and must be run manually once before training on that dataset.

## Architecture

Four modules, each a layer of the training pipeline:

- **model_builder.py** — `create_effnetb{0,3,7}_model` and `create_effnet_v2_l_model` all follow the same pattern: load torchvision weights, freeze `model.features`, replace `model.classifier` with `Dropout → Linear(in_features=<backbone-specific>, out_features=...)`. The `in_features` differ per backbone (1280 / 1536 / 2560 / 1280) — when adding a new variant, look up the correct value. Each builder returns `(model, weights)` and the **weights object is reused downstream** as `model_weights.transforms()` inside `create_image_dataloaders`, so model and dataloader creation are coupled.
- **data_setup.py** — Builds train/test/(val) `ImageFolder` datasets and `DataLoader`s. Train transform = `TrivialAugmentWide` composed with `model_weights.transforms()` when augmentation is on; test/val use only `model_weights.transforms()`. Raises `ValueError` if `model_weights` is missing.
- **engine.py** — `train_step` / `eval_step` / `train`. (`test_step` is kept as an alias of `eval_step` for compatibility.) Training uses mixed precision (`torch.amp.autocast` + `GradScaler`). `train` runs validation each epoch on the supplied `val_dataloader`, optionally steps an `lr_scheduler` (handles `ReduceLROnPlateau` separately by passing `val_loss`), saves `<model_name>_last.pth` and `<model_name>_best.pth` (lowest val loss) into `checkpoint_dir`, and stops early after `early_stopping_patience` epochs without improvement. If `ntfy_topic` is provided, after each cycle it sends a notification and calls `wait_for_stop_signal`, which **blocks on a streaming HTTP request to ntfy.sh** until a message arrives — sending `"stopp"` ends training; any other message runs another `epochs`-long loop. ntfy connection errors are retried with exponential backoff and ultimately default to STOP so the model is preserved.
- **helper_functions.py** — Side-effect utilities: `set_seed` (seeds Python/NumPy/Torch CPU+CUDA), `model_summary`, `save_model` (writes to `models/<name>.pth`), `create_writer` (TensorBoard logs to `runs/<YYYY-MM-DD>/<experiment>/<model>/<extra>/`), `plot_confusion_matrix` (accepts a `title` arg), `plot_dataloader_images` (defaults to ImageNet mean/std but accepts overrides), plus the ntfy helpers.

`SummaryWriter` is optional throughout — `engine.train` accepts `writer=None` and only logs scalars when one is passed. `whole_process.py` does not currently create one; use `create_writer(...)` from helper_functions if TensorBoard logging is wanted.

## Notes for edits

- `ntfy_topic` in `whole_process.py` is a placeholder string; leaving it set to a real-looking value will publish to that public ntfy.sh topic. Set to `None` (or remove the kwarg) to disable remote notifications/control.
- `whole_process.py` passes the val loader as `val_dataloader` to `train`, falling back to the test loader if no `val/` exists. **If you do fall back, the final confusion matrix is computed on the same data the model was selected on** — interpret accordingly.
- Per-epoch checkpoints are written to `checkpoint_dir` (default `"models"`) as `<model_name>_best.pth` and `<model_name>_last.pth`. The final `save_model(...)` call additionally writes `<model_name>.pth` (the last state) — keep or remove depending on whether you want both.
