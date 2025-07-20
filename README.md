# Drone Detection with YOLOv11 & Autodistill: Complete Repository Guide

A concise overview:  
This README explains how to operate an end-to-end, **zero-shot-labeling ➜ model-training ➜ artifact-publishing** pipeline for DJI Phantom drone segmentation. It covers local experiments, the fully automated GitHub Actions workflow, troubleshooting, and extensibility notes. Everything assumes only an `images/` folder seeded with raw pictures and the single workflow file already supplied.

## Repository Layout & Prerequisites

| Path | Purpose | Required at clone? |
|---|---|---|
| `.github/workflows/train-model.yml` | CI script that labels images with GroundedSAM, trains a YOLOv11 segmentation network, and uploads `best.pt` | **Yes** |
| `images/` | Flat directory containing unlabeled `.jpg`/`.jpeg`/`.png` files | **Yes**—place at least 1 image here |
| `README.md` | This file | **Yes** |

### Local software

- Python 3.11
- Pip ≥23
- CUDA toolkit (optional, speeds up local runs)

## 1  Quick-Start Checklist

### 1.1 Clone & seed

```bash
git clone https://github.com//drone-detection-yolov11-autodistill.git
cd drone-detection-yolov11-autodistill
mkdir -p images
# copy at least one DJI Phantom image into images/
```

### 1.2 Push to trigger

```bash
git add images/*
git commit -m "Add seed image(s)"
git push origin main
```

### 1.3 Watch the Action

1. Navigate to **Actions → Drone Detection Training**.  
2. The job logs show:
   - “Auto-labeling images with GroundedSAM …”
   - Dataset stats (e.g., `✅ Dataset created: 1 labels for 1 images`).
   - YOLO “Transferred xxx/xxx items” then 50 epochs of loss curves.
3. On success, download the **artifact** `drone-model-.zip`. GitHub stores it outside the repo for 30 days[1].

## 2  How the Pipeline Works

### 2.1 Zero-Shot Annotation

GroundedSAM fuses Grounding DINO’s text-aware detection with Segment Anything’s mask generator, enabling it to draw pixel-accurate masks solely from your English prompt[2][3]. The `CaptionOntology` maps sentences to class ids:

```python
ontology = CaptionOntology({
    "DJI Phantom drone flying in the sky": "drone",
    "DJI Phantom drone on a surface":      "drone"
})
```

When `base.label()` runs, every image in `images/` is scanned; objects matching either caption are segmented and saved into YOLO-style mask labels under `dataset/train/labels/`.

### 2.2 Dataset Autogeneration

Autodistill writes the canonical YOLO directory tree:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml          ← tells YOLO class count, paths, splits
```

Fresh creation each run guarantees **clean-slate consistency**—no mixed annotation states linger between commits.

### 2.3 Model Training

Ultralytics YOLOv11 is loaded via:

```python
model = YOLO("yolo11n-seg.pt")   # 2.8 M params starter weights[2]
model.train(data="dataset/data.yaml",
            epochs=50,           # hard-coded for CI CPU speed
            imgsz=640,
            batch=16,
            patience=25)
```

Key facts:  
- **Segmentation head:** YOLOv11’s Segment mode with PSA blocks and C2 layers[4][5].  
- **Epoch length:** limited to 50 for GitHub’s 6-hour default quota, but adjustable via the workflow dispatch input.  
- **CPU-only runner:** still completes because n-variant model is tiny (≈2.8 M params[6]).

### 2.4 Artifact Publishing

`actions/upload-artifact@v4` zips `runs/segment/train/weights/best.pt` and stores it in GitHub’s artifact blob service—not the repo itself—so your Git history stays slim and free from binaries[1][7]. Default retention is 90 days; the workflow sets 30 days.

## 3  Local Replication (Optional)

1. Install:

   ```bash
   pip install ultralytics autodistill autodistill-grounded-sam
   ```

2. Run the same script locally (copy from workflow).  
3. Inspect masks in `dataset/train/labels/`.  
4. Visualize training metrics with TensorBoard (`tensorboard --logdir runs/segment/train`).

## 4  Common Questions

| Question | Short Answer | Details |
|---|---|---|
| **Do I need manual labels?** | **No.** | GroundedSAM handles zero-shot segmentation[2]. |
| **Why delete `dataset/` each run?** | Prevents annotation drift. | Ensures all images share identical labeling rules each time. |
| **Will W&B still ask for a key?** | No. | The workflow exports `WANDB_API_KEY` inline—the hex string you captured—which auto-logs silently[8][9]. |
| **Where is the model stored?** | GitHub artifact store. | Not committed; downloadable ZIP appears under the workflow run[1]. |
| **Can I exceed storage quota?** | Yes, after ~2 GB (Pro) / 500 MB (Free). | Clean old artifacts or raise limits[10][7]. |

## 5  Extending the Pipeline

### 5.1 More Classes

Add pairs to the ontology:

```python
ontology = CaptionOntology({
    "DJI Phantom drone":      "drone",
    "Parrot Anafi drone":     "drone_parrot",
    "Battery pack":           "battery"
})
```

YOLO will auto-expand `nc` inside `data.yaml`.

### 5.2 GPU Runners

Switch the job runner:

```yaml
# Example self-hosted runner label
runs-on: [self-hosted, gpu]
```

Then raise epochs/batch for faster convergence.

### 5.3 Advanced Hyperparameters

Use YOLO’s CLI flags (`lr0`, `lrf`, `weight_decay`, etc.) documented in Ultralytics Train-mode guide[11].

## 6  Troubleshooting Cheat-Sheet

| Symptom | Possible Cause | Fix |
|---|---|---|
| `❌ No labels created` | Caption text too strict | Rephrase ontology prompts to be more generic. |
| `wandb: ERROR API key not configured` in logs | Key misspelled | Ensure `WANDB_API_KEY` string matches exactly; regenerate via https://wandb.ai/authorize[8]. |
| GitHub “Artifact storage quota hit” | Exceeded plan limit | Delete old artifacts or shorten `retention-days`[7]. |
| `CUDA out of memory` (self-hosted) | Too high `batch` | Lower `batch=8` or `imgsz=512`. |

## 7  Reference Table of Important Paths

| Purpose | Relative Path | Generated By | When |
|---|---|---|---|
| Raw images input | `images/` | You | Pre-commit |
| Auto-labeled masks | `dataset/train/labels/` | Autodistill | Workflow step “Auto-labeling …” |
| YOLO training logs | `runs/segment/train/` | Ultralytics | During training |
| Best model weights | `runs/segment/train/weights/best.pt` | Ultralytics | Training completion |
| Downloadable artifact | **Actions → Artifacts** | GitHub | Post-upload |

## 8  Security Notes

- **Never expose real API keys** in public repos—this README uses the placeholder you provided.  
- GitHub artifacts **ignore file permissions**[1]. If you need executables preserved, tar them first.  
- Delete artifacts periodically or automate cleanup to avoid storage lockouts[10][12].

## 9  Next Steps After First Successful Run

1. **Download `best.pt`** artifact.  
2. Test locally:

   ```python
   from ultralytics import YOLO
   model = YOLO("best.pt")
   model.predict("some_new_drone.jpg", save=True, imgsz=640)
   ```

3. **Iterate**: add diverse images (different angles, lighting), commit, and watch retraining.

## 10  Credits

- **Ultralytics** for YOLOv11 core engine [6][4][11].  
- **Roboflow** for Autodistill and GroundedSAM integration [2][13].  
- **GitHub Actions** team for artifact tooling [1].  
- **Weights & Biases** for experiment tracking that CI can automate via `WANDB_API_KEY` [8][9].

