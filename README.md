# TrajDetail

TrajDetail forecasts a full 3D medical image at a requested future acquisition time. It uses only the observed images, their timestamps, and the requested timestamp during inference. The benchmark covers patient follow-up MRI, cardiac cine MRI, and perfusion CT.

```mermaid
flowchart LR
    A["Observed volumes and times"] --> B["Patient trajectory"]
    T["Query time"] --> B
    B --> C["Spatial deformation and intensity change"]
    A --> D["Frozen DINOv3 descriptors"]
    C --> E["Local matching and learned benefit gate"]
    D --> E
    E --> F["Voxel and detail decoders"]
    F --> G["Local high-frequency correction"]
    G --> H["Forecast volume"]
```

The trajectory assimilates available observations and rolls a shared state forward. A velocity field moves spatial structure while a source term changes intensity. Local feature matches propose additional geometric corrections. The benefit gate is trained on paired errors with and without the proposed correction, then estimates the gain from observed images at inference. The two decoder branches produce a voxel forecast and a detail forecast. The final step computes a bounded correction along the predicted local high-frequency direction on a support region derived from the model outputs.

The TrajDetail implementation is in [methods/trajdetail](methods/trajdetail), with the primary components in [unified_model.py](methods/trajdetail/unified_model.py), [matched_control.py](methods/trajdetail/matched_control.py), and [proposed.py](methods/trajdetail/proposed.py). The shared decoders are jointly trained across ACDC, LUMIERE, and ISLES’24, and a separate instance is jointly trained across MU-Glioma-Post and DLBS. The two groups use the same model design. Tests always report each cohort on its own.

## Benchmark and results

| Validation entry point | Dataset and forecast | Total held-out subjects | TrajDetail NRMSE, mean ± SD |
| --- | --- | ---: | ---: |
| `validate_acdc.py` | ACDC cardiac cine MRI, future phase | 150 | 0.03493 ± 0.00210 |
| `validate_lumiere.py` | LUMIERE longitudinal brain MRI, future visit | 67 | 0.09637 ± 0.00697 |
| `validate_isles.py` | ISLES’24 dynamic perfusion CT, future phase | 149 | 0.04535 ± 0.00267 |
| `validate_mu.py` | MU-Glioma-Post contrast-enhanced MRI, future visit | 110 | 0.09158 ± 0.00528 |
| `validate_dlbs.py` | Dallas Lifespan Brain Study structural MRI, future visit | 179 | 0.07017 ± 0.00179 |

The frozen manifests in [configs/splits](configs/splits) specify patients, observed windows, and query times. Each patient appears in exactly one held-out fold. Validation patients are distinct from training and held-out patients within each fold. Model selection uses validation data. All methods have the same available past observations and test query. A method that takes a single starting scan uses the latest observation. Methods designed for two inputs take the latest two and repeat the first when only one exists. ACDC and ISLES use normalized examination time, LUMIERE uses acquisition weeks, and MU/DLBS use acquisition days converted to the model's time scale. The headline metric is the patient-level RMSE on the shared intensity-normalized grid, called NRMSE in the benchmark. It is reported in unit intensity range. The three-fold SD is the sample SD of the three fold means.

The [three-fold reference comparison](results/reference/nrmse_3fold.csv) includes persistence, linear extrapolation, ConvLSTM, SimVP, NODE + LSTM, ViViT, CRONOS, ImageFlowNet, BrLP, Δ-LFM (50-step Euler), 4DLoG, and TrajDetail. Comparators run through the documented cohort adaptations. ConvLSTM, SimVP, NODE + LSTM, and ViViT are architecture adaptations for the common input. CRONOS uses the original discrete architecture on ACDC, LUMIERE, and ISLES and a shared continuous implementation on MU/DLBS. BrLP uses ten latent samples, 25 DDIM steps and a decoder option selected on validation. The historical comparison grids show the original BrLP adaptation. See [BrLP repair results](results/reference/brlp_repair_nrmse.csv) for the revised metrics.

## Layout

```text
TrajDetail/
├── train.py
├── validate_{acdc,lumiere,isles,mu,dlbs}.py
├── README.md
├── .gitignore
├── methods/<method>/                   # model or author-code adapter
│   └── weight/<training-group>/foldN/<run>/
├── src/                                # data, preprocessing, metrics, orchestration
├── configs/<training-group>/<method>.json
├── sh/<training-group>_<method>.sh
└── results/
    ├── reference/                      # previously recorded benchmark results
    └── <dataset>/<run>/foldN/test/      # newly evaluated cases, figures, and summaries
```

## Environment and data

Use Python 3.11 and matching builds of PyTorch and torchvision. The local GPU checks used Python 3.11 and PyTorch 2.11.0 with CUDA 12.8. Install a build suited to the machine from the PyTorch installation instructions, then run the following from the repository.

```bash
python -m pip install -r configs/requirements.txt
python -m pip install -r configs/requirements-comparators.txt
export TRAJDETAIL_DATA=/path/to/TrajDetail_data
```

Acquire each dataset through its distribution terms. Store the prepared volumes outside the repository at `TRAJDETAIL_DATA` (default `../TrajDetail_data`). These are the five input layouts.

| Dataset | Expected prepared files | Preparation |
| --- | --- | --- |
| ACDC | `source_cache/acdc/{trn,val,tst}_dat.npy` | Run `python -m src.prepare_acdc --dataset-root /path/to/acdc/database --output-dir "$TRAJDETAIL_DATA/source_cache/acdc" --manifest /tmp/acdc_manifest.json` on the original cine scans and phase metadata. Use both original ACDC training and testing directories. The writer also produces auxiliary arrays used by the historical preprocessing. |
| LUMIERE | `source_cache/lumiere/manifest.json` and one time-series NPY per patient | Run `python -m src.prepare_lumiere --raw-root /path/to/LUMIERE/Imaging` on the v202211 registered skull-stripped T1 volumes. The script matches the frozen visit list and verifies each normalized array. With the original prepared NPY files and matching patient manifest, use `python -m src.prepare_benchmark lumiere --source /path/to/lumiere_protocol`. |
| ISLES’24 | `source_cache/isles/<subject>.npy` | Run `python -m src.prepare_isles --raw-root /path/to/ctp --output "$TRAJDETAIL_DATA/isles_protocol" --workers 8`, then `python -m src.prepare_benchmark isles --source "$TRAJDETAIL_DATA/isles_protocol"`. This converts the CTP volumes into the common time-volume grid and applies the benchmark's loader normalization. |
| MU-Glioma-Post | `longitudinal/mu/<patient>.npy` | Run `python -m src.prepare_benchmark mu --source /path/to/MU-Glioma-Post`. The upstream download layout contains the contrast-enhanced NIfTI series and the clinical spreadsheet under `metadata/`, together with the file inventory `metadata/download_manifest.json`. |
| DLBS | `longitudinal/dlbs/<participant>.npy` | Run `python -m src.prepare_benchmark dlbs --source /path/to/ds004856` on the OpenNeuro BIDS tree with `participants.tsv` and the structural images identified in `configs/protocol/dlbs_T1_inventory.json`. |

`python -m src.prepare_benchmark acdc --check` checks all stored windows against the release protocol. Substitute any dataset identifier above to check the corresponding data.

Install author code separately under its license with `python -m src.setup_upstream <method> /path/to/checkout`. Supported method identifiers are `cronos` (MIC-DKFZ/Longitudinal4DMed), `imageflownet` (KrishnaswamyLab/ImageFlowNet), `log4d` (vfb8zv/4DLoG), `brlp` (LemuelPuglisi/BrLP), `delta_lfm` (chqwer2/Disease-Progression-World-Model, directory `ICLR 2026 Delta-LFM`), and `trajdetail` (facebookresearch/dinov3). Use the source revisions in [configs/protocol/upstream.json](configs/protocol/upstream.json) where a revision was available. BrLP uses `monai-generative==0.2.3` alongside the pinned MONAI 1.x release. Δ-LFM retraining needs the author's `autoencoder_epoch273.pt` at `$TRAJDETAIL_DATA/delta_lfm/`. TrajDetail needs DINOv3 ViT-B/16 weights at `$TRAJDETAIL_DATA/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` or at the path specified by `TRAJDETAIL_DINO_WEIGHTS`.

### Local archive setup

The existing research server can register its prepared data and selected checkpoints with symlinks. This uses the local benchmark archive and keeps large files at their existing locations. Run from `/workspace/TrajDetail`.

```bash
export TRAJDETAIL_DATA=/mnt/Users/LiangJiaming/TrajDetail_data
python -B -m src.import_local --source /workspace/Generation --only assets \
  --datasets /mnt/Datasets \
  --raw-cache /dev/shm/generation_new_longitudinal_20261121 \
  --source-cache /tmp/generation_3fold_source_20261122 \
  --dino /tmp/dinov3 \
  --dino-weights /mnt/Users/LiangJiaming/SomeKeeper/SPkeeper/DinoV3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
python -B -m src.import_local --source /workspace/Generation --only weights
```

## Train and evaluate

This quick check trains ConvLSTM for one epoch on the first ACDC fold, then evaluates all 50 of its held-out patients and writes a matched-case figure. Use a free CUDA device index or set `--device cpu` for the quick check.

```bash
python -B train.py --config configs/acdc/convlstm_smoke.json --fold 1 --device cuda:0 --run smoke
python -B validate_acdc.py --method copy convlstm --fold 1 --device cuda:0 --run smoke
```

Train with the full configurations using the same entry point. There is one shell launcher per experiment configuration. The launchers distribute independent cross-validation folds over the supplied GPUs, with one process per GPU and one fixed split per process.

```bash
GPUS=0,1,2 RUN_NAME=retrain bash sh/legacy_trajdetail.sh
GPUS=0,1,2 RUN_NAME=retrain bash sh/mu-dlbs_trajdetail.sh
python -B validate_acdc.py --method trajdetail --fold 1 2 3 --run retrain --device cuda:0
python -B validate_dlbs.py --method trajdetail --fold 1 2 3 --run retrain --device cuda:0
```

A config specifies the method, training group, and training hyperparameters. Duplicate an existing config and use a new `--run` name to vary its hyperparameters. Training writes checkpoints into `methods/<method>/weight/<group>/foldN/<run>/`. The completion manifest records the selected checkpoint paths, file sizes, and hashes of the fold manifests. Classical trainable baselines also verify the selected epoch against the validation history. Each evaluation entry point finds the corresponding group and run automatically, checks the completion record, and loads the validation-selected weights. Persistence and linear extrapolation require no fitted parameters.

The evaluation command defaults to all methods and all three folds. `--method` and `--fold` accept multiple values. `--limit` previews a fixed prefix of held-out patients under a separate output path. `--save-predictions` writes volumes for subsequent measurement. Outputs include a CSV for each method, the across-method fold table, a shared grayscale prediction and absolute-error figure, and a three-fold summary when all folds are scored without a preview limit. Each method uses the same figure case and color scales.

## Attribution and distribution

Reference experiments use the dataset providers' agreements and the original authors' software and weight terms. The local repository keeps downloaded data, author code installations, checkpoints, and individual generated images out of version control by default. The manuscript methods, aggregate tables and reproduction scripts are organized here for release.

### Verification record

The documented ACDC quick check completed on the local server. The evaluator loaded the newly selected ConvLSTM checkpoint and wrote metrics for all 50 held-out patients plus a case figure. All 12 checkpoint or reference paths also ran on a common ACDC preview case. The TrajDetail checkpoints were rerun on all 655 patients in the five sets of held-out folds and reproduced the [five aggregate NRMSE values](results/reference/trajdetail_revalidation.csv). Both TrajDetail training groups completed all stages from initialization in small GPU checks with shortened schedules and one training and one validation subject per cohort. The reported benchmark scores come from the full archive checkpoints.
