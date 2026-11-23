"""Fit the DLBS Δ-LFM adaptation on the original cohorts and score Euler-50."""

import argparse
from pathlib import Path

import torch

from src.comparator_windows import checkpoint_path, load_partitions
from src.crossval_data import LEGACY, RUN, SPLITS, read, write
from methods.delta_lfm.delta_lfm_dlbs import evaluate_integrated, fit_ae, fit_flow, load_ae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--dataset", choices=LEGACY, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.seed = 42
    args.ae_steps, args.ae_validate = (2, 2) if args.smoke else (600, 200)
    args.base_epochs, args.fine_epochs = (1, 1) if args.smoke else (30, 10)
    args.batch_size, args.microbatch, args.eval_batch = 8, 2, 4
    args.train_break = args.val_break = args.test_break = None
    args.all_train_pairs = args.replay_direct = args.with_mu = False
    args.eval_steps = 50
    args.eval_cohort = "acdc" if args.dataset == "lumiere" else args.dataset
    args.result_dir = args.output
    args.legacy_euler = True
    args.flow_checkpoint = checkpoint_path(args.output, "flow.pt") if not args.smoke else args.output / "flow.pt"
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    rows = load_partitions(args.root, args.fold, args.dataset, smoke=args.smoke)[args.dataset]
    # Existing train/inference functions express elapsed time as stored time × 10.
    train, validation, test = [[(patient, volume, times / 10) for patient, volume, times in rows[split]]
                               for split in SPLITS]
    write(args.output / "protocol.json", {**vars(args), "counts": {s: len(rows[s]) for s in SPLITS},
          "time": "years for LUMIERE; fraction of the examination for ACDC and ISLES24",
          "input": "latest two visible images, or one if that is all the window permits",
          "deferred_checkpoint_archive": not args.smoke,
          "solver": "50 Euler steps on normalized residual path, the same as DLBS/MU"} | {
              k: str(v) for k, v in vars(args).items() if isinstance(v, Path)})
    ae = load_ae(args.weight)
    history = args.output / "flow_history.json"
    if not (args.flow_checkpoint.is_file() and history.is_file()
            and len(read(history)) == args.base_epochs + args.fine_epochs):
        fit_ae(ae, train, validation, args)
        ae.eval().requires_grad_(False)
        fit_flow(ae, train, validation, test, args)
    evaluate_integrated(ae, test, args)


if __name__ == "__main__":
    main()
