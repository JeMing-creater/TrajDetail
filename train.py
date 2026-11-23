"""Train a configured experiment and register its validation-selected weights."""
import argparse
import shutil

from src.experiments import complete, exclusive, fold_manifest, load, members, run_path, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", default="main")
    args = parser.parse_args()
    config = validate_config(load(args.config), args.config)
    target = run_path(config["method"], config["group"], args.fold, args.run)
    with exclusive(target):
        completed = target / "complete.json"
        if completed.exists():
            if load(completed)["config"] != config:
                parser.error("Run name already contains a completed experiment with a different config.")
            print(f"Training already completed: {target}")
            return
        from src.training import train_experiment

        for dataset in members(config["group"]):
            path = target / dataset / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fold_manifest(dataset, args.fold), path)
        train_experiment(config, target, args.fold, args.device)
        complete(target, config, args.fold)
        print(f"Validation-selected weights registered at {target}")


if __name__ == "__main__":
    main()
