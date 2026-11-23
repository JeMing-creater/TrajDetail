"""Install author implementations from checkouts or archived source directories."""
import argparse
from pathlib import Path
import re
import shutil

from src.paths import ROOT


def install(method, source):
    source = Path(source).resolve()
    dest = ROOT / "methods" / method / "upstream"
    if method == "trajdetail":
        link = dest
        if link.exists() and link.resolve() != source:
            raise FileExistsError(link)
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        return
    if method == "cronos" and (source / "src/methods").is_dir():
        source /= "src"
    if method == "delta_lfm" and (source / "ICLR 2026 Delta-LFM").is_dir():
        source /= "ICLR 2026 Delta-LFM"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if dest.exists():
        return
    whitelist = {
        "imageflownet": ("src", "external_src", "LICENSE.md"),
        "cronos": ("methods", "utils", "LICENSE"),
        "delta_lfm": ("src", "LICENSE"),
        "brlp": ("src", "LICENSE"),
        "log4d": ("diffusion_m.py", "dvit.py", "velocity_loader.py", "velocity", "LICENSE"),
    }[method]
    dest.mkdir(parents=True)
    ignores = shutil.ignore_patterns(".git", "__pycache__", "*.pth", "*.pt", "*.ipynb")
    for filename in whitelist:
        original = source / filename
        if original.is_dir():
            shutil.copytree(original, dest / filename, ignore=ignores)
        elif original.is_file():
            shutil.copyfile(original, dest / filename)
    for path in dest.rglob("*.py"):
        contents = path.read_text()
        if method in ("imageflownet", "delta_lfm"):
            contents = re.sub(r"(?m)^(\s*(?:from|import) )src\.", r"\g<1>methods." + method + ".upstream.src.", contents)
            if method == "imageflownet":
                contents = re.sub(r"(?m)^(\s*(?:from|import) )external_src\.",
                                  r"\g<1>methods.imageflownet.upstream.external_src.", contents)
        if method == "cronos":
            contents = re.sub(r"(?m)^(\s*(?:from|import) )(utils|methods)\.",
                              r"\g<1>methods.cronos.upstream.\g<2>.", contents)
        if contents != path.read_text():
            path.write_text(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("trajdetail", "cronos", "brlp", "imageflownet", "log4d", "delta_lfm"))
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    install(args.method, args.checkout)


if __name__ == "__main__":
    main()
