# -*- coding: utf-8 -*-
# scripts/make_allfold.py
import argparse
from pathlib import Path

from utils.allutils import collect_and_plot_allfold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="训练输出根目录（包含 fold0, fold1, ...）")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    fold_dirs = [model_dir / f"fold{k}" for k in range(args.folds)]
    out_dir = model_dir / "allfold"
    collect_and_plot_allfold(fold_dirs, out_dir=out_dir)
    print(f"[OK] allfold 综合图已输出到: {out_dir}")

if __name__ == "__main__":
    main()
