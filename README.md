




python main.py --config configs/config_improved.yaml


python main.py --config configs/config_original.yaml


python main.py --config configs/config_improved.yaml





python analyze_dataset.py --datasets_dir /data/lijinyang/1_sleep/datasets_dir --plot_combined_data

python analyze_dataset.py --datasets_dir /data/lijinyang/1_sleep/datasets_dir --plot_tsne --plot_limit 10000


pip install matplotlib scikit-learn numpy tqdm
2.  运行新脚本，并添加 `--plot_tsne` 标志：
```bash
python analyze_dataset_v3.py --datasets_dir /path/to/your/datasets --plot_tsne
3.  （可选）如果您想绘制更多（或更少）的点：
```bash
python analyze_dataset_v3.py --datasets_dir /path/to/your/datasets --plot_tsne --plot_limit 10000





# SleepDG
#### The code of the paper "Generalizable Sleep Staging via Multi-Level Domain Alignment" in AAAI-2024.
#### You should read the paper in https://arxiv.org/abs/2401.05363 for the newest version of the paper.
## Noting!!!!
#### There is an error in the dataset numbering in the formal version of the AAAI paper; we have corrected it in the preprint version of Arxiv.

## Datasets
The SleepEDFx dataset is on https://physionet.org/content/sleep-edfx/1.0.0/

The ISRUC dataset is on https://sleeptight.isr.uc.pt/

The SHHS dataset is on https://sleepdata.org/datasets/shhs

The HMC dataset is on https://physionet.org/content/hmc-sleep-staging/1.1/

The P2018 dataset is on https://physionet.org/content/challenge-2018/1.0.0/

## How to run:
```bash
python main_5.py
```


## Please cite:
```bibtex
@inproceedings{wang2024generalizable,
  title={Generalizable Sleep Staging via Multi-Level Domain Alignment},
  author={Wang, Jiquan and Zhao, Sha and Jiang, Haiteng and Li, Shijian and Li, Tao and Pan, Gang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={265--273},
  year={2024}
}
```
