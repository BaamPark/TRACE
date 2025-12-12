# TRACE-II

## Data Preparation
Download the files below and locate in the `data` folder.
- Download the dataset from https://zenodo.org/records/17916320
- Download Swin-B and Swin-S from [official github](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file)
- Download person detector model from https://drive.google.com/file/d/1PD38lD5x83hGmY3GBOQ3oJk3psHgQvIM/view?usp=sharing

## Training PACN
- Run `python train.py --cfg "config/swin_26.yaml" --logdir  "demo" --fold 0`

## Set trained model for inference
- After training, go to lightning_logs/demo/version_0/checkpoints and copy the path of best checkpoint file (e.g. epoch=19-val_mean_F1=0.96.ckpt).
- Open `config/swin_26.yaml` and set `SAVED_MODEL_PATH` to the copied path.

## Run TRACE
- Run `bash run_trace2.bash`
- Run `compute_ape.py` to compute APE scores.
