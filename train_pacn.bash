# Train reference models

python train.py --cfg "config/tan_alex.yaml" --logdir  "tan_alex_fold0" --fold 0
python train.py --cfg "config/tan_alex.yaml" --logdir  "tan_alex_fold1" --fold 1
python train.py --cfg "config/tan_alex.yaml" --logdir  "tan_alex_fold2" --fold 2
python train.py --cfg "config/tan_alex.yaml" --logdir  "tan_alex_fold3" --fold 3
python train.py --cfg "config/tan_alex.yaml" --logdir  "tan_alex_fold4" --fold 4

python train.py --cfg "config/tan_vgg.yaml" --logdir  "tan_vgg_fold0" --fold 0
python train.py --cfg "config/tan_vgg.yaml" --logdir  "tan_vgg_fold1" --fold 1
python train.py --cfg "config/tan_vgg.yaml" --logdir  "tan_vgg_fold2" --fold 2
python train.py --cfg "config/tan_vgg.yaml" --logdir  "tan_vgg_fold3" --fold 3
python train.py --cfg "config/tan_vgg.yaml" --logdir  "tan_vgg_fold4" --fold 4

