import yaml
from models.trainer import PPE_multitask_net
from models.build import build_model, make_loss, make_optimizer, make_scheduler
from dataset_objects.dataloader import make_data_loader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import random
import numpy as np
import torch
import argparse

def main():
    seed_everything(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=False)
    args = parser.parse_args()

    with open(args.cfg, 'r') as file:
        cfg = yaml.safe_load(file)
    

    print(f"Model: {cfg['MODEL']['NAME']}")
    print(f"Train Dataset: {cfg['DATASET']['TRAIN_LABEL_PATH']}")
    print(f"Val Dataset: {cfg['DATASET']['VAL_LABEL_PATH']}")
    print(f"Test Dataset: {cfg['DATASET']['TEST_LABEL_PATH']}")
    print(f"Number of attributes: {cfg['MODEL']['NUMNBER_OF_CLASSES']}")
    print(f"Batch Size: {cfg['HYPERPARAM']['BATCH_SIZE']}")
    print(f"Number of Epoch: {cfg['HYPERPARAM']['NUM_EPOCH']}")
    print(f"Optimizer: {cfg['HYPERPARAM']['OPTIMIZER_NAME']}")
    print(f"Scheduler: {cfg['HYPERPARAM']['SCHEDULER_NAME']}")
    print(f"Test flag: {args.test}")
    print(f"k-fold idx: {args.fold}")

    if args.fold:
        print("look###: ", args.fold)
        cfg["DATASET"]["FOLD_IDX"] = args.fold

    model = build_model(cfg)

    train_loader, val_loader, test_loader = make_data_loader(cfg)
    loss_fn = make_loss(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer, train_loader)

    PPE_multitask_pl = PPE_multitask_net(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        attribute_list=cfg["MODEL"]["NUMNBER_OF_CLASSES"],
        cfg=cfg
    )

    early_stop_callback = EarlyStopping(
        monitor="val_epoch_loss",
        patience=cfg["HYPERPARAM"]["NUM_EPOCH"],
        #patience=cfg["HYPERPARAM"]["NUM_EPOCH"]//10,
        verbose=False, 
        mode="min"
        
        )
    
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_mean_F1:.2f}",  # Filename format
        save_top_k=1,  # Save the top 3 models according to the quantity monitored
        save_last=True,
        monitor="val_mean_F1",  # Metric to monitor for saving the best models
        mode="max"  # "min" means the lowest val_loss will be considered as the best model
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger("lightning_logs", name=args.logdir)

    if cfg["CHECKPOINT_PATH"]:
        model = PPE_multitask_net.load_from_checkpoint(
            checkpoint_path = cfg["CHECKPOINT_PATH"], 
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            attribute_list=cfg["MODEL"]["NUMNBER_OF_CLASSES"]
            )

    if args.test:
        trainer = Trainer(
            accelerator=cfg["TRAINER"]["ACCELERATOR"],
            devices=cfg["TRAINER"]["DEVICES"],
            callbacks=[lr_monitor],
            max_epochs=cfg["HYPERPARAM"]["NUM_EPOCH"],
            logger=logger
        )
        trainer.validate(
            model=PPE_multitask_pl,
            ckpt_path = cfg["SAVED_MODEL_PATH"],
            dataloaders=test_loader
        )
        return
    
    trainer = Trainer(
        accelerator=cfg["TRAINER"]["ACCELERATOR"],
        devices=cfg["TRAINER"]["DEVICES"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        max_epochs=cfg["HYPERPARAM"]["NUM_EPOCH"],
        logger=logger
    )


    trainer.fit(
        PPE_multitask_pl,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )

    trainer.test(
        ckpt_path = 'best',
        dataloaders=test_loader
    )
    

def seed_everything(seed=42):
    """
    Function to set the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()   
