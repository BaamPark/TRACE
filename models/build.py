import torch
import torch.nn as nn
from .swin import SwinTransformer
from .swin_utils.utils import load_pretrained, load_pretrained_original
from .tan_vgg16 import ClothingAttributeNetVGG16
from .tan_alexnet import ClothingAttributeNet_alexnet

def build_model(cfg):
    cfg_model = cfg["MODEL"]
    if cfg_model["NAME"] == 'SwinSigmoid':
        model = SwinTransformer(img_size= cfg["DATA"]["IMG_SIZE"],
                                patch_size=cfg["MODEL"]["SWIN"]["PATCH_SIZE"], #MODEL.SWIN.PATCH_SIZE,
                                in_chans=cfg["MODEL"]["SWIN"]["IN_CHANS"], #MODEL.SWIN.IN_CHANS,
                                attribute_list= cfg["MODEL"]["NUMNBER_OF_CLASSES"],  
                                embed_dim=cfg["MODEL"]["SWIN"]["EMBED_DIM"], #MODEL.SWIN.EMBED_DIM,
                                depths=cfg["MODEL"]["SWIN"]["DEPTHS"], #MODEL.SWIN.DEPTHS,
                                num_heads=cfg["MODEL"]["SWIN"]["NUM_HEADS"], #MODEL.SWIN.NUM_HEADS,
                                window_size=cfg["MODEL"]["SWIN"]["WINDOW_SIZE"],  #MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=cfg["MODEL"]["SWIN"]["MLP_RATIO"], #MODEL.SWIN.MLP_RATIO,
                                qkv_bias=cfg["MODEL"]["SWIN"]["QKV_BIAS"], #MODEL.SWIN.QKV_BIAS,
                                qk_scale=None, #MODEL.SWIN.QK_SCALE,
                                drop_rate= cfg["MODEL"]["DROP_RATE"], #MODEL.DROP_RATE,
                                drop_path_rate=cfg["MODEL"]["DROP_PATH_RATE"], #MODEL.DROP_PATH_RATE,
                                norm_layer=nn.LayerNorm,
                                ape= cfg["MODEL"]["SWIN"]["APE"], #MODEL.SWIN.APE, 
                                patch_norm=cfg["MODEL"]["SWIN"]["PATCH_NORM"], #MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=cfg["TRAIN"]["USE_CHECKPOINT"], #TRAIN.USE_CHECKPOINT,
                                # mask_ratio = maskratio,
                                fused_window_process=False,
                                feature_channel_dim=cfg["MODEL"]["FEATURE_CHANNEL_DIM"])
        load_pretrained(cfg, model)

    elif cfg_model["NAME"] == "clothingAttributeVGG16":
        model = ClothingAttributeNetVGG16(cfg_model)

    elif cfg_model["NAME"] == "clothingAttribute_alexnet":
        model = ClothingAttributeNet_alexnet(cfg_model)
    
    else:
        raise NotImplementedError(f"Unkown model")
    
    return model


def make_loss(cfg):
    cfg_model = cfg["MODEL"]
    if cfg_model["METRIC_LOSS_TYPE"] == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif cfg_model["METRIC_LOSS_TYPE"] == "binary_cross_entropy":
        return nn.BCEWithLogitsLoss()
    

def make_optimizer(cfg, model):
    cfg_hparam = cfg["HYPERPARAM"]
    if cfg["MODEL"]["TYPE"] == "TAN":
        if cfg_hparam["OPTIMIZER_NAME"] == "SGD":
            params = [{'params': model.backbone.parameters(), 'lr': cfg_hparam["LR"]}, #https://blog.slavv.com/differential-learning-rates-59eff5209a4f
                    {'params': model.task_attention.parameters(), 'lr': cfg_hparam["LR"] * 10},
                    {'params': model.attribute_feature.parameters(), 'lr': cfg_hparam["LR"] * 10},
                    {'params': model.attribute_classifier.parameters(), 'lr': cfg_hparam["LR"] * 100}]
            optimizer = getattr(torch.optim, cfg_hparam["OPTIMIZER_NAME"])(
                params, momentum=cfg_hparam["MOMENTUM"],
                weight_decay=cfg_hparam["WEIGHT_DECAY"])
            
    if cfg["MODEL"]["TYPE"] == "swin":
        if cfg_hparam["OPTIMIZER_NAME"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_hparam["LR"], weight_decay=cfg_hparam["WEIGHT_DECAY"])


    if cfg["MODEL"]["TYPE"] == "resnet":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_hparam["LR"], weight_decay=cfg_hparam["WEIGHT_DECAY"])
    return optimizer

def make_scheduler(cfg, optimizer, dataloader):
    cfg_hparam = cfg["HYPERPARAM"]
    if cfg_hparam["SCHEDULER_NAME"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                   step_size = 30, # Period of learning rate decay
                   gamma = 0.1)
        

    elif cfg_hparam["SCHEDULER_NAME"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                   max_lr = 0.05, # Initial learning rate
                   steps_per_epoch = int(len(dataloader)), # Period of learning rate decay
                   epochs = cfg_hparam["NUM_EPOCH"])
        
    elif cfg_hparam["SCHEDULER_NAME"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                   T_max=cfg["HYPERPARAM"]["NUM_EPOCH"], #https://github.com/xwf199/PARFormer
                   eta_min=cfg_hparam["LR"]*0.05)

    elif cfg_hparam["SCHEDULER_NAME"] == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg["HYPERPARAM"]["NUM_EPOCH"]//10,
            T_mult=2, # defulat is 1
            eta_min = cfg_hparam["LR"]*0.05)

    return scheduler


if __name__ == "__main__":
    #read config/swin.yaml
    import yaml
    with open('config/swin.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    print(cfg)