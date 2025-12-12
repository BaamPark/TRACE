from dataset_objects.dataset import PPE_dataset, PPE_dataset_KFold
import torchvision.transforms as T
from torch.utils.data import DataLoader

def make_data_loader(cfg):
    dataset_cfg = cfg["DATASET"]
    hparam_cfg = cfg["HYPERPARAM"]
    train_transform = T.Compose([
            T.Resize(dataset_cfg["INPUT_IMG_SIZE"]),
            T.RandomHorizontalFlip(p=dataset_cfg["PROB"]),
            T.RandomRotation(dataset_cfg["RANDOM_ROTATION_RANGE"]),
            T.RandomGrayscale(p=0.1),
            # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    

    val_transform = T.Compose([T.Resize(dataset_cfg["INPUT_IMG_SIZE"]), 
                        T.ToTensor(), 
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    if dataset_cfg["NAMES"] == "PPE_dataset":
        train_data = PPE_dataset(dataset_cfg["TRAIN_LABEL_PATH"],train_transform)
        val_data = PPE_dataset(dataset_cfg["VAL_LABEL_PATH"],val_transform)
        test_data = PPE_dataset(dataset_cfg["TEST_LABEL_PATH"],val_transform)

    elif dataset_cfg["NAMES"] == "PPE_dataset_KFold":
        train_data = PPE_dataset_KFold(dataset_cfg["TRAIN_LABEL_PATH"], dataset_cfg["IMG_PATH"], dataset_cfg["FOLD_IDX"], 'train', train_transform)
        val_data = PPE_dataset_KFold(dataset_cfg["VAL_LABEL_PATH"], dataset_cfg["IMG_PATH"], dataset_cfg["FOLD_IDX"], 'val', val_transform)
        test_data = PPE_dataset_KFold(dataset_cfg["VAL_LABEL_PATH"], dataset_cfg["IMG_PATH"], dataset_cfg["FOLD_IDX"], 'val', val_transform)
    
    train_loader = DataLoader(
            train_data,
            batch_size=hparam_cfg["BATCH_SIZE"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    
    val_loader = DataLoader(
            val_data,
            batch_size=hparam_cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    
    test_loader = DataLoader(
            test_data,
            batch_size=hparam_cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader


