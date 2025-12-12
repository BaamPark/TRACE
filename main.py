import yaml
from models.build import build_model, make_loss, make_optimizer, make_scheduler
from dataset_objects.dataloader import make_data_loader

if __name__ == "__main__":
    with open('config/swin.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    train_loader, val_loader, test_loader = make_data_loader(cfg)