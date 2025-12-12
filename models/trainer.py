from pytorch_lightning import LightningModule
from torch import nn
import torch
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelRecall, MultilabelSpecificity, MultilabelPrecision

class PPE_multitask_net(LightningModule):
    def __init__(
        self,
        model,
        loss_fn: nn.Module,
        optimizer,
        scheduler,
        attribute_list,
        cfg: dict,
    ):
        super(PPE_multitask_net, self).__init__()
        self.model = model
        self.loss_fn = loss_fn #cross entropy or BCE
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.attribute_list = attribute_list
        self.validation_step_outputs = {"target": [], 
                                        "output": []}
        self.cfg = cfg
        self.num_labels = sum(self.cfg["MODEL"]["NUMNBER_OF_CLASSES"])
        return

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, targets  = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets) 
        
        self.log("train_step_loss", loss, on_step=True, on_epoch=False)
        # self.log('learning rate', self.scheduler.get_lr()[0], on_step=True)

        return loss


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)

        loss = self.loss_fn(outputs, targets) 
            
        if batch_idx == 0:
            self.validation_step_outputs["output"] = outputs
            self.validation_step_outputs["target"] = targets
        else:
            self.validation_step_outputs["output"] = torch.cat((self.validation_step_outputs["output"], outputs), dim=0)
            self.validation_step_outputs["target"] = torch.cat((self.validation_step_outputs["target"], targets), dim=0)

        self.log('val_step_loss', loss, on_step=True, on_epoch=False)
        self.log(f"val_epoch_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        # At the end of the validation epoch, compute and log the mAP
        y_pred = self.validation_step_outputs["output"] #[[0.3, 0.1, 0.6], [0.2, 0.7, 0.1]]
        y_true = self.validation_step_outputs["target"] #[2, 1]
        f1 = MultilabelF1Score(num_labels=self.num_labels, average='none').to(self.device) #https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html#:~:text=target)%0Atensor(%5B0.6667%2C%200.6667%2C%201.0000%5D)-,Example%20(preds%20is%20float%20tensor)%3A,-%3E%3E%3E
        auroc = MultilabelAUROC(num_labels=self.num_labels, average='none').to(self.device)
        recall = MultilabelRecall(num_labels=self.num_labels, average='none').to(self.device)
        specificity = MultilabelSpecificity(num_labels=self.num_labels, average='none').to(self.device)
        precision = MultilabelPrecision(num_labels=self.num_labels, average='none').to(self.device)
        # Compute metrics
        y_true = y_true.to(torch.long)
        F1 = f1(y_pred, y_true)
        AUROC = auroc(y_pred, y_true)
        RECALL = recall(y_pred, y_true)
        SPECIFICITY = specificity(y_pred, y_true)
        PRECISION = precision(y_pred, y_true)
        
        # Log metrics per task
        for i, (each_f1, each_auroc, each_recall, each_specificity, each_precision) in enumerate(zip(F1, AUROC, RECALL, SPECIFICITY, PRECISION)):
            self.log(f"val_F1_task{i}", each_f1)
            self.log(f"val_auroc_task{i}", each_auroc)
            self.log(f"val_recall_task{i}", each_recall)
            self.log(f"val_specificity_task{i}", each_specificity)
            self.log(f"val_precision_task{i}", each_precision)

            self.log("val_mean_F1", F1.mean())
            self.log("val_mean_auroc", AUROC.mean())
            self.log("val_mean_recall", RECALL.mean())
            self.log("val_mean_specificity", SPECIFICITY.mean())
            self.log("val_mean_precision", PRECISION.mean())

            self.validation_step_outputs = {"target": [], 
                                            "output": []}


    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)

        loss = self.loss_fn(outputs, targets) 
            
        if batch_idx == 0:
            self.validation_step_outputs["output"] = outputs
            self.validation_step_outputs["target"] = targets
        else:
            self.validation_step_outputs["output"] = torch.cat((self.validation_step_outputs["output"], outputs), dim=0)
            self.validation_step_outputs["target"] = torch.cat((self.validation_step_outputs["target"], targets), dim=0)

        self.log(f"test_epoch_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        y_pred = self.validation_step_outputs["output"] #[[0.3, 0.1, 0.6], [0.2, 0.7, 0.1]]
        y_true = self.validation_step_outputs["target"] #[2, 1]
        f1 = MultilabelF1Score(num_labels=self.num_labels, average='none').to(self.device) #https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html#:~:text=target)%0Atensor(%5B0.6667%2C%200.6667%2C%201.0000%5D)-,Example%20(preds%20is%20float%20tensor)%3A,-%3E%3E%3E
        F1 = f1(y_pred, y_true)
        for i, each_f1 in enumerate(F1):
            self.log(f"val_F1_task{i}", each_f1)

        self.log("val_mean_F1", F1.mean())

        self.validation_step_outputs = {"target": [], 
                                        "output": []}



    def configure_optimizers(self):
        optimizer = self.optimizer
        # lr_scheduler = self.scheduler
        # interval = 'step'
        # if self.cfg["HYPERPARAM"]["SCHEDULER_NAME"] in ["StepLR", "MultiStepLR"]:
        #     interval = 'epoch'
        lr_scheduler = { #https://blog.hjgwak.com/posts/learning-rate/, https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers:~:text=The%20lr_scheduler_config%20is%20a%20dictionary%20which%20contains%20the%20scheduler%20and%20its%20associated%20configuration.%20The%20default%20configuration%20is%20shown%20below.
            'scheduler': self.scheduler,
            'interval': 'epoch',
            'frequency': 1
        }
    
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)