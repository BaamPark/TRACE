import torch.nn as nn
import torch.nn.functional as F
import torch
from .vgg import vgg16
class TaskAttention(nn.Module):
    def __init__(self, num_tasks):
        super(TaskAttention, self).__init__()
        self.num_tasks = num_tasks

        self.attention_module = nn.Sequential(
             nn.Conv2d(512, 512, kernel_size=1, bias=True),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, self.num_tasks, kernel_size=1, bias=True),
             nn.Softmax(dim=1)
        )
      
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        att_value = self.attention_module(x)
        a_t = torch.split(att_value,1, dim=1)
        #pdb.set_trace()
        att_features = []
        for i in range(self.num_tasks):
            a_t_repeat = a_t[i].repeat(1,512,1,1) #expand a_t value to the same dimension with x
            att_feature = a_t_repeat * x
            att_feature = att_feature.view(x.size(0),512 * 7 * 7)
            att_features.append(self.classifier(att_feature))
        return att_features





class ClothingAttributeNetVGG16(nn.Module):
    def __init__(self, model_cfg):
        super(ClothingAttributeNetVGG16, self).__init__()
        self.num_classes = model_cfg["NUMNBER_OF_CLASSES"]

        if model_cfg["BACKBONE"] == 'vgg16':
            self.backbone = vgg16(True)    

        self.task_attention =  TaskAttention(len(self.num_classes))
        self.attribute_feature = []
        self.attribute_feature = nn.ModuleList([self._make_feature(1024) for classes in self.num_classes])

        self.attribute_classifier = []
        self.attribute_classifier = nn.ModuleList([self._make_classifier(1024,classes) for classes in self.num_classes])

    def _make_feature(self, fc_size):
        fc_feature = nn.Linear(4096, fc_size)
        fc_relu = nn.ReLU(inplace=True)
        fc_drop = nn.Dropout()

        return nn.Sequential(fc_feature,fc_relu,fc_drop)

    def _make_classifier(self, fc_size, classes):
        output = nn.Linear(fc_size, classes)
        return nn.Sequential(output)


    def forward(self, x):
        x = self.backbone(x)
        x = self.task_attention(x)

        idx = 0
        fc = []
        for att_fc in self.attribute_feature:
            fc.append(att_fc(x[idx]))
            idx += 1
        idx = 0
        output = []
        for att_classifier in self.attribute_classifier:
            output.append(att_classifier(fc[idx]))
            idx += 1

        concatenated_output = torch.cat(output, dim=1)
        return concatenated_output
    

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    model_cfg = {
        "BACKBONE": 'vgg16',
        "NUMNBER_OF_CLASSES": [3,4,4,6],
        "METRIC_LOSS_TYPE": 'cross_entropy'
    }
    model = ClothingAttributeNetVGG16(model_cfg)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output)