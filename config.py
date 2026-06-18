import os
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
import torch
from torchmetrics.classification import MulticlassAccuracy

class CONFIG:
    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE : int = 32
    TRAIN_SHUFFLE: bool =True
    TEST_SHUFFLE: bool =False
    MEAN:list=[0.485, 0.456, 0.406]
    STD:list=[0.229, 0.224, 0.225]

    TRAIN_TRANSFORMS:transforms.Compose = v2.Compose  ([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORMS:transforms.Compose =  v2.Compose  ([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])
    LEARNING_RATE:float = 0.001
    TRAIN_DIR:str="DATA/train"
    TEST_DIR:str="DATA/test"
    EPOCHS:int=10
    NUM_CLASSES:int=10
    LOSS_FUNCTION:torch.nn.Module=nn.CrossEntropyLoss()
    ACCURACY_FUNCTION:torch.nn.Module=MulticlassAccuracy(num_classes=NUM_CLASSES)
