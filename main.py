from config import CONFIG
import torch
from setup_data import creat_dataloader
from training_testing import train
from transfer_learning_class import TransferLearningModel


def main():
    train_loader,test_loader,class_names=creat_dataloader(train_dir=CONFIG.TRAIN_DIR,
                     test_dir=CONFIG.TEST_DIR,
                     train_transform=CONFIG.TRAIN_TRANSFORMS,
                     test_transform=CONFIG.TEST_TRANSFORMS,
                     batch_size=CONFIG.BATCH_SIZE,
                    num_workers=CONFIG.NUM_WORKERS,
                    train_shuffle=CONFIG.TRAIN_SHUFFLE,
                    test_shuffle=CONFIG.TEST_SHUFFLE,
                     )

    model = TransferLearningModel(num_classes=CONFIG.NUM_CLASSES,backbone_name="efficientnet_b0",)
    # PROFESYONEL İPUCU: Sadece gradyanı açık olan (requires_grad=True) parametreleri optimize et
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    result=train(model=model,
                 train_data_loader=train_loader,
                 test_data_loader=test_loader,
                 loss_function=CONFIG.LOSS_FUNCTION,
                 optimizer=torch.optim.AdamW(trainable_parameters, lr=1e-3, weight_decay=1e-4),
                 epochs=CONFIG.EPOCHS,
                 acc_function=CONFIG.ACCURACY_FUNCTION,
                 )

if __name__ == "__main__":
    main()