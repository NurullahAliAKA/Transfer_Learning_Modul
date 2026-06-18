from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def creat_dataloader(
        train_dir: str,
        test_dir: str,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        batch_size: int,
        num_workers: int,
        train_shuffle: bool,
        test_shuffle: bool,

):
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
    class_names = train_data.classes
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=test_shuffle, num_workers=num_workers)


    return train_loader,test_loader,class_names


#print(creat_dataloader())

