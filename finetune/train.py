import mlflow
import os
import argparse
import torch
from sklearn import metrics
from trainer import Trainer
from dino_model import DinoVisionTransformerClassifier
from torch_datasets import CustomDataset
from torchvision import transforms


def parse_parameters():
    parser = argparse.ArgumentParser(description="Script to train a model using Dino")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dino_version', type=str, required=True, help='Version of Dino to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for training (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--neg_label', type=str, help='Negative label in dataframe')
    parser.add_argument('--pos_label', type=str, help='Positive label in dataframe')

    return parser.parse_args()


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),  # Randomly zoom in/out
        transforms.RandomAffine(degrees=0, shear=10),  # Apply random shear transformation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly shift width and height by 10%
        transforms.ToTensor(),  # Conver to tensor
        # transforms.Normalize(mean=0.5, std=0.2)  # Normalize the image with your dataset's mean and std
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.2)
    ])

    return train_transforms, val_transforms


if __name__ == "main":
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    params = parse_parameters()
    base_path = f'/content/MedDino/data/{params.dataset}/'
    csv_path = f'/content/MedDino/data/{params.dataset}/dataframe/'

    train_transforms, val_transforms = get_transforms()

    for i in range(1, 6):
        with mlflow.start_run(run_name=f'{params.dataset}{i}'):

            mlflow.log_params(vars(params))

            train_dataset = CustomDataset(base_path=base_path,
                                          csv_path=os.path.join(csv_path, f'{params.dataset}_train_fold{i}.csv'),
                                          transform=train_transforms,
                                          label_map={params.neg_label: 0, params.pos_label: 1})

            val_dataset = CustomDataset(base_path=base_path,
                                        csv_path=os.path.join(csv_path, f'{params.dataset}_val_fold{i}.csv'),
                                        transform=val_transforms,
                                        label_map={params.neg_label: 0, params.pos_label: 1})

            test_dataset = CustomDataset(base_path=base_path,
                                         csv_path=os.path.join(csv_path, f'{params.dataset}_test_fold{i}.csv'),
                                         transform=val_transforms,
                                         label_map={params.neg_label: 0, params.pos_label: 1})

            model = DinoVisionTransformerClassifier(hub_path=params.dino_version)
            model.train()
            trainer = Trainer(model=model,
                              device=params.device,
                              train_dataset=train_dataset,
                              val_dataset=val_dataset,
                              learning_rate=params.lr,
                              batch_size=params.batch_size)

            trainer.train(params.epochs)

            # auc score for fold
            preds = []
            y_true = []

            with torch.no_grad():
                model.eval()
                for img, label in test_dataset:
                    img = img.unsqueeze(0).cuda()
                    logit = model(img)
                    pred = torch.nn.functional.sigmoid(logit)
                    preds.append(pred.item())

                    y_true.append(label)

            fpr, tpr, thresholds = metrics.roc_curve(y_true, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            mlflow.log_metric("auc", auc)
