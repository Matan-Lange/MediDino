# MedDino 

## Description
In this study, we explored the potential of adapting DinoV2, a self-supervised vision transformer, to medical imaging tasks.
By pre-training DinoV2 on domain- specific medical datasets and comparing its performance to RadImageNet, a
CNN-based medical imaging model, we demonstrated that self-supervised pre-trained transformers models have the potential 
to serve as competitive foundation models in the healthcare domain. While DinoV2 initially lagged behind RadImageNet in some tasks, particularly in Thyroid imaging, 
the model showed stronger performance after domain-specific pre-training, narrowing the performance gapin datasets such as ACL and Meniscus MRI.

![image](https://github.com/user-attachments/assets/e9a05dc6-6e8e-4ee6-8249-42f8b857503c)


## Getting Started
To get started with this project, follow these steps:

1. Clone the repository:
2. Install the required dependencies:
    ```bash
    pip install -r MediDino/dinov2-main/requirements.txt
    ```
Installation Requirements:
- Python 3.10
- Other dependencies listed in `requirements.txt`


## DinoV2 SSL Traning 
To train the DinoV2 SSL model,on your own dataset, follow these steps:
1. change the `train.dataset_path` in `dinov2/dinov2/configs/ssl_default_config.yaml`  to the path of your dataset.
dataset should images inside a folder, give the path of the folder in `train.dataset_path`.



## Fine-tuning DinoV2 
To fine-tune the DinoV2 model on your own dataset, run the following command:
if you want to load custom DINOV2 you can use `--ssl_model_path` to load the model.
else use dino_version to load the dinov2 model version from torch hub.
  ```bash
 python finetune/train.py --dataset meniscus --ssl_model_path teacher_checkpoint.pth --epochs 30 --lr 1e-05 --batch_size 32 --neg_label n --pos_label y
```
