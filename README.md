# MedDino 

## Description
A brief description of what your project does.

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

