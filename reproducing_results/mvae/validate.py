from pathlib import Path

import torch
from multivae.data.datasets.celeba import CelebAttr
from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel
from torch import nn

model = AutoModel.load_from_hf_hub("asenella/reproduce_mvae_mnist_1", allow_pickle=True)

# Uncomment the line below to evaluate on your own model
# model = AutoModel.load_from_folder("path_to_your_trained_model")


test_set = MMNISTDataset(data_path="./data", split="test", download=True)


ll_config = LikelihoodsEvaluatorConfig(batch_size=128, K=1000, batch_size_k=500)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
