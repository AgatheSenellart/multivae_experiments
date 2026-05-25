import torch
from classifiers import load_mnist_svhn_classifiers
from multivae.data.datasets.mnist_svhn import MnistSvhn
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.load_from_hf_hub("asenella/reproduce_mmvae_model", allow_pickle=True)

# TO perform evaluation on one of your trained model, uncomment the lines below
# model_path = path_to_your_model # set the model_name here
# model = AutoModel.load_from_folder(model_path)

model = model.to(device)
model.device = device

# Make sure you download the pretrained classifiers and set the path right
clfs = load_mnist_svhn_classifiers("path/to/classifiers", device=device)

test_set = MnistSvhn(split="test", data_multiplication=30)
eval_config = CoherenceEvaluatorConfig(batch_size=128, nb_samples_for_joint=10000)
module = CoherenceEvaluator(model, clfs, test_set, eval_config=eval_config)
module.eval()
module.finish()
