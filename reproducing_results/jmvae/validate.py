import argparse

from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel

model = AutoModel.load_from_hf_hub("asenella/reproduce_jmvae_seed_1", allow_pickle=True)
# If you want to evaluate on your own trained model, uncomment the lines below
# model_path = enter_trained_model_path
# model = AutoModel.load_from_folder(model_path)

test_set = MnistLabels(data_path="./", split="test", download=True)

ll_config = LikelihoodsEvaluatorConfig(
    K=1000,
    unified_implementation=False,  # Use the paper version of the likelihood, computing only the images likelihood and not the joint
)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
