"""Reproduce JMVAE results on MNIST-Labels dataset"""

import argparse

from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import JMVAE, JMVAEConfig

from multivae.trainers import BaseTrainer, BaseTrainerConfig

from architectures import *


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=8)
args = parser.parse_args()

######################################################
### Dataset

train_set = MnistLabels(data_path="./data", split="train", download=True)
test_set = MnistLabels(data_path="./data", split="test", download=True)

######################################################
### Model

model_config = JMVAEConfig(
    n_modalities=2,
    latent_dim=64,
    input_dims=dict(images=(1, 28, 28), labels=(1,)),
    decoders_dist=dict(images="bernoulli", labels="categorical"),
    alpha=0.1,
    warmup=200,
    uses_likelihood_rescaling=False,
)


model = JMVAE(
    model_config=model_config,
    encoders=dict(
        images=EncoderImage(model_config.latent_dim),
        labels=EncoderLabels(model_config.latent_dim),
    ),
    decoders=dict(
        images=ImageDecoder(model_config.latent_dim),
        labels=LabelsDecoder(model_config.latent_dim),
    ),
    joint_encoder=JointEncoder(model_config.latent_dim),
)

#########################################################
### Training

training_config = BaseTrainerConfig(
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    num_epochs=500,
    start_keep_best_epoch=model_config.warmup,
    steps_predict=5,
    seed=args.seed,
    learning_rate=1e-3,
    output_dir="./reproduce_jmvae",
)

# If you want to use wandb, uncomment the lines below
callbacks = None
# wandb_ = WandbCallback()
# wandb_.setup(training_config, model_config, project_name="reproduce_jmvae")
# callbacks = [wandb_, ProgressBarCallback()]

trainer = BaseTrainer(
    model,
    train_set,
    training_config=training_config,
    callbacks=callbacks,
    checkpoint=None,
)

trainer.train()


############################################################
### Validating


model = trainer._best_model

ll_config = LikelihoodsEvaluatorConfig(K=1000, unified_implementation=False)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
