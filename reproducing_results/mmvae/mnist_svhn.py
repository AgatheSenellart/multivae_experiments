from multivae.data.datasets import MnistSvhn
from multivae.models import MMVAE, MMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback

from architectures import *


###################################################################################
########### Load the dataset, configure model and training ########################

# Dataset
train_set = MnistSvhn(
    data_path="./", split="train", data_multiplication=30, download=True
)
test_set = MnistSvhn(
    data_path="./", split="test", data_multiplication=30, download=True
)


print(f"train : {len(train_set)}, test : {len(test_set)}")
# Model config
model_config = MMVAEConfig(
    n_modalities=2,
    latent_dim=20,
    input_dims={"mnist": (1, 28, 28), "svhn": (3, 32, 32)},
    uses_likelihood_rescaling=True,
    decoders_dist={"mnist": "laplace", "svhn": "laplace"},
    decoder_dist_params={"mnist": {"scale": 0.75}, "svhn": {"scale": 0.75}},
    K=30,
    learn_prior=True,
    prior_and_posterior_dist="laplace_with_softmax",
)


model = MMVAE(
    model_config,
    encoders={
        "mnist": EncoderMNIST(num_hidden_layers=1, config=model_config),
        "svhn": EncoderSVHN(model_config),
    },
    decoders={
        "mnist": DecoderMNIST(num_hidden_layers=1, config=model_config),
        "svhn": DecoderSVHN(config=model_config),
    },
)


# Training

training_config = BaseTrainerConfig(
    learning_rate=1e-3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_epochs=30,
    start_keep_best_epoch=30,  # save the model at each iteration without regards to the loss
    optimizer_cls="Adam",
    optimizer_params={"amsgrad": True},
    steps_predict=1,
    output_dir="./reproduce_mmvae",
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_mmvae")

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()
