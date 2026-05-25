import torch
from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback
from torch import nn
import argparse

from architectures import *

###### Model Config ########
from multivae.models.mmvaePlus import MMVAEPlus, MMVAEPlusConfig

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--K", type=int, default=1)
args = parser.parse_args()

modalities = ["m0", "m1", "m2", "m3", "m4"]

model_config = MMVAEPlusConfig(
    n_modalities=5,
    K=args.K,
    decoders_dist={m: "laplace" for m in modalities},
    decoder_dist_params={m: dict(scale=0.75) for m in modalities},
    prior_and_posterior_dist="laplace_with_softmax",
    beta=2.5,
    modalities_specific_dim=32,
    latent_dim=32,
    input_dims={m: (3, 28, 28) for m in modalities},
    learn_shared_prior=False,
    learn_modality_prior=True,
)

encoders = {
    m: Enc(model_config.modalities_specific_dim, ndim_u=model_config.latent_dim)
    for m in modalities
}
decoders = {
    m: Dec(model_config.latent_dim + model_config.modalities_specific_dim)
    for m in modalities
}

model = MMVAEPlus(model_config, encoders, decoders)


######## Dataset #########

train_data = MMNISTDataset(data_path="./data", split="train", download=True)
test_data = MMNISTDataset(data_path="./data", split="test", download=True)


########## Training #######
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig

training_config = BaseTrainerConfig(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_epochs=50 if model_config.K == 10 else 150,
    learning_rate=1e-3,
    output_dir=f"./reproduce_mmvaep/K__{model_config.K}/seed__{args.seed}",
    steps_predict=5,
    optimizer_cls="Adam",
    optimizer_params=dict(amsgrad=True),
    seed=args.seed,
)

# Set up callbacks
callbacks = None

#####  Uncomment the lines below if you want to use wandb monitoring
# wandb_cb = WandbCallback()
# wandb_cb.setup(training_config, model_config, project_name="reproducing_mmvae_plus")
# callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=None,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()

#### Validation ####
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.fids import FIDEvaluator, FIDEvaluatorConfig


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h


# Make sure, you have the classifiers in the right path
def load_mmnist_classifiers(data_path="./data/clf", device="cuda"):
    clfs = {}
    for i in range(5):
        fp = data_path + "/pretrained_img_to_digit_clf_m" + str(i)
        model_clf = ClfImg()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs


config = CoherenceEvaluatorConfig(batch_size=128)

CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(device=model.device),
    output=trainer.training_dir,
    eval_config=config,
).eval()

# Make sure you have the FID weights in the right path
config = FIDEvaluatorConfig(batch_size=512, inception_weights_path="./data")

fid = FIDEvaluator(
    model, test_data, output=trainer.training_dir, eval_config=config
).eval()
