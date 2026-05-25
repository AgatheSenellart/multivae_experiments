"""Define architectures for the MNIST-Labels experiments"""

import torch
from multivae.models.nn.default_architectures import (
    BaseDecoder,
    BaseEncoder,
    ModelOutput,
    BaseJointEncoder,
)


class EncoderImage(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )
        self.embedding_layer = torch.nn.Linear(512, latent_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, latent_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        h = torch.flatten(x, start_dim=-3, end_dim=-1)
        h = self.layers(h)
        emb = self.embedding_layer(h)
        log_var = torch.log(self.var_layer(h))

        return ModelOutput(embedding=emb, log_covariance=log_var)


class EncoderLabels(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )
        self.embedding_layer = torch.nn.Linear(512, latent_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, latent_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        x = x.squeeze(1)

        h = self.layers(x)
        emb = self.embedding_layer(h)
        log_var = torch.log(self.var_layer(h))

        return ModelOutput(embedding=emb, log_covariance=log_var)


class JointEncoder(BaseJointEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_image = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512), torch.nn.ReLU()
        )

        self.head_labels = torch.nn.Sequential(
            torch.nn.Linear(10, 512), torch.nn.ReLU()
        )

        self.shared_layer = torch.nn.Sequential(
            torch.nn.Linear(512 * 2, 512), torch.nn.ReLU()
        )

        self.embedding_layer = torch.nn.Linear(512, latent_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, latent_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        images = x["images"].flatten(start_dim=-3, end_dim=-1)
        labels = x["labels"].squeeze()

        h1 = self.head_image(images)
        h2 = self.head_labels(labels)
        h = torch.cat((h1, h2), dim=-1)
        h = self.shared_layer(h)

        return ModelOutput(
            embedding=self.embedding_layer(h),
            log_covariance=torch.log(self.var_layer(h)),
        )


class ImageDecoder(BaseDecoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.input_dim = (1, 28, 28)
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 28 * 28),
        )

    def forward(self, z):
        h = self.layers(z)
        shape = (*z.shape[:-1],) + self.input_dim
        return ModelOutput(reconstruction=h.reshape(shape))


class LabelsDecoder(BaseDecoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.input_dim = (10,)
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            torch.nn.Softmax(),
        )

    def forward(self, z):
        h = self.layers(z)
        shape = (*z.shape[:-1],) + self.input_dim
        return ModelOutput(reconstruction=h.reshape(shape))
