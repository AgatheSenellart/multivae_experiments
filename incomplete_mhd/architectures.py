
from math import prod
import torch
# Image
from pythae.models.nn.benchmarks.mnist import Encoder_Conv_VAE_MNIST, BaseDecoder

class Decoder_Conv_AE_MNIST(BaseDecoder):

    def __init__(self, latent_dim):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 28, 28)
        self.latent_dim = latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(latent_dim, 1024 * 4 * 4))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    256, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels= None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:

                out = out.reshape(prod(z.shape[:-1]), 1024, 4, 4)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                if len(z.shape) > 2:
                    output["reconstruction"] = out.reshape(*z.shape[:-1],*out.shape[1:])
                else:
                    output["reconstruction"] = out

        return output
# Sound

import torch
import torch.nn as nn
import torch.nn.functional as F
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput

FRAME_SIZE = 512
CONTEXT_FRAMES = 32
SPECTROGRAM_BINS = FRAME_SIZE//2 + 1


class SoundEncoder(BaseEncoder):
    def __init__(self, output_dim=None):
        super(SoundEncoder, self).__init__()
        self.latent_dim = output_dim

        # Properties
        self.conv_layer_0 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.fc_mu = nn.Linear(2048, output_dim)
        self.fc_logvar = nn.Linear(2048, output_dim)


    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        h = x.view(x.size(0), -1)
        return ModelOutput(
            embedding = self.fc_mu(h),
            log_covariance = self.fc_logvar(h))


class SoundDecoder(BaseDecoder):
    def __init__(self, input_dim):
        super(SoundDecoder, self).__init__()
        self.latent_dim = input_dim

        self.upsampler = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.hallucinate_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
        )


    def forward(self, z):
        
        batch_shape = z.shape[:-1]
        z = z.reshape(prod(batch_shape),-1)
        
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        z = self.hallucinate_0(z)
        z = self.hallucinate_1(z)
        out = self.hallucinate_2(z)
        
        if len(batch_shape) >1:
            out = out.reshape(*batch_shape, *out.shape[1:])
        
        return ModelOutput(reconstruction = F.sigmoid(out))



class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)
    
    
    
# Trajectory Encoder

import torch
import torch.nn as nn


# Symbol
class TrajectoryEncoder(BaseEncoder):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(TrajectoryEncoder, self).__init__()
        self.latent_dim = output_dim

        # Variables
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        enc_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]
            enc_layers.append(nn.Linear(pre, pos))
            enc_layers.append(nn.BatchNorm1d(pos))
            enc_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        # Output layer of the network
        self.fc_mu = nn.Linear(pre, output_dim)
        self.fc_logvar = nn.Linear(pre, output_dim)

        # Print information
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding = self.fc_mu(h),
                           log_covariance= self.fc_logvar(h))


class TrajectoryDecoder(BaseDecoder):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(TrajectoryDecoder, self).__init__()
        self.latent_dim = input_dim

        # Variables
        self.id = id
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        dec_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]

            # Check for input transformation
            dec_layers.append(nn.Linear(pre, pos))
            dec_layers.append(nn.BatchNorm1d(pos))
            dec_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        dec_layers.append(nn.Linear(pre, output_dim))
        self.network = nn.Sequential(*dec_layers)

        # Output Transformation
        self.out_process = nn.Sigmoid()

        # Print information
        print(f'Layers: {dec_layers}')


    def forward(self, x):
        
        batch_shape = x.shape[:-1]
        
        x = x.reshape(prod(batch_shape),-1)
        
        out = self.network(x)
        
        if len(batch_shape)>1:
            out = out.reshape(*batch_shape,*out.shape[1:])
        
        return ModelOutput(reconstruction = self.out_process(out))
    




##### Classifiers #####

class Image_Classifier(nn.Module):
    def __init__(self):
        super(Image_Classifier, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)

        return out
    
    
class Sound_Classifier(nn.Module):
    def __init__(self):
        super(Sound_Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(2048, 128),
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 64),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(),
                                nn.Linear(64, 10))




    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out



class Trajectory_Classifier(nn.Module):
    def __init__(self):
        super(Trajectory_Classifier, self).__init__()

        self.network = nn.Sequential(nn.Linear(200, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 128),
                                  nn.BatchNorm1d(128),
                                  nn.LeakyReLU())
        self.out = nn.Linear(128, 10)


    def forward(self, x):
        h = self.network(x)
        return self.out(h)

class Encoder_Conv_VAE_MNIST(BaseEncoder):
    

    def __init__(self, latent_dim):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024, latent_dim)
        self.log_var = nn.Linear(1024,latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i`
            where i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output



class wrapper_encoder_image(BaseEncoder):
    
    def __init__(self,latent_dim, private_dim):
        super().__init__()
        self.private = Encoder_Conv_VAE_MNIST(latent_dim = latent_dim)
        self.modality_specific = Encoder_Conv_VAE_MNIST(latent_dim = private_dim)
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output
    
class wrapper_encoder_sound(BaseEncoder):
    
    def __init__(self, latent_dim, private_dim):
        super().__init__()
        self.private = SoundEncoder(latent_dim)
        self.modality_specific = SoundEncoder(private_dim)
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output

class wrapper_encoder_traj(BaseEncoder):
    
    def __init__(self, latent_dim, private_dim):
        super().__init__()
        self.private = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=latent_dim)
        self.modality_specific = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=private_dim)
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output
