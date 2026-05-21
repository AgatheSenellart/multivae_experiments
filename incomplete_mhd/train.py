from eval_model import eval_model
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
import hydra
from hydra.utils import instantiate
from multivae.trainers import BaseTrainer,MultistageTrainer
from multivae.trainers.add_dcca_trainer import AddDccaTrainer
from torch.utils.data import random_split
import torch
from hydra.core.hydra_config import HydraConfig

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


@hydra.main(config_path='configs',config_name="config")
def train_model(args):

    
    #Architectures
    encoders = dict(
        image =instantiate(args.encoders.image), 
        audio = instantiate(args.encoders.audio),
        trajectory = instantiate(args.encoders.trajectory)
    ) 

    decoders = dict(
        image = instantiate(args.decoders.image),
        audio = instantiate(args.decoders.audio),
        trajectory = instantiate(args.decoders.trajectory)
    )

    if HydraConfig.get().runtime.choices.model == "jnfdcca":

        dcca_networks = dict(
            image = instantiate(args.dcca_networks.image), 
            audio = instantiate(args.dcca_networks.audio),
            trajectory = instantiate(args.dcca_networks.trajectory)
        )
        joint_encoder = instantiate(args.joint_encoder)
        model = instantiate(args.model, joint_encoder=joint_encoder,dcca_networks=dcca_networks, decoders=decoders)
    else:

        model = instantiate(args.model, encoders=encoders, decoders=decoders)


    # Training configuration

    trainer_config = instantiate(args.trainer)

    train_set = instantiate(args.datasets.train_set)
    
    test_set = instantiate(args.datasets.test_set)

    train, val = random_split(train_set, [5/6,1/6], generator=torch.Generator().manual_seed(args.seed))

    logger.info(f"Training set size: {len(train_set)}")
    

    # Set up callbacks
    wandb_cb = WandbCallback()
    wandb_cb.setup(trainer_config, model.model_config, project_name=args.wandb_project)
    wandb_cb.run.config.update(dict(args))

    callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

    if HydraConfig.get().runtime.choices.model == "jnfdcca":
        trainer = AddDccaTrainer(
            model=model, 
            train_dataset = train, 
            eval_dataset=val, 
            training_config=trainer_config, 
            callbacks=callbacks,
        )
    
    if HydraConfig.get().runtime.choices.model == "jnf":
        trainer = MultistageTrainer(
            model=model, 
            train_dataset = train, 
            eval_dataset=val, 
            training_config=trainer_config, 
            callbacks=callbacks,
        )

    else:
        trainer = BaseTrainer(
            model = model, 
            train_dataset=train, 
            eval_dataset=val,
            training_config=trainer_config, 
            callbacks=callbacks,
        )

    # Train 
    trainer.train()
    model = trainer._best_model

    # Push to HuggingFaceHub
    # save_to_hf(model, args)


    # Validate
    eval_model(trainer_config.output_dir, model, test_set, wandb_cb.run.path)

if __name__ == "__main__":
    train_model()

