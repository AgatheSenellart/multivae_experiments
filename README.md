

# Setup

Install the MultiVae package.
```
git clone https://github.com/AgatheSenellart/multimodal_vaes.git
cd multimodal_vaes
pip install -e .
```


# Comparison on Incomplete MCAR PolyMNIST

When running the comparison on PolyMNIST, you will have the following tree structure. 
The *data* folder will be downloaded automatically when running an experiment for the first time.
```
__multivae_experiments
    |__comparison_on_mnist
        |__ config
        |__ config_only_incomplete
        |__ ...
    |__reproducing_results
        |__ ...
    |__example_mmvae_plus
    |__data
        |__ MMNIST
        |__ clf
        |__ pt_inception-2015-12-05-6726825d.pth
```
## Training
        
The folders `config` and `config_only_incomplete` contains the information of the missing ratio parameter $(1-\eta)$, 
`keep_incomplete` variable (True, if the training takes the incomplete samples into consideration, False if those are discarded) and the seed. 

To launch the training of a model with the config contained in f1.json, move into the multivae_experiments folder and run:

```bash
python comparison_on_mmnist/mvae.py --param_file comparison_on_mmnist/config/f1.json
```

You can change the model, and the configuration file. 
For aggregated models (mvae, mmvae, mopoe, mvtcae), all configurations in `config` are possible, for the joint encoder models (jmvae, jnf, jnfdcca) only the configuration in `config_only_incomplete` are possible, therefore the command you must use is :
```bash
python comparison_on_mmnist/jmvae.py --param_file comparison_on_mmnist/config_only_incomplete/f1.json
```


## Evaluate models and compute metrics 

To compute metrics with the pretrained models available on Hugging Face Hub, you will need to install huggingface :

```bash
pip install huggingface_hub
````

Move into the multivae_experiments folder and run
```bash
python comparison_on_mmnist/eval_hf_models.py --param_file comparison_on_mmnist/config/f1.json --model_name MVAE
```

You can choose the configuration file and the name of the model, in order to download the chosen model from the Hub. 
Possible options for models : JMVAE, MMVAE, MVAE,MoPoE, MVTCAE, JNF, JNFDcca. Once again be careful that for joint encoder models, only the configurations in `config_only_incomplete` are possible.

You can also run the evaluation for a model trained on your own, using the `eval_local_models.py` script. 
You just need to provide the path to your trained model at the beginning of the file:
```python
model_path = "path_to_your_model"
```
and then run 
```bash
python comparison_on_mmnist/eval_local_models.py 
```

# Reproduced experiments from the original papers 

For each model, scripts are available in the reproducing_results folder.

## Setup
Make sure to download pretrained classifiers for evaluation whenever it is needed and place it in the `multivae_experiments/data` folder.

|Dataset| Path to download classifiers|
|:--:|:--:|
|Mnist-SVHN| https://huggingface.co/asenella/mnist_svhn_classifiers/tree/main|
|MMNIST |https://zenodo.org/record/4899160#.ZGeXzy0isf_|

# MMVAE+ on partial data example

Move into the multivae_experiments folder and run:
```bash 

python example_mmvae_plus/mmvae_plus.py --param_file comparison_on_mmnist/config/f1.json
```
You can change the configuration file number (f2.json, f3.json), to change the context of the experiments; the parameter $\eta$, the
`keep_incomplete` variable and the seed.

Once again, if you wish to use wandb, uncomment the following lines in the `mmvae_plus.py`file:

```python
##### Set up callbacks: Uncomment the following lines to use wandb
callbacks = None
# wandb_cb = WandbCallback()
# wandb_cb.setup(trainer_config, model_config)
# wandb_cb.run.config.update(args.__dict__)
# callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]
```

# Experiments on the incomplete MAR MHD dataset

For this experiment, scripts are located in the `incomplete_mhd` folder. 
We use hydra to manage the different configurations in these experiments, so make sure to install it ! 
You can find info [here](https://hydra.cc/docs/intro/).

You can then train any model using the command line: 
```bash
python incomplete_mhd/train.py datasets=incomplete keep_incomplete=True model=mvtcae
````

# Contact

If you have any question, don't hesitate to reach out at agathe.senellart@inria.fr !