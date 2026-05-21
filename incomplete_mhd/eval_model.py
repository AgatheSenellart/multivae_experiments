
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.metrics import ReconstructionConfig, Reconstruction
from compute_mfd import compute_mfd
from architectures import Image_Classifier, Sound_Classifier, Trajectory_Classifier
import torch
import os

def eval_model(path,model, test_set, wandb_path=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    classifiers_path = '/home/asenella/scratch/data/MHD/classifiers'

    classifiers = dict(
        image = Image_Classifier(),
        audio = Sound_Classifier(),
        trajectory = Trajectory_Classifier()
        
    )


    state_dicts = dict(
        image = torch.load(os.path.join(classifiers_path, 'best_image_classifier_model.pth.tar'), map_location=device)['state_dict'],
        audio = torch.load(os.path.join(classifiers_path, 'best_sound_classifier_model.pth.tar'), map_location=device)['state_dict'],
        trajectory = torch.load(os.path.join(classifiers_path, 'best_trajectory_classifier_model.pth.tar'), map_location=device)['state_dict'],


    )

    for s in state_dicts:
        classifiers[s].load_state_dict(state_dicts[s])
        classifiers[s].eval()
        
    
    coherence_config = CoherenceEvaluatorConfig(128, wandb_path=wandb_path, num_classes=10,give_details_per_class=True)
    
    CoherenceEvaluator(
        model=model,
        classifiers=classifiers,
        test_dataset=test_set,
        output=path,
        eval_config=coherence_config
        ).eval()
    
    for m in ['SSIM', 'MSE']:
                
        recon_config = ReconstructionConfig(
            batch_size=64,
            wandb_path=wandb_path,
            metric=m
        )
        
        recon_module = Reconstruction(
            model, 
            test_dataset=test_set,
            output=path,
            eval_config=recon_config
        )
        
        recon_module.reconstruction_from_subset(['audio'])
        recon_module.reconstruction_from_subset(['image'])
        recon_module.log_to_wandb()
        recon_module.finish()
    
    compute_mfd(model, wandb_path,path)
