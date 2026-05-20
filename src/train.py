from os import environ
import sys
import getopt
import json
import fastai
import matplotlib.pyplot as plt

from fastai.vision.all import *
from fastai.vision.models import xresnet
import optuna
from optuna.integration import FastAIPruningCallback
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

def read_json(file_path: str):
    """
    Reads in config file (json) format
    :param (str) path to config
    :ret (dict) the config file
    """
    with open(file_path, 'r') as json_obj:
        this_config = json.load(json_obj)  # read in json object

    return this_config

def get_val(dict, key):
    """
    Gets value from dictionary
    """
    return dict[key] if key in dict else None

def get_tune_hp(trial, dict):
    """
    Gets the distribution and necesary values for hyperparameter tuning using optuna
    from the config file
    :param (trial) trial object from optuna
    :param (dict) dictionary of hyperparameter being tuned
    :ret hyperparameter distribution and values for trial object
    """
    return getattr(trial,dict["dist"])(*dict["vals"].values())


def objective(trial, environ, hps, tune, arch):
    """
    Objective function for optuna
    :param (trial) trial object from optuna
    :environ (dict) dictionary of environment variables
    :hps (dict) dictionary of hyperparameters
    :tune (dict) dictionary of hyperparameters to tune
    :arch (str) CNN architecture to use
    :ret (float) auc score
    """

    #Extract dictionary of hyperparameters to tune from config file
    lr = get_val(tune, 'lr')
    wd = get_val(tune, 'weight_decay')
    moms = get_val(tune, 'momentum')
    batch_size = get_val(tune, 'batch_size')
    freeze_epochs = get_val(tune, 'freeze_epochs')

    start_mom = get_tune_hp(trial, moms)
    min_mom = trial.suggest_float('min_mom', .8, start_mom)
    end_mom = trial.suggest_float('end_mom', min_mom, .999)

    #Specify transforamtions to perform on batch data
    tfms = aug_transforms(
        max_rotate=get_val(hps, 'max_rotate'),
        max_zoom=get_val(hps, 'max_zoom'),
        p_affine=get_val(hps, 'p_affine'),
        size = get_val(hps, 'batch_image_resize'),
        min_scale=get_val(hps, 'min_scale')
    )

    #Define DataBlock for training
    breast_us = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=GrandparentSplitter('train', 'val'),
        #splitter=RandomSplitter(valid_pct=.2, seed = 42),
        get_y=parent_label,
        batch_tfms = tfms,
        item_tfms=Resize(get_val(hps, 'image_size'))
    )

    #Load data into dataloader
    dls = breast_us.dataloaders(get_val(environ, 'data_dir'), bs = get_tune_hp(trial, batch_size)).cuda()

    #Define additional metrics for comparison
    prec = Precision(average='binary')
    recall = Recall(average='binary')
    auc = RocAucBinary()
    f1 = F1Score(average='binary')

    #Create model using architecture defined in config file
    model = getattr(fastai.vision.models.all, get_val(arch, 'backbone'))

    #Create learner object using trial hyperparameters from TPE
    learn = cnn_learner(
      dls,
      model,
      opt_func = getattr(fastai.optimizer,get_val(hps, 'opt')),
      lr = get_tune_hp(trial, lr),
      wd = get_tune_hp(trial, wd),
      moms = (start_mom, min_mom, end_mom),
      metrics = [auc, prec, recall, accuracy, f1],
      loss_func = getattr(fastai.losses, get_val(hps, 'loss_func'))(),
      cbs = [FastAIPruningCallback(trial, monitor = "roc_auc_score"), MixUp()]
    )

    #Disable progress bar and logging info to clean up console output
    with learn.no_bar():
      with learn.no_logging():

        #Train model for specified number of epochs and freeze layers in config file
        learn.fine_tune(get_val(hps, 'num_epochs'), freeze_epochs = get_tune_hp(trial, freeze_epochs))

        #Save model
        model_name = get_val(environ, "model_dir") + "/" + get_val(arch, 'backbone') + '_trial_' + str(trial.number) + ".pkl"
        learn.export(model_name)
        
        #Save additional plots for analysis later
        learn.recorder.plot_loss()
        plt.savefig(get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_loss_trial_' + str(trial.number) + ".png")
        plt.close()
        
        learn.recorder.plot_sched()
        plt.savefig(get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_sched_trial_' + str(trial.number) + ".png")
        plt.close()

        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        plt.savefig(get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_conf_matrix_trial_' + str(trial.number) + ".png")
        plt.close()

        interp.plot_top_losses(16, figsize = (14,14))
        plt.savefig(get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_top_losses_trial_' + str(trial.number) + ".png")
        plt.close()

    #Return auc score for optimization
    return learn.recorder.metrics[0].value.item()

def main(argv):
    """
    Main function to be called from command line
    :param (argv) command line arguments

    """
    #Default values
    config_file = None
    
    #Parse command line arguments to look for config file
    try:
        opts, args = getopt.getopt(argv,"c:")
    except getopt.GetoptError:
        print('train.py -c <config_json>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c"):
            config_file= arg

    #Load config file and parse necessary variables
    config = read_json(config_file)
    environ = get_val(config, 'environ')
    hps = get_val(config, 'hps')
    tune = get_val(config, 'tune') 
    arch = get_val(hps, 'arch')

    #Create Optuna study object
    pruner = optuna.pruners.MedianPruner(n_startup_trials = 5, n_min_trials = 15)
    study = optuna.create_study(study_name = get_val(environ, "study_name"), direction="maximize", pruner=pruner)

    #Optimize hyperparameters using TPE
    study.optimize(lambda trial: objective(trial, environ, hps, tune, arch), n_trials=get_val(environ, "n_trials"))

    #Load best model and images for analysis to variables to be saved to neptune tags
    best_learn = load_learner(get_val(environ, "model_dir") + "/" + get_val(arch, 'backbone') + '_trial_' + str(study.best_trial.number) + ".pkl")
    best_conf = get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_conf_matrix_trial_' + str(study.best_trial.number) + ".png"
    best_loss = get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_loss_trial_' + str(study.best_trial.number) + ".png"
    best_top = get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_top_losses_trial_' + str(study.best_trial.number) + ".png"
    best_sched = get_val(environ, "img_dir") + "/" + get_val(arch, 'backbone') + '_sched_trial_' + str(study.best_trial.number) + ".png"
    
    #Start neptune experiment
    run = neptune.init(
        project="<PROJECT_NAME>",
        api_token="<API_TOKEN>",
        tags = [get_val(arch, 'backbone')]
    )

    #Log study results to neptune
    optuna_utils.log_study_metadata(study, run)

    #Log additional metrics and visualizations to neptune experiment
    run["best/precision"] = best_learn.metrics.items[1].value.item()
    run["best/recall"] = best_learn.metrics.items[2].value.item()
    run["best/accuracy"] = best_learn.metrics.items[3].value.item()
    run["best/f1"] = best_learn.metrics.items[4].value.item()

    run["study/num_trials"]  = get_val(environ, "n_trials")

    run["train/params/Epochs"] = get_val(hps, 'num_epochs')
    run["train/params/Img Size"] = (get_val(hps, 'batch_image_resize'), get_val(hps, 'batch_image_resize'))
    run["train/params/loss_func"] = get_val(hps, 'loss_func')
    run["train/params/opt"] = get_val(hps, 'opt')
    run["train/params/scheduler"] = get_val(hps, 'scheduler')

    run["augments/max_rotate"] = get_val(hps, 'max_rotate')
    run["augments/max_zoom"] = get_val(hps, 'max_zoom')
    run["augments/p_affine"] = get_val(hps, 'p_affine')
    run["augments/batch_image_resize"] = (get_val(hps, 'image_size'), get_val(hps, 'image_size'))

    run['visualizations/conf_matrix'].upload(best_conf)
    run['visualizations/train_loss'].upload(best_loss)
    run['visualizations/top_losses'].upload(best_top)
    run['visualizations/sched'].upload(best_sched)

    #Stop neptune experiment
    run.stop()

#Call main function    
if __name__ == "__main__":
  main(sys.argv[1:])

