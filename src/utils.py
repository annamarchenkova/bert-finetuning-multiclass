import requests
import re
import os
import glob
import torch
import numpy as np

# import sys
import yaml

# import pkgutil
from utils_project_dirs import PROJECT_DIR

# from src.config import *

# sys.path.append("../")

################################  GENERIC  #################################
############################################################################

def load_config(cnf_dir=PROJECT_DIR, cnf_name="config.yml"):
    """
    load the yaml file
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)


def download_proxy(url):
    """
    Return the proxy.

    Args:
    url (str): url to downoad the proxy (default in config)

    Returns:
    (str): proxy value
    """
    text = requests.get(url, verify=False).text
    proxy = re.findall(r"PROXY (proxy.*)\;\"", text)[0].replace(";", "")

    return "http://" + proxy


def list_all_filepaths(common_dir: str, folder=None, extension: str = ".txt"):
    """Get a list of all filepaths with a provided extention

    Args:
        common_dir (str): path to the folder
        folder (str): subfolder name
        extension (str, optional): file extention. Defaults to ".txt".

    Returns:
        _type_: _description_
    """
    path = os.path.join(common_dir, "**\\", "*{extension}")
    if folder:
        path = os.path.join(common_dir, folder, "**\\", f"*{extension}")
    filenames = glob.glob(path)

    if not filenames:
        path = os.path.join(
            common_dir, folder, f"*{extension}"
        )  # search only in the main directory
        filenames = glob.glob(path)

    return filenames

################################  PLOTTING  ################################
############################################################################

def despine_ax(ax, right=False, top=False, left=False, bottom=True):
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    
    
#################################  MODELS  #################################
############################################################################
    
def accuracy_per_class(preds, labels, labels_map):
    label_dict_inverse = {v: k for k, v in labels_map.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        
def evaluate(dataloader_val, model, device):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals



