"""
Distributed training entrypoint for the MoLEx anti-spoofing model.
Official implementation of the paper: 
"MoLEx: Mixture of Low-Rank Experts for Efficient Fine-Tuning of Self-Supervised Audio Models"
Author: Zihan Pan
"""



import argparse
import json
import os
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data_utils_NEW import (
    CyberDataset,
    CyberEvalDataset,
    gen_cyber_list,
)
from evaluation import compute_nist_eer
from utils import create_optimizer, seed_worker, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def cleanup():
    """Tear down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_train(args):
    """
    Main function.
    Trains, validates, and evaluates the model.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training.")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices detected.")
    device_id = rank % num_devices
    device = torch.device(f"cuda:{device_id}")

    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    fold_id = args.fold
    meta_path = Path(args.meta_dir)
    feat_file = Path(args.feat_file)

    trn_list_path = (meta_path / f"fold{fold_id}_train.tsv")
    dev_trial_path = (meta_path / f"fold{fold_id}_validation.tsv")
    eval_trial_path = (meta_path / f"fold{fold_id}_evaluation.tsv")

    exp_idx = args.exp_idx
    model_tag = "Exp_" + f"{exp_idx}"

    model_tag = output_dir / model_tag    
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / "eval_output"
    
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")


    import importlib
    model_name = model_config['model_name']
    model_class = getattr(importlib.import_module('model_MOE'), model_name)
    print("Run the model:", model_class)

    model = model_class(model_config)

    class_head_param = list(model.decoder.parameters()) + (list(model.featfusion.parameters()) if hasattr(model, 'featfusion') else [])
    lora_adapt_param = (model.get_MOE_param_list() if hasattr(model, 'num_MOE_layer') else [])
    params_backend = [lora_adapt_param, class_head_param]


    model = model.to(device)
    model = DDP(model, device_ids=[device_id],find_unused_parameters=True)


    # define dataloaders
    trn_loader, dev_loader, eval_loader, train_sampler = get_DDP_loader(args, feat_file, trn_list_path,
                                                     dev_trial_path, eval_trial_path,
                                                     args.seed, config)


    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)

    best_dev_eer = float("inf")
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    optimizer, scheduler= create_optimizer(params_backend, optim_config)    

    for epoch in range(config["num_epochs"]):

        running_loss = 0
        running_ortho_loss = 0
        num_total = 0.0
        model.train()

        train_sampler.set_epoch(epoch)

        # set objective (Loss) functions
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)

        for batch_x, batch_y, utt_id in trn_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            optimizer.zero_grad()

            batch_out = model(batch_x) 
            batch_loss = criterion(batch_out, batch_y)

            # add orthogonal loss
            orth_loss = sum(lora_orthogonality_loss(layer.smoe.experts) for layer in model.module.ssl_model.encoder.layers if hasattr(layer, 'smoe')) if any(hasattr(layer, 'smoe') for layer in model.module.ssl_model.encoder.layers) else 0
            batch_loss = batch_loss + orth_loss*0.01


            running_loss += batch_loss.item() * batch_size    
            running_ortho_loss += orth_loss * batch_size 

            batch_loss.backward()

            optimizer.step()   

        running_loss /= num_total       
        running_ortho_loss /= num_total 
      
        valid_loss = produce_evaluation_file(dev_loader, model, device,
                                metric_path / "dev_score.txt")
            

        dev_eer, dev_th = compute_nist_eer(sc_file=metric_path / "dev_score.txt",
                                        output_file=metric_path / "dev_EER_{}epo.txt".format(epoch), printout=False)
        
        scheduler.step()

        if rank == 0:
            if dev_eer <= best_dev_eer:
                best_dev_eer = dev_eer
                torch.save(model.state_dict(),
                    model_save_path / f"epoch_{epoch}_{dev_eer:03.3f}.pth")
             
            save_logs(epoch, scheduler.get_last_lr(), model_tag, running_loss, dev_eer, valid_loss, running_ortho_loss)


    # Evaluation with the best model
    if rank == 0:
        best_checkpoint_path = merge_checkpoint(model_tag, model_save_path)
        
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

        eval_score_path = model_tag / 'eval_output.txt'
        eval_loss = produce_evaluation_file(eval_loader, model, device,
                                eval_score_path)
        eval_eer, _ = compute_nist_eer(sc_file=eval_score_path,
                                    output_file=metric_path / "eval_best.txt")


    cleanup()



def lora_orthogonality_loss(adapters):
    loss = 0
    for adapter in adapters:
        up_weight = adapter.lora_fc.lora_A
        down_weight = adapter.lora_fc.lora_B
        combined_matrix = torch.matmul(up_weight, down_weight)
        # Compute Gram matrix
        gram_matrix = torch.matmul(combined_matrix, combined_matrix.T)
        identity_matrix = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        # Penalize deviation from orthogonality
        loss += F.mse_loss(gram_matrix, identity_matrix)
    return loss

def merge_checkpoint(model_tag, model_save_path):            
    # Path to your validation loss text file
    txt_path = model_tag/'valid_loss.txt'
    losses = read_losses(txt_path)

    # Sort losses and get the epochs with the 5 minimum validation losses
    min_epochs = sorted(losses, key=losses.get)[:5]

    # Path to the directory containing checkpoints
    checkpoint_dir = model_save_path

    # Generate checkpoint file names
    # Assuming the metric value in the filename isn't needed to identify the checkpoint
    checkpoint_paths = [os.path.join(checkpoint_dir, f'epoch_{epoch}_*.pth') for epoch in min_epochs]

    # Find the actual file names (since the metric value is unknown)
    import glob
    actual_checkpoint_paths = []
    for path_pattern in checkpoint_paths:
        # This will get the first file matching the pattern
        actual_checkpoint_paths.extend(glob.glob(path_pattern))

    # Average the checkpoints
    avg_checkpoint = average_checkpoints(actual_checkpoint_paths)

    best_checkpoint_path = checkpoint_dir/'averaged_checkpoint.pth'

    # Saving the averaged checkpoint
    torch.save(avg_checkpoint, best_checkpoint_path)
    return best_checkpoint_path



def save_logs(epoch, current_lr,model_tag,running_loss,dev_eer,valid_loss,running_ortho_loss):

    print("Start training epoch{:03d}".format(epoch))

    with open(model_tag/'loss_history.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {running_loss}\n')
    with open(model_tag/'Orthogonal_loss_history.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {running_ortho_loss}\n')        
    with open(model_tag/'validation_eer_history.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {dev_eer}\n')
    # with open(model_tag/'evaluation_eer_history.txt', 'a') as file:
    #     file.write(f'Epoch {epoch + 1}: {eval_eer}\n')
    with open(model_tag/'learning_rate.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {current_lr[0]}\n')
    with open(model_tag/'valid_loss.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {valid_loss}\n')
    # with open(model_tag/'eval_loss.txt', 'a') as file:
    #     file.write(f'Epoch {epoch + 1}: {eval_loss}\n')  




def get_DDP_loader(
        args,
        feat_file: str,
        trn_list_path: str,
        dev_trial_path: str,
        eval_trial_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    if os.path.exists(trn_list_path):
        trn_keys, trn_labs, trn_paths = gen_cyber_list(meta_file=trn_list_path,
                                                       feat_file=feat_file)
        print("no. training files:", len(trn_keys))

        train_set = CyberDataset(list_ids=trn_keys,
                                 labels=trn_labs,
                                 file_paths=trn_paths)
        # gen = torch.Generator()
        train_sampler = DistributedSampler(train_set)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                num_workers=16, sampler=train_sampler)
    else:
        print('[WARNING] no training file list, it is possible only for evaluation case.')
        trn_loader = None

    if os.path.exists(dev_trial_path):
        dev_keys, dev_labs, dev_paths = gen_cyber_list(meta_file=dev_trial_path,
                                                       feat_file=feat_file)
        print("no. validation files:", len(dev_keys))

        dev_set = CyberEvalDataset(list_ids=dev_keys,
                                   labels=dev_labs,
                                   file_paths=dev_paths)
        # dev_sampler = DistributedSampler(dev_set)
        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,num_workers=16)
    else:
        print('[WARNING] no dev file list, it is possible only for evaluation case.')
        dev_loader = None

    eval_keys, eval_labs, eval_paths = gen_cyber_list(meta_file=eval_trial_path,
                                                      feat_file=feat_file)
    eval_set = CyberEvalDataset(list_ids=eval_keys,
                                labels=eval_labs,
                                file_paths=eval_paths)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,num_workers=16)

    return trn_loader, dev_loader, eval_loader, train_sampler

def produce_evaluation_file(
        data_loader: DataLoader,
        model,
        device: torch.device,
        save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    lab_list = []

        # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    valid_loss = 0.0
    num_total = 0.0

    for i, (batch_x, batch_y, utt_id) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size
        with torch.no_grad():
            # _, batch_out = model(batch_x) # for AASIST
            batch_out = model(batch_x) 
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() # 1 - detect bona, 0 - detect spoof

            batch_loss = criterion(batch_out, batch_y)
            valid_loss = valid_loss + batch_loss.item()*batch_size
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        lab_list.extend(batch_y)
        #print(i, utt_id)

    with open(save_path, "w") as fh:
        for fn, lab, sco in zip(fname_list, lab_list, score_list):
            lab = "bonafide" if lab == 1 else "spoof"
            fh.write(f"{fn}\t{lab}\t{sco}\n")
    # print("Scores saved to {}".format(save_path))

    valid_loss /= num_total

    return valid_loss

# Function to read validation losses from the text file
def read_losses(file_path):
    losses = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Epoch'):
                parts = line.split(':')
                epoch = int(parts[0].split()[1])  # Adjusting for zero-indexing
                loss = float(parts[1].strip())
                losses[epoch] = loss
    return losses

def average_checkpoints(checkpoint_paths):
    # Load all checkpoints and store their state_dicts
    state_dicts = [torch.load(path) for path in checkpoint_paths]

    # Initialize a dictionary to store the averaged parameters
    avg_state_dict = {key: torch.zeros_like(state_dicts[0][key]) for key in state_dicts[0]}

    # Sum and average the parameters
    for key in state_dicts[0]:
        # Convert to float, sum and average the parameters
        avg_state_dict[key] = sum([state_dict[key].float() for state_dict in state_dicts]) / len(state_dicts)

        # Convert back to original data type if necessary
        if state_dicts[0][key].dtype != torch.float32:
            avg_state_dict[key] = avg_state_dict[key].type(state_dicts[0][key].dtype)

    return avg_state_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio deepfake detection system")


    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )  
    parser.add_argument(
        "--meta_dir",
        dest="meta_dir",
        type=str,
        help="processed meta files following cyber_cookies format",
        default="./data/meta/",
    ) 
    parser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        help="fold number",
        default=1,
    )
    parser.add_argument(
        "--feat_file",
        dest="feat_file",
        type=str,
        help="file with all features, follows cyber_cookies format (wav.scp)",
        default="./data/meta/wav.scp",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--SSL_num",
                        type=int,
                        default=12,
                        help="number of the layers in SSL model")    
    parser.add_argument("--pretrain_checkpoint",
                        type=str,
                        default=None,
                        help="the checkpoint path")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_path",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument(
        "--num_gpu",
        action="store_true",
        help="when this flag is given, continue train the model from pre-trained checkpoint")
    parser.add_argument("--exp_idx",
                        type=int,
                        default=0,
                        help="index of running experiment")    
    

    run_train(parser.parse_args())
 
