import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import copy
import os
import yaml

from torch.utils.data.dataloader import DataLoader
from dataset import TrainDataset, EvalDataset
from tqdm import tqdm
from utils import AverageMeter, plot_loss_metrics, Dashboard, count_model_parameters, estimate_model_size
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
from model import AOD_CNN



def main(config):
    
    NUM_EPOCH = config['epochs']
    LOSS_FUNC = config['loss']
    LR = config['lr']
    BATCH_SIZE = config['batch_size']
    PRED_DIR = config['prediction_dir']
    TRAIN_DIR = config['training_dir']
    VALID_DIR = config['validation_dir']
    GPU_DEVICE = config['gpu_device']
    MODEL = config['model']
    
    """Main function for training and evaluating the AOD_CNN model."""
    prediction_dir = PRED_DIR
    
    start_training_date = datetime.datetime.now()
    logger.info("Start training session '{}'".format(start_training_date))

    model = AOD_CNN()
    
    logger.info("Number of GPU(s) {}: ".format(torch.cuda.device_count()))
    logger.info("GPU(s) in use {}: ".format(GPU_DEVICE))
    logger.info("------")
    
    # Set device (GPU or CPU) and move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device='cuda')
    nb_parameters = count_model_parameters(model=model)  # Count model parameters
    
    # Loss function and optimizer
    if LOSS_FUNC == "MSE":    
        criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load training and validation data
    train_path = pd.read_csv(TRAIN_DIR)
    valid_path = pd.read_csv(VALID_DIR)
    logger.info("Number of Training data: {}".format(len(train_path)))
    logger.info("Number of Validation data: {}".format(len(valid_path)))
    
    # Initialize datasets and data loaders
    train_dataset = TrainDataset(df_path=train_path, augment=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    eval_dataset = EvalDataset(df_path=valid_path)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)
    
    metrics_dict = {}
    
    # Training loop
    for epoch in range(NUM_EPOCH):
        train_losses = AverageMeter()
        eval_losses = AverageMeter()
        model.train()  # Set model to training mode
        
        # Progress bar
        with tqdm(total=(len(train_dataset) - len(train_dataset) % 250), ncols=100, colour='#3eedc4') as t:
            t.set_description(f'Epoch: {epoch}/{NUM_EPOCH - 1}')
        
            # Training step
            for data in train_dataloader:
                optimizer.zero_grad()  # Clear gradients
                images_inputs, targets = data
                images_inputs = images_inputs.to(device)
                targets = targets.to(device).unsqueeze(1)  # Adjust target dimensions
                preds = model(images_inputs)
                
                loss_train = criterion(preds.to(torch.float32), targets.to(torch.float32))  # Compute loss
                loss_train.backward()  # Backpropagation
                optimizer.step()  # Optimizer step
                
                train_losses.update(loss_train.item(), len(images_inputs))  # Track training loss
                t.set_postfix(loss=f'{train_losses.avg:.6f}')
                t.update(len(images_inputs))

        # Evaluation step
        model.eval()  # Set model to evaluation mode
        targets = []
        preds = []
        
        for data in eval_dataloader:
            images_inputs, target = data
            images_inputs = images_inputs.to(device)
            target = target.to(device).unsqueeze(1)
            
            with torch.no_grad():
                pred = model(images_inputs)  # Get model predictions
                eval_loss = criterion(pred.to(torch.float32), target.to(torch.float32))  # Compute validation loss
                
            eval_losses.update(eval_loss.item(), len(images_inputs))  # Track evaluation loss
            targets.append(target.detach().cpu().numpy())  # Store target values
            preds.append(pred.detach().cpu().numpy())  # Store predictions

        # Log and store evaluation metrics
        mse = eval_losses.avg
        logger.info(f'Epoch {epoch} Eval MSE - Loss: {mse}')
        
        metrics_dict[epoch] = {"MSE": mse, "loss_train": train_losses.avg, "loss_eval": eval_losses.avg}

        # Plot and save metrics
        plots = f'loss_plot_e{NUM_EPOCH}_b{BATCH_SIZE}_{MODEL}.png'
        plot_loss_metrics(metrics=metrics_dict, save_path=prediction_dir, plot_version=plots)

        # Save validation metrics as a DataFrame
        df_metrics = pd.DataFrame(metrics_dict).T
        df_mean_metrics = df_metrics.mean().T

        # if epoch == 0:
        #     df_val_metrics = pd.DataFrame(columns=df_mean_metrics.index)
        # df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics], ignore_index=True)

        # Generate and save dashboard with metrics
        # dashboard = Dashboard(df_val_metrics)
        # dashboard.generate_dashboard()
        # dashboard.save_dashboard(directory_path=prediction_dir)

        # Save best model weights based on evaluation performance
        if epoch == 0 or mse < best_mse:
            best_epoch = epoch
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
    
    logger.info(f'Best epoch: {best_epoch}, Best MSE: {best_mse}')
    torch.save(best_weights, os.path.join(prediction_dir, 'best_e{NUM_EPOCH}_b{BATCH_SIZE}_{MODEL}.pth'))

    # Log training duration and save final metrics
    end_training_date = datetime.datetime.now()
    training_duration = end_training_date - start_training_date
    logger.info(f'Training Duration: {training_duration}')
    
    # df_val_metrics['Training_duration'] = training_duration
    # df_val_metrics['nb_parameters'] = nb_parameters
    model_size = estimate_model_size(model)
    logger.info(f'Model size: {model_size}')
    # df_val_metrics['model_size'] = model_size
    # df_val_metrics.to_csv(os.path.join(prediction_dir, 'valid_metrics_log.csv'))

    
if __name__ == '__main__':
    
    
# Load YAML configuration
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)  # Load the YAML file safely
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML file: {exc}")  # Log error if YAML parsing fails
        
    main(config)