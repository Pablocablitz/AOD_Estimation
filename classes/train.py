import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import copy
import os

from torch.utils.data.dataloader import DataLoader
from dataset import TrainDataset, EvalDataset
from tqdm import tqdm
from utils import AverageMeter, plot_loss_metrics, Dashboard, count_model_parameters, estimate_model_size
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
from model import AOD_CNN


def main():
    """Main function for training and evaluating the model."""
    prediction_dir = '/home/eouser/programming/side_projects/aod_esti/prediction'
    
    start_training_date = datetime.datetime.now()
    logger.info("Start training session '{}'".format(start_training_date))

    model = AOD_CNN()
    
    logger.info("Number of GPU(s) {}: ".format(torch.cuda.device_count()))
    logger.info("GPU(s) in use {}: ".format(0))
    logger.info("------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device='cuda')
    nb_parameters = count_model_parameters(model=model)
    
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_path = pd.read_csv('/home/eouser/programming/side_projects/aod_esti/data/data_split/train_data/train_output.csv')
    # train_path = train_path[:100]
    valid_path = pd.read_csv('/home/eouser/programming/side_projects/aod_esti/data/data_split/valid_data/val_output.csv')
    # valid_path = valid_path[:100]
    logger.info("Number of Training data: {}".format(len(train_path)))
    logger.info("------")
    logger.info("Number of Validation data: {}".format(len(valid_path)))
    logger.info("------")
    
    train_dataset = TrainDataset(df_path=train_path)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=0)
    
    eval_dataset = EvalDataset(df_path=valid_path)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)
    
    num_epoch = 50
    metrics_dict = {}
    for epoch in range(num_epoch):
        train_losses = AverageMeter()
        eval_losses = AverageMeter()
        model.train()
        
        with tqdm(total=(len(train_dataset) - len(train_dataset) % 250), ncols=100, colour='#3eedc4') as t:
            t.set_description('Epoch: {}/{}'.format(epoch, num_epoch - 1))
        
            for data in train_dataloader:
                optimizer.zero_grad()
                images_inputs, targets = data
                images_inputs = images_inputs.to(device)
                targets = targets.to(device)
                targets = targets.unsqueeze(1)  # Adjust target dimensions
                preds = model(images_inputs)
                
                loss_train = criterion(preds.to(torch.float32), targets.to(torch.float32))
                loss_train.backward()
                optimizer.step()
                
                train_losses.update(loss_train.item(), len(images_inputs))
                t.set_postfix(loss='{:.6f}'.format(train_losses.avg))
                t.update(len(images_inputs))

        # Evaluation phase
        model.eval()
        targets = []
        preds = []
        
        for data in eval_dataloader:
            images_inputs, target = data
            images_inputs = images_inputs.to(device)
            target = target.to(device)
            target = target.unsqueeze(1)

            with torch.no_grad():
                pred = model(images_inputs)
                eval_loss = criterion(pred.to(torch.float32), target.to(torch.float32))
            
            eval_losses.update(eval_loss.item(), len(images_inputs))
            target = target.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            
            targets.append(target)
            preds.append(pred)

        # Log regression metrics (e.g., MSE)
        mse = eval_losses.avg
        logger.info(f'Epoch {epoch} Eval MSE - Loss: {mse}')
        
        metrics_dict[epoch] = { "MSE": mse,
                                'loss_train': train_losses.avg,
                                'loss_eval': eval_losses.avg}
        print(metrics_dict)

        # Plot metrics
        plots = 'loss_plot_50.png'
        plot_loss_metrics(metrics=metrics_dict, save_path=prediction_dir, plot_version = plots)

        # Save validation metrics
        df_metrics = pd.DataFrame(metrics_dict).T
        df_mean_metrics = df_metrics.mean()
        df_mean_metrics = pd.DataFrame(df_mean_metrics).T

        if epoch == 0:
            df_val_metrics = pd.DataFrame(columns=df_mean_metrics.columns)
            df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])
        else:
            df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])
            df_val_metrics = df_val_metrics.reset_index(drop=True)

        dashboard = Dashboard(df_val_metrics)
        dashboard.generate_dashboard()
        dashboard.save_dashboard(directory_path=prediction_dir)

        # Save best model
        if epoch == 0:
            best_epoch = epoch
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
        elif mse < best_mse:  # Minimize MSE
            best_epoch = epoch
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
            
    
    logger.info(f'Best epoch: {best_epoch}, Best MSE: {best_mse}')
    torch.save(best_weights, os.path.join(prediction_dir, 'best50.pth'))
    
    # Final training duration log
    end_training_date = datetime.datetime.now()
    training_duration = end_training_date - start_training_date
    logger.info(f'Training Duration: {training_duration}')
    
    df_val_metrics['Training_duration'] = training_duration
    df_val_metrics['nb_parameters'] = nb_parameters
    model_size = estimate_model_size(model)
    logger.info(f'Model size: {model_size}')
    df_val_metrics['model_size'] = model_size
    df_val_metrics.to_csv(os.path.join(prediction_dir, 'valid_metrics_log.csv'))
    

if __name__ == '__main__':
    main()