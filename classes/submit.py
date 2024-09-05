import argparse
import torch

import pandas as pd
from loguru import logger
from dataset import TestDataset
from torch.utils.data.dataloader import DataLoader

from model import AOD_CNN
import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    
    model_path = '/home/eouser/programming/side_projects/aod_esti/prediction/best100.pth'
    
    model = AOD_CNN()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    logger.info(f"Model is on Cuda: {next(model.parameters()).is_cuda}")
    
    dfs = pd.read_csv("/home/eouser/programming/side_projects/aod_esti/data/sample_answer.csv", header=None)
    
    submit_path = pd.read_csv('/home/eouser/programming/side_projects/aod_esti/data/test_images.csv')
    
    test_dataset = TestDataset(df_path=submit_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    preds_submit = []
    
    for index, data in enumerate(test_dataloader):
    
        images_inputs = data
        images_inputs = images_inputs.to(device)
    
        with torch.no_grad():

            pred = model(images_inputs)
            
        pred = torch.squeeze(pred,0)
        pred = pred.detach().cpu().numpy()
        pred = pred[0]
    
        preds_submit.append(pred)

    dfs[1] = preds_submit
    dfs.to_csv("submit100.csv", header=False, index=False)
    
if __name__ == '__main__':
    main()