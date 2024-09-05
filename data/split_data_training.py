import os
import shutil
import random
import glob 
import natsort
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


# Paths
source_dir = '/home/eouser/programming/side_projects/aod_esti/data/train_images'
images = natsort.natsorted(glob.glob(os.path.join(source_dir, "*.tif"), recursive=False))

df_csv = pd.read_csv("/home/eouser/programming/side_projects/aod_esti/data/train_answer.csv", header=None)
df_csv.columns = ['file','location','target']
df = pd.DataFrame()
df['image_path'] = images
df['target'] = df_csv['target']

train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

# Save the shuffled data to CSV files
train_df.to_csv('/home/eouser/programming/side_projects/aod_esti/data/data_split/train_data/train_output.csv', index=False)
val_df.to_csv('/home/eouser/programming/side_projects/aod_esti/data/data_split/valid_data/val_output.csv', index=False)

print("Training and validation files have been created.")



