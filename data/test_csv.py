import os
import shutil
import random
import glob 
import natsort
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


# Paths
source_dir = '/home/eouser/programming/side_projects/aod_esti/data/test_images'
images = natsort.natsorted(glob.glob(os.path.join(source_dir, "*.tif"), recursive=False))


df = pd.DataFrame()
df['image_path'] = images


# Save the shuffled data to CSV files
df.to_csv('/home/eouser/programming/side_projects/aod_esti/data/test_images.csv')

