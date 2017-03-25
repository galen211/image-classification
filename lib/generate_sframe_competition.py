import graphlab as gl
from os.path import join
import pandas as pd
import numpy as np

# set paths
sframe_dir = '/Users/galen/Desktop/image_classification/data/sframe'
img_dir = '/Users/galen/Desktop/image_classification/data/img_resize_competition'
sift_file = '/Users/galen/Desktop/image_classification/data/csv_features/sift_features_competition.csv'

# resized image
path_resize = join(img_dir,'img_resize')

sf_resized = gl.image_analysis.load_images(path_resize, format='JPG', with_path=False, random_order=False)

# create data frame object from csv
df = pd.read_csv(sift_file, dtype=np.float32)
df = df.transpose()
sdf = df.values.tolist()

# create SFrame object
sift = gl.SFrame({'sift_features': sdf})
sift.add_columns(sf_resized['image'], namelist=['image'])
sift = sift.add_row_number()
sift.save(join(sframe_dir, 'sift_features_competition'))