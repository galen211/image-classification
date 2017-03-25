import cv2
from os import listdir
from os.path import isfile, join


def process(filename, key):
    image = cv2.imread(join(rawimg_path,filename))
    print(image.shape)
    r = 100.0 / image.shape[1]
    dim = (100, int(image.shape[0] * r))
    imageresized = cv2.resize(image, (256, 256), dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(join(resized_path,'image_{0:04d}.jpg'.format(key)), imageresized)
    print('image_{0:04d}.jpg'.format(key))

# image path
rawimg_path = '/Users/galen/Desktop/image_classification/data/img_raw_competition'
resized_path = '/Users/galen/Desktop/image_classification/data/img_resize_competition'

file_names = [f for f in listdir(rawimg_path) if isfile(join(rawimg_path, f))]
for i in range(0, len(file_names)):
    process(file_names[i], i+1)