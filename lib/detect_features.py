import graphlab as gl
from os.path import join

resize_path = '/Users/galen/Desktop/image_classification/data/img_resize'
edge_path = '/Users/galen/Desktop/image_classification/data/img_edge'
sobel_path = '/Users/galen/Desktop/image_classification/data/img_sobel'
sframe_path = '/Users/galen/Desktop/image_classification/data/sframe'
label_file = '/Users/galen/Desktop/image_classification/data/csv_labels/labels.csv'

ext_1 = gl.feature_engineering.DeepFeatureExtractor(features = 'image',
                                                        model='auto')
ext_2 = gl.feature_engineering.DeepFeatureExtractor(features = 'image',
                                                        model='auto')
ext_3 = gl.feature_engineering.DeepFeatureExtractor(features='image',
                                                        model='auto')

edge = gl.image_analysis.load_images(edge_path)
sobel = gl.image_analysis.load_images(sobel_path)
resize = gl.image_analysis.load_images(resize_path)

labels = gl.SFrame.read_csv(label_file)
labels = labels.apply(lambda x: 'chicken' if x['V1']==0 else 'dog')

edge.add_column(labels,name='label')
sobel.add_column(labels,name='label')
resize.add_column(labels,name='label')
edge = edge.add_row_number()
sobel = sobel.add_row_number()
resize = resize.add_row_number()

ext_1 = ext_1.fit(edge)
ext_2 = ext_2.fit(sobel)
ext_3 = ext_3.fit(resize)

edge_features = ext_1.transform(edge)
sobel_features = ext_2.transform(sobel)
resize_features = ext_3.transform(resize)

edge_features.save(join(sframe_path, 'edge_features'))
sobel_features.save(join(sframe_path, 'sobel_features'))
resize_features.save(join(sframe_path, 'resize_features'))