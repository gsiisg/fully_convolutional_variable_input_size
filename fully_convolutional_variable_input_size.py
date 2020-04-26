# https://towardsdatascience.com/implementing-a-fully-convolutional-network-fcn-in-tensorflow-2-3c46fb61de3b
import tensorflow as tf
import os
import numpy as np
import glob
import matplotlib
#%matplotlib inline

# ###### Uncomment below to download data the first time ########################

# import os, cv2
# from shutil import copy2
# import numpy as np
# import tensorflow as tf 

# def download_dataset():
    
#     BASE_PATH = tf.keras.utils.get_file('flower_photos',
#                             'http://download.tensorflow.org/example_images/flower_photos.tgz',
#                             untar=True, cache_dir='.')
#     print(f"Downloaded and extracted at {BASE_PATH}")

#     return BASE_PATH

# def split_dataset(BASE_PATH = 'flower_photos', DATASET_PATH = 'dataset', train_images = 300, val_images = 50):
#     # Specify path to the downloaded folder
#     classes = os.listdir(BASE_PATH)

#     # Specify path for copying the dataset into train and val sets
#     os.makedirs(DATASET_PATH, exist_ok=True)

#     # Creating train directory
#     train_dir = os.path.join(DATASET_PATH, 'train')
#     os.makedirs(train_dir, exist_ok=True)

#     # Creating val directory
#     val_dir = os.path.join(DATASET_PATH, 'val')
#     os.makedirs(val_dir, exist_ok=True)    

#     # Copying images from original folder to dataset folder
#     for class_name in classes:
#         if len(class_name.split('.')) >= 2:
#             continue
#         print(f"Copying images for {class_name}...")
        
#         # Creating destination folder (train and val)
#         class_train_dir = os.path.join(train_dir, class_name)
#         os.makedirs(class_train_dir, exist_ok=True)
        
#         class_val_dir = os.path.join(val_dir, class_name)
#         os.makedirs(class_val_dir, exist_ok=True)

#         # Shuffling the image list
#         class_path = os.path.join(BASE_PATH, class_name)
#         class_images = os.listdir(class_path)
#         np.random.shuffle(class_images)

#         for image in class_images[:train_images]:
#             copy2(os.path.join(class_path, image), class_train_dir)
#         for image in class_images[train_images:train_images+val_images]:
#             copy2(os.path.join(class_path, image), class_val_dir)

# def get_dataset_stats(DATASET_PATH = 'dataset'):
#     """
#         This utility gives the following stats for the dataset:
#         TOTAL_IMAGES: Total number of images for each class in train and val sets
#         AVG_IMG_HEIGHT: Average height of images across complete dataset (incl. train and val)
#         AVG_IMG_WIDTH: Average width of images across complete dataset (incl. train and val)
#         MIN_HEIGHT: Minimum height of images across complete dataset (incl. train and val)
#         MIN_WIDTH: Minimum width of images across complete dataset (incl. train and val)
#         MAX_HEIGHT: Maximum height of images across complete dataset (incl. train and val)
#         MAX_WIDTH: Maximum width of images across complete dataset (incl. train and val)

#         NOTE: You should have enough memory to load complete dataset
#     """
#     train_dir = os.path.join(DATASET_PATH, 'train')
#     val_dir = os.path.join(DATASET_PATH, 'val')

#     len_classes = len(os.listdir(train_dir))

#     assert len(os.listdir(train_dir)) == len(os.listdir(val_dir))

#     avg_height = 0
#     min_height = np.inf
#     max_height = 0

#     avg_width = 0
#     min_width = np.inf
#     max_width = 0

#     total_train = 0
#     print('Training dataset stats:\n')
#     for class_name in os.listdir(train_dir):
#         class_path = os.path.join(train_dir, class_name)
#         class_images = os.listdir(class_path)
        
#         for img_name in class_images:
#             h, w, c = cv2.imread(os.path.join(class_path, img_name)).shape
#             avg_height += h
#             avg_width += w
#             min_height = min(min_height, h)
#             min_width = min(min_width, w)
#             max_height = max(max_height, h)
#             max_width = max(max_width, w)
        
#         total_train += len(class_images)
#         print(f'--> Images in {class_name}: {len(class_images)}')
    
#     total_val = 0
#     print('Validation dataset stats:')
#     for class_name in os.listdir(val_dir):
#         class_path = os.path.join(val_dir, class_name)
#         class_images = os.listdir(class_path)
        
#         for img_name in class_images:
#             h, w, c = cv2.imread(os.path.join(class_path, img_name)).shape
#             avg_height += h
#             avg_width += w
#             min_height = min(min_height, h)
#             min_width = min(min_width, w)
#             max_height = max(max_height, h)
#             max_width = max(max_width, w)

#         total_val += len(class_images)
#         print(f'--> Images in {class_name}: {len(os.listdir(os.path.join(val_dir, class_name)))}')

#     IMG_HEIGHT = avg_height // total_train
#     IMG_WIDTH = avg_width // total_train
    
#     print()
#     print(f'AVG_IMG_HEIGHT: {IMG_HEIGHT}')
#     print(f'AVG_IMG_WIDTH: {IMG_WIDTH}')
#     print(f'MIN_HEIGHT: {min_height}')
#     print(f'MIN_WIDTH: {min_width}')
#     print(f'MAX_HEIGHT: {max_height}')
#     print(f'MAX_WIDTH: {max_width}')
#     print()

#     return len_classes, train_dir, val_dir, IMG_HEIGHT, IMG_WIDTH, total_train, total_val

# # if __name__ == "__main__":
    
# BASE_PATH = download_dataset()
# # Number of images required in train and val sets
# train_images = 500
# val_images = 100
# split_dataset(BASE_PATH=BASE_PATH, train_images = train_images, val_images = val_images)
# get_dataset_stats()

# ##### Comment out above if data already downloaded the first time ###########################


def FCN_model(len_classes=5, dropout_rate=0.2):
    inp = tf.keras.layers.Input(shape=(None,1))
    x = tf.keras.layers.Embedding(257,4)(inp)
    
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1)(inp)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 1
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 2
    x = tf.keras.layers.Conv1D(filters=len_classes, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    predictions = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=predictions)
    
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

    return model

model = FCN_model(len_classes=5, dropout_rate=0.1)

# The below folders are created using utils.py
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# If you get out of memory error try reducing the batch size
BATCH_SIZE=8
epochs=1

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = './snapshots'
os.makedirs(checkpoint_path, exist_ok=True)
model_path = os.path.join(checkpoint_path, 'model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_loss_{val_loss:.2f}_val_acc_{val_accuracy:.2f}.h5')


# get list of files
train_files = glob.glob('dataset\\train\\*\\*.*')
val_files = glob.glob('dataset\\val\\*\\*.*')
label_dict = {'daisy':0,
              'dandelion':1,
              'roses':2,
              'sunflowers':3,
              'tulips':4
             }

train_labels = []
for file in train_files:
    train_labels.append(label_dict[file.split('\\')[-2]])
val_labels = []
for file in val_files:
    val_labels.append(label_dict[file.split('\\')[-2]]) 
    
def parser_fn(file_path):
    # read in the file then decode it
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image)
    #image = tf.dtypes.cast(image, tf.float32)
    image = tf.keras.backend.flatten(image)
    # needs the extra 1 as channel
    image = tf.reshape(image, tf.concat([tf.shape(image), [1]],0))
    return image

def create_prefetch_dataset(filenames, labels):
    d = tf.data.Dataset.from_tensor_slices(filenames)
    d = d.map(parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    l = tf.data.Dataset.from_tensor_slices(tf.one_hot(labels, depth=5))
    dataset = tf.data.Dataset.zip((d, l))
    #dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, None, 3),(5)))
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,1),(5)))
    dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

index = np.arange(len(train_files))
np.random.seed(9999)
np.random.shuffle(index)
print(index)
train_files = np.array(train_files)[index]
train_labels = np.array(train_labels)[index]
steps_per_epoch = np.ceil(len(train_labels)/BATCH_SIZE)
validation_size = 100
validation_steps = np.ceil(validation_size/BATCH_SIZE)


train_prefetch_dataset = create_prefetch_dataset(train_files, train_labels)
validation_prefetch_dataset = create_prefetch_dataset(train_files[:validation_size], train_labels[:validation_size])

import time
def benchmark(train_dataset, validation_dataset, num_epochs=1):
    start_time = time.perf_counter()
    model.fit(train_dataset,
              steps_per_epoch = steps_per_epoch,
              validation_steps = validation_steps,
              epochs=num_epochs,
              validation_data=validation_dataset,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)],
    )
    tf.print("Execution time:", time.perf_counter() - start_time)
    
benchmark(train_prefetch_dataset, validation_prefetch_dataset, num_epochs=5)