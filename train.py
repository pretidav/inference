from sys import path_importer_cache
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import os


def get_dataset():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))
    return train_images, train_labels, test_images, test_labels

def train_model(epochs, train_images, train_labels, test_images, test_labels):
    model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, name='Dense')
    ])
    model.summary()

    model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))
    print('\nTest loss: {}'.format(test_loss))
    return model

def export_model(model, model_dir, version):
    export_path = os.path.join(model_dir, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    print('\nSaved model: {}'.format(export_path))

def start_serving(model_dir, version): 
    export_path = os.path.join(model_dir, str(version))
    os.system('docker pull tensorflow/serving')
    os.system(' docker run -p 8501:8501 -v '+str(os.path.abspath(model_dir))+'/:/serve/model -e MODEL_NAME=model -e MODEL_BASE_PATH=/serve/ tensorflow/serving &') 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='./trained')
    parser.add_argument("--version", default=1)
    parser.add_argument("--epoch", default=10)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--serve", action='store_true')
    
    args = parser.parse_args()

    if args.train:
        train_images, train_labels, test_images, test_labels = get_dataset()
        model                     = train_model(epochs       = 10,  
                                                train_images = train_images,
                                                train_labels = train_labels,
                                                test_images  = test_images,
                                                test_labels  = test_labels)
        export_model(model      = model,
                    model_dir   = args.path,
                    version     = args.version)
    if args.serve: 
        start_serving(model_dir=args.path,
                      version = args.version)
