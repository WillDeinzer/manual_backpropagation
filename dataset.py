import numpy as np
import tensorflow as tf

def get_mnist_images():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")

    # One hot encoding the labels for use in cross entropy loss
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    return train_images, train_labels, test_images, test_labels

def get_batches(train_images, train_labels, batch_size):
    N = train_images.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        yield train_images[batch_indices], train_labels[batch_indices]

