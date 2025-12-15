import numpy as np
from dataset import get_mnist_images
from model import MLPModel
from loss import CrossEntropyLoss
from optimizers import SGD, Adam
from training import training_loop

def main():
    train_images, train_labels, test_images, test_labels = get_mnist_images()
    train_images = train_images.reshape(60000, 784)
    test_images = test_images.reshape(10000, 784)
    model = MLPModel()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters, lr=0.05)
    training_loop(model, loss_fn, optimizer, train_images, train_labels, 20, 32)

    predictions = np.argmax(model.forward(test_images), axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    accuracy = (np.sum(predictions == test_labels) / len(predictions)) * 100
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
