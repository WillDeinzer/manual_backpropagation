import numpy as np
from dataset import get_mnist_images
from model import MLPModel
from loss import CrossEntropyLoss
from optimizers import SGD, Adam, AdamW
from training import training_loop
from sklearn.model_selection import train_test_split

def main():
    train_images, train_labels, test_images, test_labels = get_mnist_images()
    train_images = train_images.reshape(60000, 784)
    test_images = test_images.reshape(10000, 784)
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    model = MLPModel()
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters)
    training_loop(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, 20, 32)

    predictions = np.argmax(model.forward(test_images), axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    accuracy = (np.sum(predictions == test_labels) / len(predictions)) * 100
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
