from dataset import get_batches
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from optimizers import SGD, Adam, AdamW

def plot_loss(losses, optimizer_type):
    plt.figure(figsize=(10, 7))
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    x_data = np.arange(len(losses)) + 1
    plt.plot(x_data, losses, c='b')
    plt.title('Loss per batch')
    plt.savefig(f"plots/loss_plot_{optimizer_type}.png")

def training_loop(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    losses = []
    for epoch in tqdm(range(epochs), desc=f"Training"):
        epoch_loss = 0
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            logits = model.forward(X_batch)

            loss = loss_fn.forward(logits, y_batch)
            epoch_loss += loss
            losses.append(loss)

            model.backward(loss_fn.backward())

            optimizer.step()

        print(f"Epoch {epoch + 1} | Loss: {epoch_loss}")

        # Validation

        predictions = np.argmax(model.forward(X_val), axis=1)
        val_labels = np.argmax(y_val, axis=1)
        accuracy = (np.sum(predictions == val_labels) / len(predictions)) * 100
        print(f"Validation accuracy: {accuracy:.2f}%")

    if isinstance(optimizer, SGD):
        plot_loss(losses, 'sgd')
    elif isinstance(optimizer, Adam):
        plot_loss(losses, 'adam')
    else:
        plot_loss(losses, 'adamW')