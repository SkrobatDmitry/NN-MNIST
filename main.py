import numpy as np
import matplotlib.pyplot as plt

from mn import DataLoader
from nn import NeuralNetwork
from functions import CrossEntropy


def batch_generator(x, y, batch_size=64):
    n = x.shape[0]
    for i in np.arange(0, n, batch_size):
        begin, end = i, min(i + batch_size, n)
        yield x[begin:end].T, y[begin:end].T


def accuracy(y_real, y_predicted):
    return np.sum(y_predicted == y_real, axis=0) / len(y_real)


def fit(model, x_train, y_train, epochs=5):
    for epoch in range(epochs):
        loss, acc = [], []
        for x_batch, y_batch in batch_generator(x_train, y_train):
            o = model.forward(x_batch)

            loss.append(np.mean(CrossEntropy.loss(y_batch, o)))
            acc.append(accuracy(np.argmax(y_batch, axis=0), np.argmax(o, axis=0)))

            error = CrossEntropy.gradient(y_batch, o)
            model.backward(x_batch, error)

        yield epoch + 1, np.mean(loss), np.mean(acc)


def show_loss(train, val):
    x_ticks = list((range(1, len(val) + 1)))
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.plot(x_ticks, train, label='Train', marker='o')
    plt.plot(x_ticks, val, label='Validation', marker='o')
    plt.legend()
    plt.show()


def predict_digit(model, x):
    while True:
        try:
            index = min(int(input("Enter a number (0 - 9999): ")), 9999)

            img = x[index]
            plt.imshow(img.reshape(28, 28), cmap="Greys")

            img.shape += (1,)
            o = model.forward(img)

            digit = 'none' if np.argmax(o) == 10 else np.argmax(o)
            plt.title(f"It's a {digit}")
            plt.show()
        except:
            break


def main():
    data_loader = DataLoader()
    x_train, y_train = data_loader.get_training()
    x_test, y_test = data_loader.get_testing()

    model = NeuralNetwork()
    train_loss, val_loss = [], []

    for epoch, loss, acc in fit(model, x_train, y_train, epochs=3):
        o = model.forward(x_test.T)

        v_loss = np.mean(CrossEntropy.loss(y_test.T, o))
        v_acc = accuracy(np.argmax(y_test.T, axis=0), np.argmax(o, axis=0))

        train_loss.append(loss)
        val_loss.append(v_loss)

        print(f"Epoch {epoch}, Loss: {loss}, Acc: {acc}, Val Loss: {v_loss}, Val Acc: {v_acc}")

    show_loss(train_loss, val_loss)
    predict_digit(model, x_test)


if __name__ == "__main__":
    main()
