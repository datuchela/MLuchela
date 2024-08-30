import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DEFAULT_PLOT = False
DEFAULT_SEED = None
DEFAULT_EPOCHS = 30 * 100
DEFAULT_LEARNING_RATE = 0.00001

parser = argparse.ArgumentParser("Polynomial Regression by datuchela")

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    metavar="Int",
    help=f"seed for random numbers (default: {DEFAULT_SEED})",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS,
    metavar="Int",
    help=f"number of epochs to train (default: {DEFAULT_EPOCHS})",
)
parser.add_argument(
    "-l",
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    metavar="Float",
    help=f"learning rate (default: {DEFAULT_LEARNING_RATE})",
)
parser.add_argument(
    "-p",
    "--plot",
    action="store_true",
    default=DEFAULT_PLOT,
    help=f"whether should render plot or not (default: {DEFAULT_PLOT})",
)

args = parser.parse_args()

X_TRAIN = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Y_TRAIN = np.array([9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64, 81])

X_TEST = np.array([10, 11, 12, 13, 14])
Y_TEST = np.array([100, 121, 144, 169, 196])

np.random.seed(args.seed)

WEIGHTS = np.array(
    [
        100 * np.random.normal(),
        100 * np.random.normal(),
        100 * np.random.normal(),
    ]
)


def predict(x, weights):
    result = 0
    for i in range(weights.size):
        result += (x**i) * weights[i]
    return result


if args.plot is True:
    _, axs = plt.subplots()
    fignum = plt.get_fignums()[0]
    plt.ion()


def train(weights, x_train, y_train, epochs, learning_rate, plot):
    weights_log = tqdm(bar_format="{desc}")
    mean_squared_error_log = tqdm(bar_format="{desc}")
    epochs_bar = tqdm(
        range(epochs),
        desc="EPOCHS",
        ncols=79,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{bar}]",
    )
    errors = []

    for epoch in range(epochs):
        gradients = np.zeros(weights.size)
        errors_sum = 0
        for n in range(x_train.size):
            error = predict(x_train[n], weights) - y_train[n]
            errors_sum += error
            for i in range(gradients.size):
                gradients[i] += error * x_train[n] ** i

        for i in range(weights.size):
            weights[i] -= learning_rate * gradients[i]

        # Logging
        weights_log.set_description_str(f"WEIGHTS: {weights}")
        mean_squared_error_log.set_description_str(
            f"MEAN SQUARED ERROR: {0.5*(errors_sum/x_train.size)**2}"
        )
        epochs_bar.update()

        if plot is True:
            errors.append(0.5 * ((errors_sum / x_train.size) ** 2))
            axs.cla()
            # axs.plot(x_train, predict(x_train, weights))

            axs.set_xlabel("Number of epochs")
            axs.set_ylabel("Mean squared error")
            axs.plot(errors)
            axs.set_xlim(0, epochs)
            axs.set_ylim(0, 2 * errors[epoch])
            plt.pause(0.01)
            plt.show()

            if not plt.fignum_exists(fignum):
                break
    return weights


def test(x_test, y_test, weights):
    sum_of_errors = 0
    for i in range(x_test.size):
        x = x_test[i]
        y = y_test[i]
        prediction = predict(x, weights)
        error = prediction - y
        sum_of_errors += error
        print("input:", x, "output:", prediction,
              "expected:", y, "error:", error)
    return 0.5 * (sum_of_errors / x_test.size) ** 2


def main():
    print(
        "===================== TRAINING ======================================"
    )
    print("SEED:", args.seed)
    print("LEARNING RATE:", args.learning_rate)
    print("INITIAL WEIGHTS:", WEIGHTS)
    weights = train(
        weights=np.copy(WEIGHTS),
        x_train=X_TRAIN,
        y_train=Y_TRAIN,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        plot=args.plot,
    )

    print(
        "\n===================== TRAINING DONE ==============================="
    )
    print("INITIAL WEIGHTS:", WEIGHTS)
    print("FINAL WEIGHTS:  ", weights)

    print(
        "\n===================== PREDICTIONS ================================="
    )
    mean_squared_error = test(X_TEST, Y_TEST, weights)
    print("MEAN SQUARED ERROR:", mean_squared_error)

    if args.plot is True:
        while plt.fignum_exists(fignum):
            plt.pause(1.0)


# Auto-execute
if __name__ == "__main__":
    main()
