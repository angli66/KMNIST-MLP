################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
import data
from data import write_to_file
from neuralnet import *


def train(x_train, y_train, x_val, y_val, config, experiment=None):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        y_train: The train labels
        x_val: The validation set patterns
        y_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None

    # Initialize the model
    model = NeuralNetwork(config=config)

    # Indicator recording consecutive diverge epochs on validation set's loss
    diverge = 0

    # Training
    print("Training...")
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}...")

        running_train_loss = 0.0

        # SGD
        x_train, y_train = data.shuffle((x_train, y_train))
        batches = data.generate_minibatches((x_train, y_train), batch_size=config["batch_size"])
        for batch in batches:
            x, y = batch
            _, l1 = model(x, y)
            running_train_loss += l1

            # Backpropagation
            model.backward()

            # Update weights
            if config['momentum'] == False: # Normal SGD
                for layer in model.layers:
                    if type(layer) is Layer:
                        # Check if regularization is enabled
                        norm_w = np.zeros_like(layer.w)
                        norm_b = np.zeros_like(layer.b)
                        if config['regularization'] == True:
                            if config['L2_norm'] == True:
                                norm_w = layer.w
                                norm_b = layer.b
                            else:
                                norm_w = np.select([layer.w == 0, layer.w > 0, layer.w < 0], [0, 1, -1])
                                norm_b = np.select([layer.b == 0, layer.b > 0, layer.b < 0], [0, 1, -1])

                        layer.w += config["learning_rate"] * layer.d_w / len(x) - config['penalty'] * norm_w
                        layer.b += config["learning_rate"] * layer.d_b / len(x) - config['penalty'] * norm_b
            else: # SGD with momentum
                for layer in model.layers:
                    if type(layer) is Layer:
                        # Check if regularization is enabled
                        norm_w = np.zeros_like(layer.w)
                        norm_b = np.zeros_like(layer.b)
                        if config['regularization'] == True:
                            if config['L2_norm'] == True:
                                norm_w = layer.w
                                norm_b = layer.b
                            else:
                                norm_w = np.select([layer.w == 0, layer.w > 0, layer.w < 0], [0, 1, -1])
                                norm_b = np.select([layer.b == 0, layer.b > 0, layer.b < 0], [0, 1, -1])

                        if epoch == 0:
                            layer.v_w = config["learning_rate"] * layer.d_w / len(x)
                            layer.v_b = config["learning_rate"] * layer.d_b / len(x)
                            layer.w += layer.v_w - config['penalty'] * norm_w
                            layer.b += layer.v_b - config['penalty'] * norm_b
                        else:
                            layer.v_w = config["momentum_gamma"] * layer.v_w + config["learning_rate"] * layer.d_w / len(x)
                            layer.v_b = config["momentum_gamma"] * layer.v_b + config["learning_rate"] * layer.d_b / len(x)
                            layer.w += layer.v_w - config['penalty'] * norm_w
                            layer.b += layer.v_b - config['penalty'] * norm_b

        # Accuracy on train set
        p_train = data.one_hot_decoding(model(x_train))
        gt_train = data.one_hot_decoding(y_train)
        train_acc.append(np.sum((p_train == gt_train).astype(int)) / len(x_train))

        # Accuracy on validation set
        p_val = data.one_hot_decoding(model(x_val))
        gt_val = data.one_hot_decoding(y_val)
        val_acc.append(np.sum((p_val == gt_val).astype(int)) / len(x_val))

        # Loss on train set
        train_loss.append(running_train_loss / len(x_train))

        # Loss on validation set
        _, l2 = model(x_val, y_val)
        val_loss.append(l2 / len(x_val))

        print("Accuracy on train set: ",train_acc[-1])
        print("Accuracy on validation set: ",val_acc[-1])
        print("Loss on train set: ",train_loss[-1])
        print("Loss on validation set: ",val_loss[-1])

        # Early stop logic
        if config["early_stop"] == True:
            if epoch == 0:
                continue

            if val_loss[-1] >= val_loss[-2]:
                diverge += 1
            else:
                diverge = 0
            
            if diverge >= config["early_stop_epoch"]:
                break
    print("Finish training.")

    best_model = model

    return train_acc, val_acc, train_loss, val_loss, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    output, total_loss = model(x_test, y_test)

    loss = total_loss / len(x_test)

    p = data.one_hot_decoding(output)
    gt = data.one_hot_decoding(y_test)
    acc = np.sum((p == gt).astype(int)) / len(x_test)

    return loss, acc


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def check_gradients(x_train, y_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    if len(x_train) > 3:
        x_train = x_train[:3, :]
        y_train = y_train[:3]

    for i, (x, y) in enumerate(zip(x_train, y_train)):
        x, y = x.reshape(1, -1), y.reshape(1, -1)

        print(f"Pattern {i}")
        model = NeuralNetwork(config=config)
        _, _ = model(x, y)
        model.backward()
        d1 = model.layers[-1].d_b[0][0]
        d2 = model.layers[-3].d_b[0][0]
        d3 = model.layers[-1].d_w[0][0]
        d4 = model.layers[-1].d_w[0][1]
        d5 = model.layers[-3].d_w[0][0]
        d6 = model.layers[-3].d_w[0][1]
        
        eps = 1e-2

        print("Output bias weight")
        model.layers[-1].b[0][0] += eps
        _, E1 = model(x, y)
        model.layers[-1].b[0][0] -= 2*eps
        _, E2 = model(x, y)
        d1_approx = (E1 - E2) / (2*eps)
        model.layers[-1].b[0][0] += eps # Restore
        print(f"Gradient calculated by back prop: {-d1}")
        print(f"Gradient calculated by numerical approximation: {d1_approx}")
        print("")

        print("Hidden bias weight")
        model.layers[-3].b[0][0] += eps
        _, E1 = model(x, y)
        model.layers[-3].b[0][0] -= 2*eps
        _, E2 = model(x, y)
        d2_approx = (E1 - E2) / (2*eps)
        model.layers[-3].b[0][0] += eps # Restore
        print(f"Gradient calculated by back prop: {-d2}")
        print(f"Gradient calculated by numerical approximation: {d2_approx}")
        print("")

        print("Hidden to output weight 1")
        model.layers[-1].w[0][0] += eps
        _, E1 = model(x, y)
        model.layers[-1].w[0][0] -= 2*eps
        _, E2 = model(x, y)
        d3_approx = (E1 - E2) / (2*eps)
        model.layers[-1].w[0][0] += eps # Restore
        print(f"Gradient calculated by back prop: {-d3}")
        print(f"Gradient calculated by numerical approximation: {d3_approx}")
        print("")

        print("Hidden to output weight 2")
        model.layers[-1].w[0][1] += eps
        _, E1 = model(x, y)
        model.layers[-1].w[0][1] -= 2*eps
        _, E2 = model(x, y)
        d4_approx = (E1 - E2) / (2*eps)
        model.layers[-1].w[0][1] += eps # Restore
        print(f"Gradient calculated by back prop: {-d4}")
        print(f"Gradient calculated by numerical approximation: {d4_approx}")
        print("")

        print("Input to hidden weight 1")
        model.layers[-3].w[0][0] += eps
        _, E1 = model(x, y)
        model.layers[-3].w[0][0] -= 2*eps
        _, E2 = model(x, y)
        d5_approx = (E1 - E2) / (2*eps)
        model.layers[-3].w[0][0] += eps # Restore
        print(f"Gradient calculated by back prop: {-d5}")
        print(f"Gradient calculated by numerical approximation: {d5_approx}")
        print("")

        print("Input to hidden weight 2")
        model.layers[-3].w[0][1] += eps
        _, E1 = model(x, y)
        model.layers[-3].w[0][1] -= 2*eps
        _, E2 = model(x, y)
        d6_approx = (E1 - E2) / (2*eps)
        model.layers[-3].w[0][1] += eps # Restore
        print(f"Gradient calculated by back prop: {-d6}")
        print(f"Gradient calculated by numerical approximation: {d6_approx}")
        print("")
