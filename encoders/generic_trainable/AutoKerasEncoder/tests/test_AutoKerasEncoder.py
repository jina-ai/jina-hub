from .. import AutoKerasEncoder
import numpy as np
import os

# unit test helpers


def Mnist(train_method, num_trials=1, num_epochs=1):
    import pickle

    with open("mnist_min_data.pkl", "rb") as file:
        mnist_data = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = mnist_data

    model_type = "vision"  # may not be explicitly set and passed, the default model type is vision

    # training with just 2 trail and 1 epoch for test purposes only, it can be controlled by time as well
    encoder = AutoKerasEncoder(
        model_type=model_type, train_type=train_method, max_trials=num_trials
    )
    encoder.train((x_train, y_train), epochs=num_epochs)
    targetdims = encoder.output_shape
    print(
        "#########################################################################################################################"
    )
    print(
        "testing encoder for vision mode trained with "
        + train_method
        + " architecture search, for "
        + str(num_trials)
        + "trials and "
        + str(num_epochs)
        + " epochs each."
    )
    encoded_data = encoder.encode((x_test, y_test))
    print(
        "encoder evaluation [loss, accuracy]: " + str(encoder.get_training_inference())
    )
    print("encoded feature shape(e.g.): " + str(encoded_data[0].shape[0]))
    print("example feature(e.g.): " + str(encoded_data[0]))
    assert encoded_data[0].shape[0] == targetdims[1]


def IMDB(train_method, num_trials=1, num_epochs=1):
    import pickle

    with open("imdb_min_data.pkl", "rb") as file:
        imdb_data = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = imdb_data

    model_type = (
        "bert"  # needed to be explicitly specified as the default model_type is vision
    )

    # training with just 2 trail and 1 epoch for test purposes only, it can be controlled by time as well
    encoder = AutoKerasEncoder(
        model_type=model_type, train_type=train_method, max_trials=num_trials
    )
    encoder.train((x_train, y_train), epochs=num_epochs)
    targetdims = encoder.output_shape
    print(
        "#########################################################################################################################"
    )
    print(
        "testing encoder for bert mode trained with "
        + train_method
        + " architecture search, for "
        + str(num_trials)
        + "trials and "
        + str(num_epochs)
        + " epochs each."
    )
    encoded_data = encoder.encode((x_test, y_test))
    print(
        "encoder evaluation [loss, accuracy]: " + str(encoder.get_training_inference())
    )
    print("encoded feature shape(e.g.): " + str(encoded_data[0].shape[0]))
    print("encoded feature(e.g.): " + str(encoded_data[0]))
    assert encoded_data[0].shape[0] == targetdims[1]


def test_direct_encode_vision(train_method="classifier", num_trials=1, num_epochs=1):
    import pickle

    with open("mnist_min_data.pkl", "rb") as file:
        mnist_data = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = mnist_data

    model_type = "vision"  # may not be explicitly set and passed, the default model type is vision

    # training with just 2 trail and 1 epoch for test purposes only, it can be controlled by time as well
    encoder = AutoKerasEncoder(
        model_type=model_type, train_type=train_method, max_trials=num_trials
    )
    print(
        "#########################################################################################################################"
    )
    print(
        "testing encoder for vision mode trained inside encode directly via "
        + train_method
        + " architecture search, for "
        + str(num_trials)
        + "trials and "
        + str(num_epochs)
        + " epochs each."
    )
    encoded_data = encoder.encode((x_train, y_train))
    targetdims = encoder.output_shape
    print(
        "encoder evaluation [loss, accuracy]: " + str(encoder.get_training_inference())
    )
    print("encoded feature shape(e.g.): " + str(encoded_data[0].shape[0]))
    print("example feature(e.g.): " + str(encoded_data[0]))
    assert encoded_data[0].shape[0] == targetdims[1]


def test_direct_encode_bert(train_method="classifier", num_trials=1, num_epochs=1):
    import pickle

    with open("imdb_min_data.pkl", "rb") as file:
        imdb_data = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = imdb_data

    model_type = (
        "bert"  # needed to be explicitly specified as the default model_type is vision
    )

    # training with just 2 trail and 1 epoch for test purposes only, it can be controlled by time as well
    encoder = AutoKerasEncoder(
        model_type=model_type, train_type=train_method, max_trials=num_trials
    )
    print(
        "#########################################################################################################################"
    )
    print(
        "testing encoder for vision mode trained inside encode directly via "
        + train_method
        + " architecture search, for "
        + str(num_trials)
        + "trials and "
        + str(num_epochs)
        + " epochs each."
    )
    encoded_data = encoder.encode((x_train, y_train))
    targetdims = encoder.output_shape
    print(
        "encoder evaluation [loss, accuracy]: " + str(encoder.get_training_inference())
    )
    print("encoded feature shape(e.g.): " + str(encoded_data[0].shape[0]))
    print("example feature(e.g.): " + str(encoded_data[0]))
    assert encoded_data[0].shape[0] == targetdims[1]


# tests for encoder in vision mode
def test_classifier_based_vision_encoder_mnist():
    """
    Tests encoder in vision mode for classifier based training
    """
    # setting and passing num_trials and num_epochs only to reduce testing time at cost of accuracy, they are optional and can be ignored.
    num_trials = 1
    num_epochs = 2
    Mnist("classifier", num_trials, num_epochs)


def test_regressor_based_vision_encoder_mnist():
    """
    Tests encoder in vision mode for regressor based training
    """
    # setting and passing num_trials and num_epochs only to reduce testing time at cost of accuracy, they are optional and can be ignored.
    num_trials = 1
    num_epochs = 2
    Mnist("regressor", num_trials, num_epochs)


# tests for encoder in bert mode
def test_classifier_based_bert_encoder_imdb():
    """
    Tests encoder in bert mode for classifier based training
    """
    # setting and passing num_trials and num_epochs only to reduce time at cost of accuracy, they are optional and can be ignored.
    num_trials = 1
    num_epochs = 2
    IMDB("classifier", num_trials, num_epochs)


def test_regressor_based_bert_encoder_imdb():
    """
    Tests encoder in bert mode for regressor based training
    """
    # setting and passing num_trials and num_epochs only to reduce time at cost of accuracy, they are optional and can be ignored.
    num_trials = 1
    num_epochs = 2
    IMDB("regressor", num_trials, num_epochs)


def test_save_and_load_vision():
    from tensorflow.keras.datasets import mnist
    from jina.executors import BaseExecutor

    import pickle

    # (x_train, y_train), (x_test, y_test) = np.load('mnist_min_data.npy')
    with open("mnist_min_data.pkl", "rb") as file:
        mnist_data = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = mnist_data

    model_type = "vision"  # may not be explicitly set and passed, the default model type is vision
    # setting and passing num_trials and num_epochs only to reduce time at cost of accuracy, they are optional and can be ignored.
    num_trials = 1
    num_epochs = 2
    # training with just 1 trail and 2 epoch for test purposes only
    encoder = AutoKerasEncoder(model_type=model_type, max_trials=num_trials)
    encoder.train((x_train, y_train), epochs=num_epochs)
    targetdims = encoder.output_shape
    encoded_data_control = encoder.encode((x_test, y_test))
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode((x_test, y_test))
    print("encoded feature shape(e.g.): " + str(encoded_data_test[0].shape[0]))
    assert encoded_data_test[0].shape[0] == targetdims[1]
    assert encoder_loaded.channel_axis == encoder.channel_axis
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


def test_save_and_load_bert():
    from tensorflow.keras.datasets import mnist
    from jina.executors import BaseExecutor

    import pickle

    # (x_train, y_train), (x_test, y_test) = np.load('mnist_min_data.npy')
    with open("imdb_min_data.pkl", "rb") as file:
        imdb_data = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = imdb_data

    model_type = (
        "bert"  # needed to be explicitly specified as the default model_type is vision
    )

    # setting and passing num_trials and num_epochs only to reduce time at cost of accuracy, they are optional and can be ignored.
    num_trials = 1
    num_epochs = 2
    # training with just 1 trail and 2 epoch for test purposes only with default classifier training method
    encoder = AutoKerasEncoder(model_type=model_type, max_trials=num_trials)
    encoder.train((x_train, y_train), epochs=num_epochs)
    targetdims = encoder.output_shape
    encoded_data_control = encoder.encode((x_test, y_test))
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode((x_test, y_test))
    print("encoded feature shape(e.g.): " + str(encoded_data_test[0].shape[0]))
    assert encoded_data_test[0].shape[0] == targetdims[1]
    assert encoder_loaded.channel_axis == encoder.channel_axis
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
