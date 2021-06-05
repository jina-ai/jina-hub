__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTFEncoder
from jina.excepts import PretrainedModelFileDoesNotExist
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union


class AutoKerasEncoder(BaseTFEncoder):
    """
    :class:`AutoKerasEncoder` trains and encodes the documents with custom encoder architectures suiting the dataset, using neural architectural search via AutoKeras.

    > data supported - image/text

    > Data Format: tuple of numpy.ndarray or tf.data.Dataset. The two elements are,

        1. input data - x
                for vision (image) 	: The shape of the data should be should be (samples, width, height) or (samples, width, height, channels).
                for bert (text) 	: The data should be one dimensional. Each element in the data should be a string which is a full sentence.
        2. output data - y (labels)
                for classification based training 	: It can be raw labels, one-hot encoded if more than two classes, or binary encoded for binary classification. The raw labels will be encoded to one column if two classes were found, or one-hot encoded if more than two classes were found.
                for regression based training 		: It can be single-column or multi-column. The values should all be numerical.

    > Training types supported - classification/regression [default: classification]

    > model architectures checked and tuned

    'vision' : ResNet(variants), Xception(variants), conv2d
    'bert' : Vanilla, Transformer, ngram

    """

    def __init__(
        self,
        model_type: Optional[
            str
        ] = "vision",  #TODO infer data type automatically using metas if possible (investigate)
        train_type: Optional[str] = "classifier",
        model_save_path: Optional[str] = "autokeras-encoder-tf",
        arch_save_path: Optional[str] = "AutoKerasEncoder-NeuralArchitecture.png",
        multi_label: bool = False,
        # loss: types.LossType = None,                         # loss can be passed only if autokeras utils imported globally
        # metrics: Optional[types.MetricsType] = None,         # metrics can be passed only if autokeras utils imported globally
        project_name: str = "image_classifier",
        max_trials: int = 50,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        # tuner: Union[str, Type[tuner.AutoTuner]] = None,     # keras tuner object can be passed only if autokeras utils imported globally
        overwrite: bool = False,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        channel_axis: int = 1,
        *args,
        **kwargs
    ):

        """
        > Arguments

        :param model_type: Optional[str]: String. Defaults to 'vision'. [accepted values : 'vision' or 'bert']
        :param train_type: Optional[str]: String. Defaults to 'classifier'. [accepted values : 'classifier' or 'regressor']
        :param model_save_path: Optional[str]: String. Defaults to 'autokeras-encoder-tf'. [accepted values : 'vision' or 'bert']
        :param arch_save_path: Optional[str]: String. Defaults to 'AutoKerasEncoder-NeuralArchitecture.png'. [accepted values : 'classifier' or 'regressor']

        > Autokeras based Arguments(supported) can be passed through here (ref: https://autokeras.com/image_classifier/)

        :param num_classes: Optional[int]: Int. Defaults to None. If None, it will be inferred from the data.
        :param multi_label: bool: Boolean. Defaults to False.
        :param project_name: str: String. The name of the AutoModel. Defaults to 'image_classifier'.
        :param max_trials: int: Int. The maximum number of different Keras Models to try. The search may finish before reaching the max_trials. Defaults to 100.
        :param directory: Optional[Union[str, pathlib.Path]]: String. The path to a directory for storing the search outputs. Defaults to None, which would create a folder with the name of the AutoModel in the current directory.
        :param objective: str: String. Name of model metric to minimize or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        :param overwrite: bool: Boolean. Defaults to False. If False, reloads an existing project of the same name if one is found. Otherwise, overwrites the project.
        :param seed: Optional[int]: Int. Random seed.
        :param max_model_size: Optional[int]: Int. Maximum number of scalars in the parameters of a model. Models larger than this are rejected.

        """
        super().__init__(*args, **kwargs)

        # The below parameters can be avoided by setting the model here directly - peformance fix for later : needs ak and tf available in init scope

        self.model_type = model_type
        self.train_type = train_type
        self.max_trials = max_trials
        self.directory = directory
        self.project_name = project_name
        self.objective = objective
        # self.tuner=tuner
        self.multi_label = (multi_label,)
        # self.loss=loss,
        # self.metrics=metrics,
        self.overwrite = overwrite
        self.seed = seed
        self.max_model_size = max_model_size
        self.channel_axis = channel_axis
        self.trained = False
        self.training_eval = (0, 0)
        self.model_save_path = model_save_path
        self.arch_save_path = arch_save_path
        self.output_shape = None

    def post_init(self):
        self.to_device()
        import tensorflow as tf
        import autokeras as ak

        # build symbolic hypermodel 
	#TODO - strcutured data and audio data
        if self.model_type == "vision":
            input_node = ak.ImageInput()
            output_node = ak.ImageBlock()(input_node)
        elif self.model_type == "bert":
            input_node = ak.TextInput()
            output_node = ak.TextBlock()(input_node)
        else:
            print(
                "unsupported model_type passed to init [appcepted values : 'vision' or 'bert']"
            )

        output_node = ak.DenseBlock()(output_node)

        if self.train_type == "classifier":
            output_node = ak.ClassificationHead()(output_node)
        elif self.train_type == "regressor":
            output_node = ak.RegressionHead()(output_node)
        else:
            print(
                "unsupported model_type passed to init [appcepted values : 'classifier' or 'regressor']"
            )

        self.hypermodel = ak.AutoModel(
            inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
        )

    def train(
        self,
        content: "np.ndarray",
        test_size=None,
        epochs=None,
        model_save_path="autokeras-encoder-tf",
        arch_save_path="AutoKerasEncoder-NeuralArchitecture.png",
        *args,
        **kwargs
    ):
        """
        :param content: The content data should be a tuple of

        1. (input data) numpy.ndarray or tf.data.Dataset. The shape of the data should be should be (samples, width, height) or (samples, width, height, channels).
        2. (output data) tf.data.Dataset, np.ndarray, pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two classes, or binary encoded for binary classification
        The raw labels will be encoded to one column if two classes were found, or one-hot encoded if more than two classes were found.

        :param test_size: train test split ratio
        :param epochs: training epochs, tuned automatically if not provided
        """

        import tensorflow as tf
        import autokeras as ak
        from sklearn.model_selection import train_test_split

        x, y = content
        if test_size is None:
            test_size = 0.33
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )
        # Reshape the images to have the channel dimension.
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        # TODO - tf.data pipelining with cache, prefetch etc
        # train_set = tf.data.Dataset.from_tensor_slices(((x_train,), (y_train,)))
        # test_set = tf.data.Dataset.from_tensor_slices(((x_test,), (y_test,)))

        self.hypermodel.fit(
            x_train, y_train, validation_data=(x_test, y_test), epochs=epochs
        )  # save model if nessecary to port with no coupling or find checkpoints in auto_model folder
        model = self.hypermodel.export_model()
        self.training_eval = self.hypermodel.evaluate(x_test, y_test)
        self.trained = True
        # model.summary()
        encoder_model = tf.keras.Model(model.input, model.get_layer("dense_1").output)
        self.output_shape = encoder_model.layers[-1].output_shape
        print("best output dimension found: " + str(self.output_shape))
        tf.keras.utils.plot_model(
            encoder_model,
            show_shapes=True,
            expand_nested=False,
            to_file=self.arch_save_path,
        )
        print("best neural architecture saved at: " + self.arch_save_path)
        encoder_model.save(self.model_save_path, save_format="tf")

    def get_training_inference(self):
        return self.training_eval

    def get_output_dims(self):
        return self.output_shape

    @as_ndarray
    def encode(
        self, content: "np.ndarray", test_size=None, *args, **kwargs
    ) -> "np.ndarray":
        """
        :param content: The content data should be a tuple of

        1. (input data) numpy.ndarray or tf.data.Dataset. The shape of the data should be should be (samples, width, height) or (samples, width, height, channels).
        2. (output data) tf.data.Dataset, np.ndarray, pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two classes, or binary encoded for binary classification
        The raw labels will be encoded to one column if two classes were found, or one-hot encoded if more than two classes were found.

        :return: a `B x D` numpy ``ndarray``, `D` is the output dimension
        """
        import tensorflow as tf

        if not self.trained or self.model_save_path is "":
            print(
                "model not trained, training with available data [you can manually train on full data before calling 'encode' by 'calling train']"
            )
            train(content)
        (x, y) = content

        if self.model_save_path and os.path.exists(self.model_save_path):
            encoder_model = tf.keras.models.load_model(self.model_save_path)

        return encoder_model.predict(x)
