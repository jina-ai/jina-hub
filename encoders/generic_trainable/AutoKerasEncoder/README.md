# AutoKerasEncoder

**`AutoKerasEncoder`** trains and encodes the documents with custom(best) encoder architecture suiting the dataset, using neural architectural search via AutoKeras.
    
> data supported - image/text
    
> Data Format: tuple of numpy.ndarray or tf.data.Dataset. The two elements are,
    
    1. input data - x 
	for vision (image) : The shape of the data should be should be (samples, width, height) or (samples, width, height, channels).
        for bert (text) : The data should be one dimensional. Each element in the data should be a string which is a full sentence.
    2. output data - y (labels)
	for classification based training : It can be raw labels, one-hot encoded if more than two classes, or binary encoded for binary classification. The raw labels will be encoded to one column if two classes were found, or one-hot encoded if more than two classes were found.
	for regression based training : It can be single-column or multi-column. The values should all be numerical.
    
> training types supported - classification/regression
    
> model architectures checked and tuned 
    
    'vision' : ResNet(variants), Xception(variants), conv2d
    'bert' : Vanilla, Transformer, ngram


## Docker build

for vision: docker build -f Dockerfile_vision .

for bert: docker build -f Dockerfile_bert .

# yaml configs

for vision: config_vision.yml 			
for bert: config_bert.yml 

Tip: add arguments/AutoKeras arguments to 'with' directive for changing training types, architecture search control etc. not mentioning any arguments will initialize a classification based vision encoder with default settings.

# standalone encoder usage

encoder = AutoKerasEncoder(model_type='vision')				# init
encoder.train((x_train, y_train))					# architecture search and train
encoder.encode((x_catalog, y_catalog))                         		# encode

or,

encoder = AutoKerasEncoder(model_type='vision')				# init
encoder.encode((x_full, y_full))					# architecture search, train and encode

