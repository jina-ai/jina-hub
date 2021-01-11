import os
import numpy as np

from cvae import cvae

target_output_dim = 2
batch_size = 10
input_dim = 28
model_path = 'model'
data_path = 'data'
os.mkdir(model_path)
os.mkdir(data_path)

test_data = np.random.rand(batch_size, input_dim)

for idx, features in enumerate(test_data):
    np.save(os.path.join(data_path, str(idx)), features)

# Train the CVAE on the test data to build a model saved in `logdir`.
model = cvae.CompressionVAE(data_path,
                            dim_latent=target_output_dim,
                            logdir=model_path,
                            feature_normalization=False)
model.train()
