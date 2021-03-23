import os
import numpy as np
import pickle

from umap import UMAP

target_output_dim = 2
batch_size = 10
input_dim = 28
model_path = 'pre-trained.model'

train_data = np.random.rand(batch_size, input_dim)

model = UMAP(n_components=target_output_dim, random_state=42)
model.fit(train_data)
pickle.dump(model, open(model_path, 'wb'))
