
#%%
# setup
#base
import os
import numpy as np
import pandas as pd

#tensorflow
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Subtract, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility

# utils
import utils.file_handling as io
from utils import plotting as plt






#%%

training_path = os.getcwd()+'/data/training/'
train_data = io.get_tasks(training_path)

def preprocess_task(task):
    train_input_list = []
    train_output_list = []
    test_input_list = []
    test_output_list = []

    for task in train_data:
        for sample in task['train']:
            train_input_list.append(sample['input'])
            train_output_list.append(sample['output'])
        for sample in task['test']:
            test_input_list.append(sample['input'])
            test_output_list.append(sample['output'])
    return train_input_list, train_output_list, test_input_list, test_output_list


def enhance_mat_30x30(mat):
    empty_array = np.full((30, 30), 0, dtype=np.float32)
    if(len(mat) != 0):
        mat = np.asarray(mat, dtype=np.float32)
        empty_array[:mat.shape[0], : mat.shape[1]] = mat
        
    return np.expand_dims(empty_array, axis= 2) 


#%%

train_input = []
train_output = []
y_train_col_len  = []
y_train_row_len  = []
test_input_list = []
test_output_list = []

for task in train_data:
    for sample in task['train']:
        train_input.append(enhance_mat_30x30(sample['input']))
        train_output.append(enhance_mat_30x30(sample['output']))
        y_train_col_len.append(len(sample['output']))
        y_train_row_len.append(len(sample['output'][0]))
    for sample in task['test']:
        test_input_list.append(enhance_mat_30x30(sample['input']))
        test_output_list.append(enhance_mat_30x30(sample['output']))



#%%




#%%

input_X1 = Input(shape=(30, 30, 1))
x_1 = Conv2D(64, (3, 3), activation='relu')(input_X1)
x_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
x_1 = Dropout(0.25)(x_1)
x_1 = Flatten()(x_1)
x_1 = Dense(128, activation='relu')(x_1)
x_1 = Dropout(0.5)(x_1)

out_1 = Dense(128, activation='relu')(x_1)
out_1 = Dense(1, activation='linear', name='rows')(out_1)

out_2 = Dense(128, activation='relu')(x_1)
out_2 = Dense(1, activation='linear', name='cols')(out_2)

model = Model(inputs=[input_X1], outputs=[out_1, out_2])

opt = Adam(lr=1e-3, decay=1e-3)
losses = {
    "rows": "mean_absolute_error",
    "cols": "mean_absolute_error"}

model.compile(loss=losses, optimizer=opt)



#%%

y_hat = model.predict(np.array(train_input_list))
y_hat[0]


#%%

history = model.fit(
    [np.array(train_input)],
    [np.array(y_train_col_len), np.array(y_train_row_len)],
    epochs=100)

#%%
import matplotlib.pyplot as plt 

plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.title('training loss')
plt.show()

#%%
