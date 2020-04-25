
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
    [np.array(y_train_col_len), np.array(y_train_row_len)])


#%%
# MULTI-Input-Output-CNN

def create_model():
    input_X1 = Input(shape=(None, 1))
    input_X2 = Input(shape=(None, 1))
    input_Y1 = Input(shape=(None, 1))
    input_Y2 = Input(shape=(1024, 1))
    x_1 = Conv1D(filters=32, kernel_size=(4), strides=1, padding='same')(input_X1)
    x_2 = Conv1D(filters=32, kernel_size=(4), strides=1, padding='same')(input_X2)
    x_sub = Subtract()([x_1, x_2])
    x_fin = Conv1D(32, kernel_size=(4), strides=1, padding='same', name='training_task_final_layer')(x_sub)
    
    y_1 = Conv1D(filters=32, kernel_size=(4), strides=1, padding='same')(input_Y1)
    y_1_pooling = MaxPooling1D(4)(y_1)
    #y_1_flatten = Flatten()(y_1_pooling)
    y_2 = Conv1D(filters=32, kernel_size=(4), strides=1, padding='same')(input_Y2)
    y_2_pooling = MaxPooling1D(4)(y_2)
    #y_2_flatten = Flatten()(y_2_pooling)
    y_con = concatenate([y_1_pooling, y_2_pooling])
    y_fin = Conv1D(32, kernel_size=(4), strides=1, padding='same', name='test_task_final_layer')(y_con)
    
    merge = concatenate([x_fin, y_fin])
    #flat_layer = Flatten()(merge)

    
    out_1 = Dense(11, activation='relu')(merge)
    out_1 = Dense(128, activation='relu')(out_1)
    out_1 = Dense(256, activation='relu')(out_1)
    out_1 = Dense(512, activation='relu')(out_1)
    out_1 = Dense(1, activation='softmax', name='pixel_predictor')(out_1)

    out_2 = Dense(64, activation='relu')(merge)
    out_2 = Dense(1, activation='linear', name='shape_predictor')(out_2)
    
    model = Model(inputs=[input_X1, input_X2, input_Y1, input_Y2], outputs=[out_1, out_2])
    
    opt = Adam(lr=1e-3, decay=1e-3)
    losses = {
        "pixel_predictor": "categorical_crossentropy",
        "shape_predictor": "mean_absolute_error",
    }

    model.compile(loss=losses, optimizer=opt)
    return model

#multi_cnn = create_model()


#%%

output_pixel_all_iterations = []
for task in train_data:
    full_output = np.array(task['test'][0]['output']).flatten()
    all_iters_task = []
    for i in range(len(full_output)): # number of iterations = number of pixels in test ouput
        pixel = full_output[i]
        pixel_dummies = np.array([pixel == i for i in range(0, 11)])
        all_iters_task.append(pixel_dummies)
    output_pixel_all_iterations.append(all_iters_task)
output_pixel_all_iterations[0][0]

#%%
