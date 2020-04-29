
#%%
# setup
#base
import os
import numpy as np
import pandas as pd

import json
import time 

import random

#tensorflow

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Subtract, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility

# utils
import utils.file_handling as io
from utils import plotting as plt # TODO rename shortcut
from utils import pretrain_task as pretrain
from utils import pretrain_data_generator as pretrain_generator

#%%

training_path = os.getcwd()+'/data/training/'
train_data = io.get_tasks(training_path)

class logger:
    def __init__(self, ID):
        self.name = ID
        self.loss = {}
        self.model = None
        self.model_path = 'data/weights/{}.h5'.format(ID)
        self.logs_path = 'data/logs/{}.json'.format(ID)

    def save_experiment(self, model, history):
        self.model = model 
        
        # save NN
        model.save(self.model_path)

        # save loss
        loss = history.history
        for l in loss:
            loss[l] = [float(num) for num in loss[l]]
        self.loss = loss
        print(os.getcwd())
        with open(self.logs_path, 'w') as json_file:
            json.dump(self.loss, json_file)

    def load_experiment(self):
        # load model
        model = load_model(self.model_path)

        # load loss
        with open(self.logs_path, 'r') as json_file:
            loss = json.load(json_file)
        log.loss = loss
        return model


def enhance_mat_30x30(mat):
    empty_array = np.full((30, 30), 0, dtype=np.float32)
    if(len(mat) != 0):
        mat = np.asarray(mat, dtype=np.float32)
        empty_array[:mat.shape[0], : mat.shape[1]] = mat
        
    return np.expand_dims(empty_array, axis= 2) 

#%%
train_input = []

cnt = 0
# use all the tasks in train
# use input and output

for task in train_data:
    for sample in task['train']:
        _input = sample['input']
        _output = sample['output']
                
        if len(_input) > 1:
            train_input.append(_input)
        else:
            cnt += 1
        
        if len(_output) > 1:
            train_input.append(_output)
        else:
            cnt += 1

print('Thrown away samples: ')
print(cnt)

print('Total pretrain samples: ')
print(len(train_input))

#%%

# generate all the pretrain data
# 1. modified output tasks
# 2. y_labels

PRETRAIN_FUNCTIONS = [
    pretrain.rotate_tasks,
    pretrain.multiply_tasks,
    pretrain.change_random_color_tasks,
    pretrain.remove_line_tasks,
    pretrain.shift_line_tasks,
    pretrain.multiply_rotation_tasks,
    pretrain.mirror_tasks,
    pretrain.double_line_with_multiple_colors_tasks,
]

train_output, y_labels, y_train_row_len, y_train_col_len = pretrain_generator.generate_pretrain_data(PRETRAIN_FUNCTIONS, train_input)
_train_input = [enhance_mat_30x30(task) for task in train_input]

train_input = []

for i in range(len(PRETRAIN_FUNCTIONS)):
    train_input = train_input + _train_input

train_input = [np.array(el) for el in train_input]
train_output = [np.array(el) for el in train_output]

print("LEN train_input")
print(len(train_input))

print("LEN train_output")
print(len(train_output))

print("LEN y_labels")
print(len(y_labels[0]))

print("LEN y_train_row_len")
print(len(y_train_row_len))

print("LEN y_train_col_len")
print(len(y_train_col_len))


#%%
#%%
# Raphael Model
# input
input_ = Input(shape=(30, 30, 1), name='train_input')
output_ = Input(shape=(30, 30, 1), name='train_output')

# convolution layers
x_1 = Conv2D(64, (3, 3), activation='relu')(input_)
x_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
x_1 = Dropout(0.25)(x_1)
x_1 = Flatten()(x_1)

x_2 = Conv2D(64, (3, 3), activation='relu')(output_)
x_2 = MaxPooling2D(pool_size=(2, 2))(x_2)
x_2 = Dropout(0.25)(x_2)
x_2 = Flatten()(x_2)

merge = concatenate([x_1, x_2])

merge = Dense(128, activation='relu')(merge)
merge = Dense(128, activation='relu')(merge)
merge = Dropout(0.3)(merge)

pretrain.mirror_tasks,
pretrain.double_line_with_multiple_colors_tasks,

# regression layers
out_1 = Dense(128, activation='relu')(merge)
out_1 = Dense(1, activation='linear', name='rows')(out_1)

out_2 = Dense(128, activation='relu')(merge)
out_2 = Dense(1, activation='linear', name='cols')(out_2)

out_9 = Dense(128, activation='relu')(merge)
out_9 = Dense(1, activation='linear', name='removed_line_nr')(out_9)

out_11 = Dense(128, activation='relu')(merge)
out_11 = Dense(1, activation='linear', name='shifted_line_nr')(out_11)

out_14 = Dense(128, activation='relu')(merge)
out_14 = Dense(1, activation='linear', name='doubled_line_nr')(out_14)

# multi-label classification layers
out_4 = Dense(128, activation='relu')(merge)
out_4 = Dense(4, activation='sigmoid', name='rotation_angle')(out_4)

out_5 = Dense(128, activation='relu')(merge)
out_5 = Dense(3, activation='sigmoid', name='multiply_factor')(out_5)

out_6 = Dense(128, activation='relu')(merge)
out_6 = Dense(10, activation='sigmoid', name='changed_color_old')(out_6)

out_7 = Dense(128, activation='relu')(merge)
out_7 = Dense(10, activation='sigmoid', name='changed_color_new')(out_7)

out_8 = Dense(128, activation='relu')(merge)
out_8 = Dense(3, activation='sigmoid', name='removed_row_or_column')(out_8)

out_10 = Dense(128, activation='relu')(merge)
out_10 = Dense(3, activation='sigmoid', name='shifted_row_or_column')(out_10)

out_11 = Dense(128, activation='relu')(merge)
out_11 = Dense(3, activation='sigmoid', name='multiply_rotation_factor')(out_11)

out_12 = Dense(128, activation='relu')(merge)
out_12 = Dense(3, activation='sigmoid', name='multiply_mirror_factor')(out_12)

out_13 = Dense(128, activation='relu')(merge)
out_13 = Dense(3, activation='sigmoid', name='doubled_row_or_column')(out_13)


model = Model(inputs=[input_, output_], outputs=[
    out_1, out_2, out_4,
    out_5, out_6, out_7,
    out_8, out_9, out_10,
    out_11, out_12, out_13,
    out_14
    ])

opt = Adam(lr=1e-3, decay=1e-3)
losses = {
    "rows": "mean_absolute_error",
    "cols": "mean_absolute_error",
    "removed_line_nr": "mean_absolute_error",
    "shifted_line_nr": "mean_absolute_error",
    "rotation_angle": "binary_crossentropy",
    "multiply_factor": "binary_crossentropy",
    "changed_color_old": "binary_crossentropy",
    "changed_color_new": "binary_crossentropy",
    "removed_line_or_column": "binary_crossentropy",
    "shifted_line_or_column": "binary_crossentropy",
    "multiply_rotation_factor": "binary_crossentropy",
    "multiply_mirror_factor": "binary_crossentropy",
    "doubled_row_or_column": "binary_crossentropy",
    "doubled_line_nr": "mean_absolute_error"
    }

model.compile(loss=losses, optimizer=opt)

#%%

#%%
# Train the model

start = time.time()

history = model.fit(
    [
        np.array(train_input),
        np.array(train_output)
    ],
    [
        np.array(y_train_col_len),
        np.array(y_train_row_len),
        np.array(to_categorical(y_labels[0])),
        np.array(to_categorical(y_labels[1])),
        np.array(to_categorical(y_labels[2])),
        np.array(to_categorical(y_labels[3])),
        np.array(to_categorical(y_labels[4])),
        np.array(y_labels[5]),
        np.array(to_categorical(y_labels[6])),
        np.array(y_labels[7]),
    ],
    epochs=5)

print('training time {} minutes'.format(round(time.time()-start)/60))

#%%
log = logger('2_input_5_pretrain')
log.save_experiment(model, history)

#%%
plt.plot_loss(log, 'loss', save=False)