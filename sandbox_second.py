
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
from tensorflow.keras.estimator import model_to_estimator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility

# utils
import utils.file_handling as io
from utils.logging import Logger
from utils import plotting as plt # TODO rename shortcut
from utils import pretrain_task as pretrain
from utils import pretrain_data_generator as pretrain_generator

#%%

training_path = os.getcwd()+'/data/training/'
train_data = io.get_tasks(training_path)


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
    #pretrain.double_line_with_multiple_colors_tasks,
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

len(train_input)
len(train_output)
len(test_input)


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
out_1 = Dense(1, activation='linear', name='fzn_rows')(out_1)

out_2 = Dense(128, activation='relu')(merge)
out_2 = Dense(1, activation='linear', name='fzn_cols')(out_2)

out_9 = Dense(128, activation='relu')(merge)
out_9 = Dense(1, activation='linear', name='fzn_removed_line_nr')(out_9)

out_11 = Dense(128, activation='relu')(merge)
out_11 = Dense(1, activation='linear', name='fzn_shifted_line_nr')(out_11)

# out_15 = Dense(128, activation='relu')(merge)
# out_15 = Dense(1, activation='linear', name='doubled_line_nr')(out_15)

# multi-label classification layers
out_4 = Dense(128, activation='relu')(merge)
out_4 = Dense(4, activation='sigmoid', name='fzn_rotation_angle')(out_4)

out_5 = Dense(128, activation='relu')(merge)
out_5 = Dense(3, activation='sigmoid', name='fzn_multiply_factor')(out_5)

out_6 = Dense(128, activation='relu')(merge)
out_6 = Dense(10, activation='sigmoid', name='fzn_changed_color_old')(out_6)

out_7 = Dense(128, activation='relu')(merge)
out_7 = Dense(10, activation='sigmoid', name='fzn_changed_color_new')(out_7)

out_8 = Dense(128, activation='relu')(merge)
out_8 = Dense(3, activation='sigmoid', name='fzn_removed_row_or_column')(out_8)

out_10 = Dense(128, activation='relu')(merge)
out_10 = Dense(3, activation='sigmoid', name='fzn_shifted_row_or_column')(out_10)

out_12 = Dense(128, activation='relu')(merge)
out_12 = Dense(3, activation='sigmoid', name='fzn_multiply_rotation_factor')(out_12)

out_13 = Dense(128, activation='relu')(merge)
out_13 = Dense(3, activation='sigmoid', name='fzn_multiply_mirror_factor')(out_13)

# out_14 = Dense(128, activation='relu')(merge)
# out_14 = Dense(3, activation='sigmoid', name='doubled_row_or_column')(out_14)


model = Model(inputs=[input_, output_], outputs=[
    out_1, out_2, out_4,
    out_5, out_6, out_7,
    out_8, out_9, out_10,
    out_11, out_12, out_13,
    # out_14, out_15
    ])

opt = Adam(lr=1e-3, decay=1e-3)
losses = {
    "fzn_rows": "mean_absolute_error",
    "fzn_cols": "mean_absolute_error",
    "fzn_removed_line_nr": "mean_absolute_error",
    "fzn_shifted_line_nr": "mean_absolute_error",
    "fzn_rotation_angle": "binary_crossentropy",
    "fzn_multiply_factor": "binary_crossentropy",
    "fzn_changed_color_old": "binary_crossentropy",
    "fzn_changed_color_new": "binary_crossentropy",
    "fzn_removed_row_or_column": "binary_crossentropy",
    "fzn_shifted_row_or_column": "binary_crossentropy",
    "fzn_multiply_rotation_factor": "binary_crossentropy",
    "fzn_multiply_mirror_factor": "binary_crossentropy",
    # "doubled_row_or_column": "binary_crossentropy",
    # "doubled_line_nr": "mean_absolute_error"
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
        np.array(to_categorical(y_labels[8])),
        np.array(to_categorical(y_labels[9])),
    ],
    epochs=100)

print('training time {} minutes'.format(round(time.time()-start)/60))

#%%

log = Logger('2_input_12_pretrain')
log.save_experiment(model, history)

#%%

plt.plot_loss(log, 'loss', save=True)

#%%

log = Logger('2_input_12_pretrain')
model = log.load_experiment()
model.summary()

#%%
"""
path_frozen = 'data/weights_frozen/{}'.format(log.name)
if not os.path.exists(path_frozen):
    os.mkdir(path_frozen)
model.save(path_frozen)
"""

#%%

for layer in model.layers[:-12]:
    layer.trainable = False 
    #model.layers.pop()
model.summary()


#%%
# predict whole task model

input_test_task = Input(shape=(30, 30, 1), name='test_input')
input_frozen_model = concatenate(model.output, name='concat_frozen_layers')

# convolution layers
x_test = Conv2D(64, (3, 3), activation='relu', name='test_convolution')(input_test_task)
x_test = MaxPooling2D(pool_size=(2, 2), name='test_pooling')(x_test)
x_test = Dropout(0.25, name='test_dropout')(x_test)
x_test = Flatten(name='test_flatten')(x_test)

# merge frozen layers
merge_frozen = concatenate([
    x_test,
    input_frozen_model
     ], name='concat_test_frozen') 

# out layers
out_final = Dense(128, activation='relu', name='test_out_dense_1')(merge_frozen)
out_final = Dense(128, activation='relu', name='test_out_dense_2')(out_final)

out_final_list = []
for i in range(900):
        out_final_list.append(Dense(10, activation='softmax', name=str('ouput_' + str(i)))(out_final))

new_model = Model(
    inputs=[
        input_test_task,
        model.input], 
    outputs=[
        out_final_list
    ])

opt = Adam(lr=1e-3, decay=1e-3)

new_model.compile(loss="binary_crossentropy", optimizer=opt)

new_model.summary()

#%%
# final task generation

train_input = []
train_output = []
test_input = []
test_output = []

cnt = 0
# use all the tasks in train
# use input and output

for task in train_data:
    for sample in task['train']:
        _input = sample['input']
        _output = sample['output']
                
        if (len(_input) > 1) & (len(_output) > 1):
            train_input.append(enhance_mat_30x30(_input))
            train_output.append(enhance_mat_30x30(_output))
            test_input.append(enhance_mat_30x30(task['test'][0]['input']))
            test_output.append(enhance_mat_30x30(task['test'][0]['output']))
        else:
            cnt += 1

print('Thrown away samples: ')
print(cnt)

print('Total train input samples: ')
print(len(train_input))

print('Total train input samples: ')
print(len(train_output))

print('Total test samples: ')
print(len(test_input))


#%%


test_output_processed = []
for i in range(900):
    test_output_processed.append([])

for task in test_output:
    all_pixel = to_categorical(task.flatten(), num_classes=10)
    for i in range(900):
        test_output_processed[i].append(all_pixel[i])


#%%

plt.plot_matrix(test_input[1].reshape(30, 30))
plt.plot_matrix(train_input[1].reshape(30, 30))
plt.plot_matrix(train_output[1].reshape(30, 30))
plt.plot_matrix(test_output[1].reshape(30, 30))


#%%

plt.plot_matrix(test_input[7].reshape(30, 30))
plt.plot_matrix(train_input[7].reshape(30, 30))
plt.plot_matrix(train_output[7].reshape(30, 30))
plt.plot_matrix(test_output[7].reshape(30, 30))

#%%
# train final model

history = new_model.fit([
    np.array(test_input),
    np.array(train_input),
    np.array(train_output)
    ],
    test_output_processed,
    epochs=10,
    verbose=1
    )


# %%

history = new_model.history

log = Logger('final_model_0.0')
log.save_experiment(new_model, history)


#%%

plt.plot_loss(log, 'loss', save=True)


#%%

y_hat = new_model.predict([
    np.array(test_input),
    np.array(train_input),
    np.array(train_output)
    ])

#%%
# post processing 

y_hat_processed = []

for i in range(len(y_hat[0])):
    y_hat_processed.append([])
    for pixel in y_hat:
        y_hat_processed[i].append(np.argmax(pixel[i]))
