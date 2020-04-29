
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

import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility

# utils
import utils.file_handling as io
from utils import plotting as plt # TODO rename shortcut



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
# load experiment 


log = logger('three_input_feature_extraction_model')
model = log.load_experiment()
model.summary()


#%%

from pandas.api.types import CategoricalDtype
cat_color = CategoricalDtype(categories=[i for i in range(10)]) 

flatten = lambda x: [item for row in x for item in row]
train_input = []
train_output = []

train_input_2 = []
train_output_2 = []
y_train_col_len  = []
y_train_row_len  = []
y_train_unique_colors_sum = []
y_train_unique_colors_cat = []
test_input_list = []
test_output_list = []

for task in train_data:
    for sample in task['train']:

        input_1 = sample['input']
        sample_2 = {'input':[]}
        while sample_2['input'] != input_1:
            sample_2 = random.choice(task['train'])
        
        train_input.append(enhance_mat_30x30(input_1))
        train_output.append(enhance_mat_30x30(sample['output']))
        
        train_input_2.append(enhance_mat_30x30(sample_2['input']))
        train_output_2.append(enhance_mat_30x30(sample_2['output']))

        unique_colors = pd.Series(list(set(flatten(sample['output'])))).astype(cat_color)
        y_train_col_len.append(len(sample['output']))
        y_train_row_len.append(len(sample['output'][0]))
        y_train_unique_colors_sum.append(len(set(flatten(sample['output']))))
        y_train_unique_colors_cat.append([i in unique_colors for i in range(10)])

    for sample in task['test']:
        test_input_list.append(enhance_mat_30x30(sample['input']))
        test_output_list.append(enhance_mat_30x30(sample['output']))


#%%

# input
input_X1 = Input(shape=(30, 30, 1))
input_X2 = Input(shape=(30, 30, 1))
output_X2 = Input(shape=(30, 30, 1))

# convolution layers
x_1 = Conv2D(64, (3, 3), activation='relu')(input_X1)
x_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
x_1 = Dropout(0.25)(x_1)
x_1 = Flatten()(x_1)

x_2 = Conv2D(64, (3, 3), activation='relu')(input_X2)
x_2 = MaxPooling2D(pool_size=(2, 2))(x_2)
x_2 = Dropout(0.25)(x_2)
x_2 = Flatten()(x_2)

x_2_out = Conv2D(64, (3, 3), activation='relu')(output_X2)
x_2_out = MaxPooling2D(pool_size=(2, 2))(x_2_out)
x_2_out = Dropout(0.25)(x_2_out)
x_2_out = Flatten()(x_2_out)

merge = concatenate([x_1, x_2, x_2_out])

merge = Dense(128, activation='relu')(merge)
merge = Dropout(0.5)(merge)

# regression layers
out_1 = Dense(128, activation='relu')(merge)
out_1 = Dense(1, activation='linear', name='rows')(out_1)

out_2 = Dense(128, activation='relu')(merge)
out_2 = Dense(1, activation='linear', name='cols')(out_2)

out_3 = Dense(128, activation='relu')(merge)
out_3 = Dense(1, activation='linear', name='unique_colors_sum')(out_3)

# multi-label classification layers
out_4 = Dense(128, activation='relu')(merge)
out_4 = Dense(10, activation='sigmoid', name='unique_colors_cat')(out_4)

model = Model(inputs=[input_X1, input_X2, output_X2], outputs=[out_1, out_2, out_3, out_4])

opt = Adam(lr=1e-3, decay=1e-3)
losses = {
    "rows": "mean_absolute_error",
    "cols": "mean_absolute_error",
    "unique_colors_sum": "mean_absolute_error",
    "unique_colors_cat": "binary_crossentropy"
    }

model.compile(loss=losses, optimizer=opt)



#%%

start = time.time()

history = model.fit(
    [
        np.array(train_input),
        np.array(train_input_2),
        np.array(train_output_2)],
    [
        np.array(y_train_col_len),
        np.array(y_train_row_len),
        np.array(y_train_unique_colors_sum),
        np.array(y_train_unique_colors_cat)
    ],
    epochs=500)

print('training time {} minutes'.format(round(time.time()-start)/60))


#%%

log = logger('three_input_feature_extraction_model')
log.save_experiment(model, history)


#%%

plt.plot_loss(log, 'rows_loss', save=False)


#%%


#%%

y_hat = model.predict([
        np.array(train_input),
        np.array(train_input_2),
        np.array(train_output_2)])
y_hat[0][0]


#%%

y_hat[0][1]

#%%

y_train_row_len[1]

# %%
# plot input output

plt.plot_matrix(train_data[0]['train'][1]['input'])
plt.plot_matrix(train_data[0]['train'][1]['output'])

