
#%%
import os 
import json 
from tensorflow.keras.models import load_model


class Logger:
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
        self.loss = loss
        return model


