
import json 
import os


def get_tasks(path):
    training_tasks = os.listdir(path)

    train_task_list = []
    for task_name in training_tasks:
        task_file = str(path + task_name)

        with open(task_file, 'r') as f:
            task = json.load(f)
            if len(task['train']) < 6:
                train_task_list.append(task)
                
    return train_task_list





