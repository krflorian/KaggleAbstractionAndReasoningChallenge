from plotting import plot_matrix
import random

ANGLE_LIST = [
    1, # rotate 90 degrees
    2, # rotate 180 degrees
    3, # rotate 270 degrees
]

# Takes task_list and rotates each task randomly
# Returns list of rotated tasks + list of lables about rotation
def rotate_tasks(task_list):
    rotated_task_list = []
    y_rotation = []
    
    for task in task_list:
        angle = random.choice(ANGLE_LIST)
        y_rotation.append(angle)
        rotated_task_list.append(rotate_task(task=task, angle=angle))
        
    return rotated_task_list, y_rotation

# rotates one task according to spec (clockwise all the time)
# returns rotated task
def rotate_task(task, angle):
    for i in range(angle):
        task = list(zip(*reversed(task)))
    # convert list of tuples to list of lists
    return [list(elem) for elem in task]
    
if __name__ == "__main__":
    test_task_1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    test_task_2 = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    
    test_task_list = [test_task_1, test_task_2]

    rotated_test_task_list, y_rotation = rotate_tasks(test_task_list)

    #plot_matrix(test_task_list[0])
    #plot_matrix(rotated_test_task_list[0])

    print(test_task_list)
    print(rotated_test_task_list)
    print(y_rotation)
    

    
