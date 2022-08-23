import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import json


def get_best_acc(file_name):
    '''
    Reads in losses / accuracies from files output by seg_epi.py or seg_UNet.py

    Parameters
    ----------
    file_name : str

    Returns
    -------
    t_loss_acc : list
        Training losses / accuracies formatted as a np array.
    v_loss_acc : list
        Validation losses / accuracies formatted as a np array.
    '''

    acc_file = open(file_name, "r")

    file_lines = acc_file.readlines()

    pos = [file_lines[i].find(":") for i in range(0,len(file_lines))]
    start = [pos[i] + 2 for i in range(0, len(pos))]
    end = [pos[i] + 8 for i in range(0, len(pos))]     
    t_loss_acc = [file_lines[i][start[i]:end[i]].strip(" ,\n") for i in range(0,len(file_lines))]

    pos = [file_lines[i].rfind(":") for i in range(0,len(file_lines))]
    start = [pos[i] + 2 for i in range(0, len(pos))]
    end = [pos[i] + 8 for i in range(0, len(pos))]     
    v_acc = [file_lines[i][start[i]:end[i]].strip(" ,\n") for i in range(0,len(file_lines))]

    return max(v_acc)


def get_file_list():

    dir_names = glob.glob('UNet*')
    file_list = []
    print(dir_names)
    for dir_name in dir_names:
        file_list.append(os.path.join(dir_name, "accuracy.txt"))

    return dir_names, file_list


dir_names, file_list = get_file_list()

best_acc = [get_best_acc(file_name) for file_name in file_list]

list_to_write = list(zip(dir_names, best_acc))
print(list_to_write)

list_to_write = [" ".join(item) for item in list_to_write]
print(list_to_write)

f = open("best_acc.txt", "w")
[f.write(item + "\n") for item in list_to_write]

