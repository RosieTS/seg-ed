import os
import matplotlib.pyplot as plt
import numpy as np
import json


def get_loss_acc(file_name):
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

    loss_acc_file = open(file_name, "r")

    file_lines = loss_acc_file.readlines()

    pos = [file_lines[i].find(":") for i in range(0,len(file_lines))]
    start = [pos[i] + 2 for i in range(0, len(pos))]
    end = [pos[i] + 8 for i in range(0, len(pos))]     
    t_loss_acc = [file_lines[i][start[i]:end[i]].strip(" ,\n") for i in range(0,len(file_lines))]

    pos = [file_lines[i].rfind(":") for i in range(0,len(file_lines))]
    start = [pos[i] + 2 for i in range(0, len(pos))]
    end = [pos[i] + 8 for i in range(0, len(pos))]     
    v_loss_acc = [file_lines[i][start[i]:end[i]].strip(" ,\n") for i in range(0,len(file_lines))]

    return np.array(t_loss_acc, dtype = np.float32), np.array(v_loss_acc, dtype = np.float32)


def plot_losses_and_accuracies(loss_file = "losses.txt", acc_file = "accuracy.txt",
    command_file = "command_line_args.txt"):
    '''
    Plots losses / accuracies from files output by seg_epi.py or seg_UNet.py,
    showing key model parameters.

    Parameters
    ----------
    loss_file : str
        Name of losses file.
    acc_file : str
        Name of accuracies file.
    command_file : str
        Name of json file holding dictionary of command-line input parameters.

    Returns
    -------
    fig : matplotlib figure
        Figure with two subplots showing accuracies and losses by epoch.
    '''

    t_loss, v_loss = get_loss_acc(loss_file)
    t_acc, v_acc = get_loss_acc(acc_file)

    io = open(command_file, "r")
    command_args = json.load(io)

    fig, (axs1, axs2) = plt.subplots(nrows = 1, ncols = 2, figsize = [8,4])

    axs1.set_xlim(0, len(t_loss))
    axs1.plot(t_loss)
    axs1.plot(v_loss)
    axs1.legend(["Training", "Validation"])
    axs1.set_title("Losses")

    axs2.set_xlim(0, len(t_acc))
    axs2.set_ylim(0, 1)
    axs2.plot(t_acc)
    axs2.plot(v_acc)
    axs2.legend(["Training", "Validation"])
    axs2.set_title("Accuracies")

    fig.suptitle("BS= {}, LR={}, WD={}".format(command_args['bs'], command_args['lr'], command_args['wd']))

    return fig


if __name__ == "__main__":
    fig = plot_losses_and_accuracies()
    plt.show()
    fig.savefig("loss_acc.png", facecolor = "white")
    plt.close(fig)

