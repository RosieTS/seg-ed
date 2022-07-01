import json
import ast

def read_in_annotations(filepath):

    with open(filepath, 'r') as annot_file:
        annotations = annot_file.readlines()

    annot_split = [annotations[i].split("{", 1) for i in range(0, len(annotations))]

    # Remove header line by starting range at 1.
    annot_data = ["{" + annot_split[i][1].strip() for i in range(1, len(annot_split))]

    data_dict = []

    for i in range(0, len(annot_data)):
        #print(i)
        data_dict.append(ast.literal_eval(annot_data[i]))
        #print(type(data_dict[i-1]))

    return data_dict

filepath = '../aida_data/annotations.csv'
data_dict = read_in_annotations(filepath)
print(len(data_dict))