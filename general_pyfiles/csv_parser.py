'''
File with a function to get data from the dataset and save it to python dictionarys to be used in train and testing. The script was taken from
https://github.com/you359/Keras-FasterRCNN/blob/master/keras_frcnn/simple_parser.py and adapted to read csv files instead of txt and with the
folder structure that we use, in order to use the same files and structure used on the Tensorflow Object API Detection
'''

#csv module used instead of pandas because iterating through rows in pandas is inneficient
import csv
import os


def read_file(input_path, train, all_imgs, class_mapping, classes_count):
    '''
    Method that does what is described at the start of the file.
    input_path should be the path to the csv file with the data. 
    added train as argument, that is a boolean indicating if the file passed is for training or testing (In the original script, only 1 file
    was used to both train and vailidation, and the separation was done randomly)
    all_imgs, class_mapping and classes_count used globally to save train and validation data at the same time.
    '''
    with open(input_path,'r') as f:

        print('Parsing annotation files')

        reader = csv.reader(f, delimiter=',')

        for i,line in enumerate(reader):
            if i == 0:
                continue
            (filename,width,height,class_name,x1,y1,x2,y2) = line

            #Add the folder to the filename in order to have the folder structure that we are using
            if train:
                filename = os.path.join("dataset","train", filename)
            else:
                filename = os.path.join("dataset","test", filename)

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = int(width)
                all_imgs[filename]['height'] = int(height)
                all_imgs[filename]['bboxes'] = []
                if train:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


def get_data(dict_path):
    '''
    Method that is used to read train and test labels files. After reading them, it passes the dictionary associated with every file to a
    single array
    '''
    #Dictionary that, for every image, saves all the information of taken from the csv file into another image.
    all_imgs = {}
    #For every class, checks how many times it appears
    classes_count = {}
    #For every class, assigns a unique id (starting from 0 and increasing 1 by 1)
    class_mapping = {}

    read_file(os.path.join(dict_path, "train_labels.csv"), True, all_imgs, class_mapping, classes_count)
    read_file(os.path.join(dict_path, "test_labels.csv"), False, all_imgs, class_mapping, classes_count)

    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    return all_data, classes_count, class_mapping
