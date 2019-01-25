from Globals import DATA_DIR, SEED, TRAIN_PATH, TEST_PATH, LABEL_PATH
import os
import random
import csv


# get the classes name
def get_class():
    return [classname for classname in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, classname))]


# write to the file
def write_label(classes):
    with open(os.path.join(DATA_DIR, 'Label.csv'), 'w') as lf:
        for i, name in enumerate(classes):
            lf.write(str(i)+','+name+'\n')


# get the all image path and the
def get_all_path(classes):
    dataset = []
    for i, labelname in enumerate(classes):
        imagelist = [DATA_DIR+'/'+labelname+'/'+imagename for imagename in os.listdir(os.path.join(DATA_DIR, labelname))]
        pairlist = list(zip(imagelist, [str(i) for j in range(len(imagelist))]))
        dataset.extend(pairlist)
    return dataset


# set the seed
def split_pathlist(dataset):
    random.seed(SEED)
    # double shuffle the list
    random.shuffle(dataset)
    random.shuffle(dataset)

    traindata = dataset[:int(len(dataset)*0.9)]
    testdata = dataset[int(len(dataset)*0.9):]
    return traindata, testdata


def write_path(train, test):
    with open(os.path.join(DATA_DIR, 'test.csv'), 'w') as f:
        for it in test:
            f.write(it[0]+','+it[1]+'\n')
    with open(os.path.join(DATA_DIR, 'train.csv'), 'w') as f:
        for it in train:
            f.write(it[0]+','+it[1]+'\n')


if __name__ == "__main__":
    classes = get_class()
    dataset = get_all_path(classes)
    train, test = split_pathlist(dataset)
    write_label(classes)
    write_path(train, test)
    # train_reader = csv.reader(open(TRAIN_PATH, 'r'))
    # # for it in train_reader:
    # #     print(it)
