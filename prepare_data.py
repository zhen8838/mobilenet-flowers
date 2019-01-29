import argparse
import sys
import os
import random
import csv


# get the classes name
def get_class():
    classnum = [classname for classname in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, classname))]
    classnum.sort()
    return classnum


# write to the file
def write_label(classes):
    with open(os.path.join(args.data_dir, 'Label.csv'), 'w') as lf:
        for i, name in enumerate(classes):
            lf.write(str(i)+','+name+'\n')


# get the all image path and the
def get_all_path(classes):
    dataset = []
    for i, labelname in enumerate(classes):
        imagelist = [args.data_dir+'/'+labelname+'/'+imagename for imagename in os.listdir(os.path.join(args.data_dir, labelname))]
        pairlist = list(zip(imagelist, [str(i) for j in range(len(imagelist))]))
        dataset.extend(pairlist)
    return dataset


# set the seed
def split_pathlist(dataset):
    random.seed(args.seed)
    # double shuffle the list
    random.shuffle(dataset)
    random.shuffle(dataset)

    traindata = dataset[:int(len(dataset)*0.9)]
    testdata = dataset[int(len(dataset)*0.9):]
    return traindata, testdata


def write_path(train, test):
    with open(os.path.join(args.data_dir, 'test.csv'), 'w') as f:
        for it in test:
            f.write(it[0]+','+it[1]+'\n')
    with open(os.path.join(args.data_dir, 'train.csv'), 'w') as f:
        for it in train:
            f.write(it[0]+','+it[1]+'\n')


def mian(args):
    classes = get_class()
    dataset = get_all_path(classes)
    train, test = split_pathlist(dataset)
    write_label(classes)
    write_path(train, test)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        help='Path to the data directory containing train file.',
                        default='/media/zqh/Datas/DataSet/flower_photos')

    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
