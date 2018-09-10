import numpy as np
import os
import collections
from keras.preprocessing.sequence import pad_sequences

def read_data(data_dir, flag):
    if flag == 'train':
        label = np.loadtxt(os.path.join(data_dir, 'safetype.txt'), dtype={'names':('id', 'y1', 'y2'), 'formats':('|S32', np.int, np.int)}, delimiter=";")

        label_dict_1 = dict([(x[0],x[1]) for x in label])
        label_dict_2 = dict([(x[0],x[2]) for x in label])

        train_dex_1 = dict()
        train_dex_2 = dict()
        train_dex_3 = dict()

        f_dex = open(os.path.join(data_dir,'dex.txt'), 'r')
        raw_data = f_dex.readlines()
        raw_data = raw_data[1:]
        for raw_data_ in raw_data:
            a = raw_data_.split(';')[1:8]
            a.extend(raw_data_.split(';')[10:])
            train_dex_1[raw_data_.split(';')[0]]=a
            train_dex_2[raw_data_.split(';')[0]]=raw_data_.split(';')[8].split(',')
            if raw_data_.split(';')[9].split(',')[0] == '':
                train_dex_3[raw_data_.split(';')[0]] = '0'
            else:
                train_dex_3[raw_data_.split(';')[0]] = raw_data_.split(';')[9].split(',')
        train_sandbox = dict()
        f_dex = open(os.path.join(data_dir,'sandbox_behaviorlist.txt'), 'r')
        raw_data = f_dex.readlines()
        raw_data = raw_data[1:]
        for raw_data_ in raw_data:
            if raw_data_.split(';')[1] == '\n':
                train_sandbox[raw_data_.split(';')[0]] = '0'
            else:
                train_sandbox[raw_data_.split(';')[0]] = raw_data_.split(';')[1].split(',')


        label_1 = []
        label_2 = []
        x_1 = []
        x_2 = []
        x_3 = []
        x_sandbox = []

        for key in label_dict_1.keys():
            label_1.append(label_dict_1[key])
            label_2.append(label_dict_2[key])
            x_1.append(map(int, train_dex_1[key]))
            x_2.append(map(int, train_dex_2[key]))
            x_3.append(map(int, train_dex_3[key]))
            x_sandbox.append(map(int, train_sandbox[key]))
        y_train_1 = np.array(label_1)
        y_train_2 = np.array(label_2)

        x_train_1 = np.array(x_1)
        x_train_2 = np.array(x_2)
        x_train_1 = np.concatenate((x_train_1, x_train_2), axis=1)
        x_train_3 = np.zeros((x_train_1.shape[0], 147))
        for i in xrange(x_train_1.shape[0]):
            tmp = np.array(x_3[i])
            x_train_3[i, np.unique(tmp)] = 1
        x_train_dex = np.concatenate((x_train_1, x_train_3), axis=1)
        x_train_sandbox = pad_sequences(x_sandbox, padding='post')

        print y_train_1.shape
        print y_train_2.shape
        print x_train_dex.shape
        print x_train_sandbox.shape

        np.savez(data_dir + '.npz', y_train_malware = y_train_1, y_train_family = y_train_2,
                 x_train_dex = x_train_dex, x_train__sandbox = x_train_sandbox)

    elif flag == 'test':
        test_dex_1 = dict()
        test_dex_2 = dict()
        test_dex_3 = dict()

        f_dex = open(os.path.join(data_dir, 'dex.txt'), 'r')
        raw_data = f_dex.readlines()
        raw_data = raw_data[1:]
        for raw_data_ in raw_data:
            a = raw_data_.split(';')[1:8]
            a.extend(raw_data_.split(';')[10:])
            test_dex_1[raw_data_.split(';')[0]] = a
            test_dex_2[raw_data_.split(';')[0]] = raw_data_.split(';')[8].split(',')
            if raw_data_.split(';')[9].split(',')[0] == '':
                test_dex_3[raw_data_.split(';')[0]] = '0'
            else:
                test_dex_3[raw_data_.split(';')[0]] = raw_data_.split(';')[9].split(',')

        test_sandbox = dict()
        f_dex = open(os.path.join(data_dir, 'sandbox_behaviorlist.txt'), 'r')
        raw_data = f_dex.readlines()
        raw_data = raw_data[1:]
        for raw_data_ in raw_data:
            if raw_data_.split(';')[1] == '\r\n':
                test_sandbox[raw_data_.split(';')[0]] = '0'
            else:
                test_sandbox[raw_data_.split(';')[0]] = raw_data_.split(';')[1].split(',')

        x_1 = []
        x_2 = []
        x_3 = []
        x_sandbox = []
        label = []

        for key in test_dex_1.keys():
            label.append(key)
            x_1.append(map(int, test_dex_1[key]))
            x_2.append(map(int, test_dex_2[key]))
            x_3.append(map(int, test_dex_3[key]))
            x_sandbox.append(map(int, test_sandbox[key]))

        y_test = np.array(label)
        x_test_1 = np.array(x_1)
        x_test_2 = np.array(x_2)
        x_test_1 = np.concatenate((x_test_1, x_test_2), axis=1)
        x_test_3 = np.zeros((x_test_1.shape[0], 147))
        for i in xrange(x_test_1.shape[0]):
            tmp = np.array(x_3[i])
            x_test_3[i, np.unique(tmp)] = 1
        x_test_dex = np.concatenate((x_test_1, x_test_3), axis=1)
        x_test_sandbox = pad_sequences(x_sandbox, padding='post')

        print y_test.shape
        print x_test_dex.shape
        print x_test_sandbox.shape

        np.savez(data_dir + '.npz', y_test=y_test,
                 x_test_dex=x_test_dex, x_test__sandbox=x_test_sandbox)

read_data(data_dir='../trace1_test', flag='test')