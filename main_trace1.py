#import os
#os.environ["THEANO_FLAGS"] = "device=gpu0, floatX=float32"
import numpy as np
from keras.layers import Dense, concatenate, LSTM, Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model, load_model
from keras.utils import to_categorical

class test_model(object):
    def __init__(self, test_path):
        self.load_data(test_path)

    def normalize(self, x):
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        return x

    def load_data(self, test_path):
        recover_test_files = np.load(test_path)
        self.x_test_dex_op = self.normalize(recover_test_files['x_test_dex'][:,0:11+256+1])
        self.x_test_dex_permission = recover_test_files['x_test_dex'][:,11+256+1:]
        self.x_test_dex_permission = np.expand_dims(self.x_test_dex_permission, axis=-1)
        self.x_test_sandbox = recover_test_files['x_test__sandbox']
        self.x_test_sandbox_1 = np.expand_dims(self.x_test_sandbox, axis = -1)
        self.sha1 = recover_test_files['y_test']

    def get_label_malware(self, model_malware_path):
        self.model = load_model(model_malware_path)
        self.pred_test = self.model.predict({'input_dex_op':self.x_test_dex_op, 'input_dex_permission': self.x_test_dex_permission,
                        'input_sandbox':self.x_test_sandbox, 'input_sandbox_1':self.x_test_sandbox_1}, verbose=True)[:,1]
        self.pred_test[np.where(self.pred_test>=0.5)] = 1
        self.pred_test[np.where(self.pred_test<0.5)] = 0
        return self.sha1, self.pred_test

    def get_label_family(self, model_family_path):
        self.model = load_model(model_family_path)
        self.pred_test = self.model.predict({'input_dex_op':self.x_test_dex_op, 'input_dex_permission': self.x_test_dex_permission,
                        'input_sandbox':self.x_test_sandbox,'input_sandbox_1':self.x_test_sandbox_1}, verbose=True)
        self.pred_label = np.argmax(self.pred_test,-1)+1
        return self.sha1, self.pred_label

if __name__ == '__main__':
    test_path = '../trace1_test.npz'
    malware_path = 'zzzz_4/dt_malware_10.h5' 
    family_path = 'zzzz_4/dt_family_10.h5'
    model = test_model(test_path=test_path) 
    sha1, malware = model.get_label_malware(malware_path)
    _, family = model.get_label_family(family_path)
    with open('trace1.csv', 'w') as fw:
        for sha1_, malware_, family_ in zip(sha1, malware, family):
            if malware_ == 0:
                fw.write(sha1_.decode('UTF-8')+','+np.array2string(malware_.astype(int))+','+'0'+'\n')
            else:
                fw.write(sha1_.decode('UTF-8')+','+np.array2string(malware_.astype(int))+','+np.array2string(family_.astype(int))+'\n')
    print('finish writing CSV')
