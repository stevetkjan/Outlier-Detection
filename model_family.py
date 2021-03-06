import os
os.environ["THEANO_FLAGS"] = "device=gpu1, floatX=float32"
import numpy as np
from keras.layers import Dense, concatenate, LSTM, Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional
from keras.models import Model
from keras.utils import to_categorical

# delete the file size and the header first
# onehot encoding for the permission list

#input layer:
# normalization data:
# word embedding
# MLP + RNN + CNN
# x_dex_op (261, )

class model(object):
    def __init__(self, data_path_1, data_path_2):
        self.load_data(data_path_1, data_path_2)
        self.build_model(dim_op=self.x_dex_op.shape[1], dim_per=self.x_dex_permission.shape[1],
                         dim_sand=self.x_sandbox.shape[1])

    def normalize(self, x):
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        return x

    def load_data(self, data_path_1, data_path_2, use_malware_only = 1):
        recover_files = np.load(data_path_1)
        dex = recover_files['x_train_dex']
        x_opcode_count = dex[:,0:11+256+1]
        self.x_dex_op = self.normalize(x_opcode_count)
        self.x_dex_permission = dex[:,11+256+1:]
        self.x_sandbox = recover_files['x_train__sandbox'].astype('float32')
        self.y_fal_1 = recover_files['y_train_family']
    
        recover_files_2 = np.load(data_path_2)
        dex_2 = recover_files_2['x_train_dex']
        x_opcode_count = dex_2[:,0:11+256+1]
        self.x_dex_op = np.vstack((self.x_dex_op,self.normalize(x_opcode_count)))
        self.x_dex_permission = np.vstack((self.x_dex_permission, dex_2[:,11+256+1:]))
        self.x_dex_permission = np.expand_dims(self.x_dex_permission, axis=-1)
        self.x_sandbox = np.vstack((self.x_sandbox, recover_files_2['x_train__sandbox'].astype('float32')))
        self.x_sandbox_1 = np.expand_dims(self.x_sandbox, axis=-1)
	    self.y_fal_1 = np.concatenate((self.y_fal_1, recover_files_2['y_train_family']))
        self.y_fal = to_categorical(self.y_fal_1)

        # print self.y_fal_1.shape
        # print np.where(self.y_fal_1==1)[0].shape[0]
        # print np.where(self.y_fal_1==2)[0].shape[0]
        # print np.where(self.y_fal_1==3)[0].shape[0]
        # print np.where(self.y_fal_1==4)[0].shape[0]
        # print np.where(self.y_fal_1==5)[0].shape[0]

        if use_malware_only == 1:
            nonzero_row = np.where(self.y_fal_1==0)[0]
            self.x_dex_op = np.delete(self.x_dex_op, nonzero_row, 0)
            self.x_dex_permission = np.delete(self.x_dex_permission, nonzero_row, 0)
            self.x_sandbox = np.delete(self.x_sandbox, nonzero_row, 0)
            self.y_fal_1 = np.delete(self.y_fal_1, nonzero_row, 0) - 1
	    #print self.y_fal_1
            self.y_fal = to_categorical(self.y_fal_1)
	    self.x_sandbox_1 = np.expand_dims(self.x_sandbox, axis=-1)
            #print self.x_dex_op.shape
            #print self.x_dex_permission.shape
            #print self.x_sandbox.shape
            #print self.y_fal_1.shape
            #print self.y_fal.shape

    def build_model(self, dim_op, dim_per, dim_sand):

        x_dex_op = Input(shape = (dim_op,), name = 'input_dex_op')
        x_dex_permission = Input(shape=(dim_per,1), name='input_dex_permission')
        x_sandbox = Input(shape=(dim_sand,), name='input_sandbox')
        x_sandbox_1 = Input(shape=(dim_sand,1), name='input_sandbox_1')

        x_dex_op_1 = Dense(200, input_dim=dim_op, activation='relu')(x_dex_op)
        x_dex_op_1 = Dropout(0.25)(x_dex_op_1)
        x_dex_permission_1 = Conv1D(filters=16, kernel_size=2, activation='relu')(x_dex_permission)
        x_dex_permission_2 = MaxPooling1D(4)(x_dex_permission_1)
        x_dex_permission_embedded = Flatten()(x_dex_permission_2)
        x_embedded = concatenate([x_dex_op_1, x_dex_permission_embedded])
        hidden_1 = Dense(100, activation='relu')(x_embedded)
        hidden_1 = Dropout(0.25)(hidden_1)
        x_sandbox_embedded = Embedding(201, 16, input_length = dim_sand)(x_sandbox)
        hidden_sandbox_1 = Bidirectional(LSTM(units=10, activation='tanh', input_shape = (dim_sand, 16), return_sequences=1))(x_sandbox_embedded)
        hidden_sandbox_2 = Bidirectional(LSTM(units=10, activation='tanh', input_shape = (dim_sand, 16), return_sequences=0))(hidden_sandbox_1)	
	    hidden_1_merged = concatenate([hidden_1, hidden_sandbox_2])
        hidden_2 = Dense(100, activation='relu')(hidden_1_merged)
        hidden_2 = Dropout(0.25)(hidden_2)
        hidden_3 = Dense(50, activation='relu')(hidden_2)
        hidden_3 = Dropout(0.25)(hidden_3)
        #hidden_4 = Dense(30, activation='relu')(hidden_3)
        output = Dense(5, activation='softmax')(hidden_3)
        self.model = Model(inputs = [x_dex_op, x_dex_permission, x_sandbox, x_sandbox_1], outputs = output)

    def train(self, batch_size, epochs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        self.model.fit({'input_dex_op':self.x_dex_op, 'input_dex_permission': self.x_dex_permission,
                        'input_sandbox':self.x_sandbox, 'input_sandbox_1':self.x_sandbox_1}, self.y_fal, batch_size = batch_size, epochs = epochs)
        self.model.save('dt_family_10.h5')
        acc_1 = self.model.evaluate({'input_dex_op':self.x_dex_op, 'input_dex_permission': self.x_dex_permission,
                                     'input_sandbox':self.x_sandbox, 'input_sandbox_1':self.x_sandbox_1}, self.y_fal, verbose=False)
        print acc_1

	
        
if __name__ == '__main__':
    data_path_1 = '../../trace1_train1.npz'
    data_path_2 = '../../trace1_train2.npz'

    batch_size = 3000
    epochs = 10
    model_test = model(data_path_1=data_path_1, data_path_2 = data_path_2)
    model_test.model.summary()
    model_test.train(batch_size, epochs)
