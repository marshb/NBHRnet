from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, multiply, Bidirectional, concatenate
from keras.layers import Conv2D, AveragePooling2D, Lambda, Reshape, PReLU, RepeatVector, Conv1D, Conv3D, MaxPooling2D
from keras.layers import LSTM, TimeDistributed, BatchNormalization, Softmax, UpSampling1D, UpSampling2D

from keras import regularizers


class NNModels():
    def __init__(self, input_shape=(60, 128, 128, 3)):
        self.input_shape = input_shape
        # self.threshold_atten = threshold_atten
    
    def TD_conv2d(self, last_layer, filters=32, kernel_size=(3, 3),
                  stride=(1, 1), padding="same", dilation_rate=(1, 1), name="tdc", reg=-1.0):
        if reg < 0:
            out = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                         padding=padding, dilation_rate=dilation_rate), name=name)(last_layer)
        else:
            out = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                         padding=padding, dilation_rate=dilation_rate,
                                         kernel_regularizer=regularizers.l2(reg)), name=name)(last_layer)
        return out
    
    def TD_conv2d_dropout(self, last_layer, filters=32, kernel_size=(3, 3),
                          stride=(1, 1), padding="same", dilation_rate=(1, 1), name="tdc", reg=-1.0):
        if reg < 0:
            out = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                         padding=padding, dilation_rate=dilation_rate), name=name)(last_layer)
        else:
            out = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                         padding=padding, dilation_rate=dilation_rate,
                                         kernel_regularizer=regularizers.l2(reg)), name=name)(last_layer)
        return out
    
    def parallel_dilated_conv(self, input, filters=16, name_base="D1", last_filters=64):
        # dalition convolution
        kernel_size = (3, 3)
        # dilation_rate=1
        
        d1_tdc1 = self.TD_conv2d(last_layer=input, filters=filters, name=name_base + "_d1t1")
        d1_tdc1 = TimeDistributed(PReLU(), name=name_base + "_d1t1P1")(d1_tdc1)
        
        d1_tdc1 = BatchNormalization(axis=-1, name=name_base + "_d1t1BN1")(d1_tdc1)
        d1_tdc2 = self.TD_conv2d(last_layer=d1_tdc1, filters=filters, stride=(2, 2), name=name_base + "_d1t2")
        d1_tdc2 = TimeDistributed(PReLU(), name=name_base + "_d1t1P2")(d1_tdc2)
        
        # dilation_rate=2
        d2_tdc1 = self.TD_conv2d(last_layer=input, filters=filters, dilation_rate=(2, 2),
                                 name=name_base + "_d2t1")
        d2_tdc1 = TimeDistributed(PReLU(), name=name_base + "_d2t1P1")(d2_tdc1)
        
        d2_tdc1 = BatchNormalization(axis=-1, name=name_base + "_d2t1BN1")(d2_tdc1)
        d2_tdc2 = self.TD_conv2d(last_layer=d2_tdc1, filters=filters, stride=(2, 2), name=name_base + "_d2t2")
        d2_tdc2 = TimeDistributed(PReLU(), name=name_base + "_d2t1P2")(d2_tdc2)
        
        # dilation_rate=3
        d3_tdc1 = self.TD_conv2d(last_layer=input, filters=filters, dilation_rate=(2, 2), name=name_base + "_d3t1")
        d3_tdc1 = TimeDistributed(PReLU(), name=name_base + "_d3t1P1")(d3_tdc1)
        
        d3_tdc1 = BatchNormalization(axis=-1, name=name_base + "_d3t1BN1")(d3_tdc1)
        d3_tdc2 = self.TD_conv2d(last_layer=d3_tdc1, filters=filters, stride=(2, 2), name=name_base + "_d3t2")
        d3_tdc2 = TimeDistributed(PReLU(), name=name_base + "_d3t1P2")(d3_tdc2)
        
        # concat and Conv
        concat_1 = concatenate(inputs=[d1_tdc2, d2_tdc2, d3_tdc2], axis=-1)
        r3 = BatchNormalization(axis=-1, name=name_base + "concatBN")(concat_1)
        r3 = TimeDistributed(Conv2D(last_filters, kernel_size=(1, 1), padding='same'), name=name_base + "fusion")(r3)
        r3 = TimeDistributed(PReLU(), name=name_base + "fusion_P1")(r3)
        return r3
    
    
    def PPG_extractor_model(self):
        input_video = Input(shape=self.input_shape)
    
        tdc1 = self.TD_conv2d(last_layer=input_video, filters=32, name="tdc1")
        tdc1 = TimeDistributed(PReLU())(tdc1)
        tdc1 = BatchNormalization(axis=-1)(tdc1)
    
        tdc2 = self.TD_conv2d(last_layer=tdc1, filters=32, stride=(2, 2), name="tdc2")
        tdc2 = TimeDistributed(PReLU())(tdc2)
        tdc2 = BatchNormalization(axis=-1)(tdc2)
    
        # dilation conv
        d_out1 = self.parallel_dilated_conv(tdc2, filters=16, name_base="D1", last_filters=48)
    
        d_out2 = self.parallel_dilated_conv(d_out1, filters=32, name_base="D2", last_filters=96)
    
        tdc3 = self.TD_conv2d(last_layer=d_out2, filters=128, name="tdc3")
        tdc3 = TimeDistributed(PReLU())(tdc3)
        tdc3 = BatchNormalization(axis=-1)(tdc3)
    
        tdc4 = self.TD_conv2d(last_layer=tdc3, filters=192, stride=(2, 2), name="tdc4")
        tdc4 = TimeDistributed(PReLU())(tdc4)
        tdc4 = BatchNormalization(axis=-1)(tdc4)
    
        tdc5 = self.TD_conv2d(last_layer=tdc4, filters=192, name="tdc5")
        tdc5 = TimeDistributed(PReLU())(tdc5)
        tdc5 = BatchNormalization(axis=-1)(tdc5)
    
        tdc6 = self.TD_conv2d(last_layer=tdc5, filters=256, stride=(2, 2), name="tdc6")
        tdc6 = TimeDistributed(PReLU())(tdc6)
        # d_tdc10 = BatchNormalization(axis=-1)(d_tdc10)
    
        pool_size = (4, 4)
        atten_pool = TimeDistributed(AveragePooling2D(pool_size))(tdc6)
        atten_squeez = Reshape((self.input_shape[0], 256))(atten_pool)
    
        bn_lstm1 = BatchNormalization(axis=-1)(atten_squeez)
        lstm_1 = Bidirectional(LSTM(96, return_sequences=True, dropout=0.25))(bn_lstm1)
        lstm_2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.25))(lstm_1)
        lstm_3 = LSTM(8, return_sequences=True)(lstm_2)
    
        bn_conv1d = BatchNormalization(axis=-1)(lstm_3)
        conv1d_1 = Conv1D(filters=1, kernel_size=(3,), padding="same", name="conv1d_1", activation="relu")(bn_conv1d)
        PPG_out = Reshape((60,), name="PPG_out")(conv1d_1)
        model = Model(inputs=input_video, outputs=PPG_out)
        return model
    
    
    def HR_extractor_model(self):
        input_PPG = Input(shape=(60, 1))
        lstm_1 = Bidirectional(LSTM(32, return_sequences=True), name="HR_bilstm1")(input_PPG)
        lstm_2 = Bidirectional(LSTM(24, return_sequences=True),
                               name="HR_bilstm2")(lstm_1)
        lstm_3 = Bidirectional(LSTM(8, return_sequences=True),
                               name="HR_bilstm3")(lstm_2)
        lstm_4 = LSTM(1, return_sequences=True, name="HR_lstm_4")(lstm_3)

        HR_lstm_squeez = Reshape((60,), name="HR_reshape")(lstm_4)

        dense_1 = Dense(32, activation="tanh", kernel_regularizer=regularizers.l2(0.001),
                        name="HR_dense_1")(HR_lstm_squeez)
        dense_1 = Dropout(0.25, name="HR_dropout_1")(dense_1)
        HR_out = Dense(1, activation="relu", name="HR_out")(dense_1)
        
        model = Model(inputs=input_PPG, outputs=HR_out)
        return model
    
    
    def Fusion_model(self):
        input_video = Input(shape=self.input_shape)

        tdc1 = self.TD_conv2d(last_layer=input_video, filters=32, name="tdc1")
        tdc1 = TimeDistributed(PReLU())(tdc1)
        tdc1 = BatchNormalization(axis=-1)(tdc1)

        tdc2 = self.TD_conv2d(last_layer=tdc1, filters=32, stride=(2, 2), name="tdc2")
        tdc2 = TimeDistributed(PReLU())(tdc2)
        tdc2 = BatchNormalization(axis=-1)(tdc2)

        # dilation conv
        d_out1 = self.parallel_dilated_conv(tdc2, filters=16, name_base="D1", last_filters=48)

        d_out2 = self.parallel_dilated_conv(d_out1, filters=32, name_base="D2", last_filters=96)

        tdc3 = self.TD_conv2d(last_layer=d_out2, filters=128, name="tdc3")
        tdc3 = TimeDistributed(PReLU())(tdc3)
        tdc3 = BatchNormalization(axis=-1)(tdc3)

        tdc4 = self.TD_conv2d(last_layer=tdc3, filters=192, stride=(2, 2), name="tdc4")
        tdc4 = TimeDistributed(PReLU())(tdc4)
        tdc4 = BatchNormalization(axis=-1)(tdc4)

        tdc5 = self.TD_conv2d(last_layer=tdc4, filters=192, name="tdc5")
        tdc5 = TimeDistributed(PReLU())(tdc5)
        tdc5 = BatchNormalization(axis=-1)(tdc5)

        tdc6 = self.TD_conv2d(last_layer=tdc5, filters=256, stride=(2, 2), name="tdc6")
        tdc6 = TimeDistributed(PReLU())(tdc6)
        # d_tdc10 = BatchNormalization(axis=-1)(d_tdc10)

        pool_size = (4, 4)
        atten_pool = TimeDistributed(AveragePooling2D(pool_size))(tdc6)
        atten_squeez = Reshape((self.input_shape[0], 256))(atten_pool)

        bn_lstm1 = BatchNormalization(axis=-1)(atten_squeez)
        lstm_1 = Bidirectional(LSTM(96, return_sequences=True, dropout=0.25))(bn_lstm1)
        lstm_2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.25))(lstm_1)
        lstm_3 = LSTM(8, return_sequences=True)(lstm_2)

        bn_conv1d = BatchNormalization(axis=-1)(lstm_3)
        conv1d_1 = Conv1D(filters=1, kernel_size=(3, ), padding="same", name="conv1d_1", activation="relu")(bn_conv1d)
        PPG_out = Reshape((60,), name="PPG_out")(conv1d_1)

        input_BN = BatchNormalization(axis=-1, name="HR_BN_input")(conv1d_1)
        lstm_1 = Bidirectional(LSTM(32, return_sequences=True), name="HR_bilstm1")(input_BN)
        lstm_2 = Bidirectional(LSTM(24, return_sequences=True),
                               name="HR_bilstm2")(lstm_1)
        lstm_3 = Bidirectional(LSTM(8, return_sequences=True),
                               name="HR_bilstm3")(lstm_2)
        lstm_4 = LSTM(1, return_sequences=True, name="HR_lstm_4")(lstm_3)

        HR_lstm_squeez = Reshape((60,), name="HR_reshape")(lstm_4)

        dense_1 = Dense(32, activation="tanh", kernel_regularizer=regularizers.l2(0.001), name="HR_dense_1")(
            HR_lstm_squeez)
        dense_1 = Dropout(0.25, name="HR_dropout_1")(dense_1)
        HR_out = Dense(1, activation="relu", name="HR_out")(dense_1)

        model = Model(inputs=input_video, outputs=[HR_out, PPG_out])

        return model

