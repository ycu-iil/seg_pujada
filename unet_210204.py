from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization


class UNet(object):
    def __init__(self, input_size, add_drop_layer=False, dropout_ratio = 0, add_drop_layer_encoder=False, layer_num = 5, BN=0, nb_class = 5):
        self.INPUT_SIZE = input_size

        inputs = Input((self.INPUT_SIZE, self.INPUT_SIZE, 3))
        print(inputs.shape)

        
        encodeLayer1 = self.__add_encode_layers(64, inputs, is_first=True, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
        encodeLayer2 = self.__add_encode_layers(128, encodeLayer1, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
        encodeLayer3 = self.__add_encode_layers(256, encodeLayer2, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
        encodeLayer4 = self.__add_encode_layers(512, encodeLayer3, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
        encodeLayer5 = self.__add_encode_layers(1024, encodeLayer4, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
        

        
        if layer_num == 6:
            encodeLayer6 = self.__add_encode_layers(1024, encodeLayer5, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
            decodeLayer0 = self.__add_decode_layers(
                512, encodeLayer6, encodeLayer5, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
            encodeLayer5 = decodeLayer0
        elif layer_num == 7:
            encodeLayer6 = self.__add_encode_layers(1024, encodeLayer5, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
            encodeLayer7 = self.__add_encode_layers(1024, encodeLayer6, add_drop_layer=add_drop_layer_encoder, dropout_ratio = dropout_ratio, BN=BN)
            
            decodeLayerM1 = self.__add_decode_layers(
                512, encodeLayer7, encodeLayer6, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
            decodeLayer0 = self.__add_decode_layers(
                512, decodeLayerM1, encodeLayer5, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
            encodeLayer5 = decodeLayer0

        decodeLayer1 = self.__add_decode_layers(
            512, encodeLayer5, encodeLayer4, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
        decodeLayer2 = self.__add_decode_layers(
            256, decodeLayer1, encodeLayer3, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
        decodeLayer3 = self.__add_decode_layers(
            128, decodeLayer2, encodeLayer2, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
        decodeLayer4 = self.__add_decode_layers(
            64, decodeLayer3, encodeLayer1, add_drop_layer=add_drop_layer, dropout_ratio = dropout_ratio)
        
        #nb_class = 5
        h = Conv2D(nb_class, (1, 1), activation = 'relu')(decodeLayer4)
        op = Activation('softmax')(h)
        self.MODEL = Model(inputs = [inputs], outputs = op)

        #outputs = Conv2D(1, 1, activation='sigmoid')(decodeLayer4)
        #print(outputs.shape)

        #self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def __add_encode_layers(self, filter_size, input_layer, is_first=False, add_drop_layer=False, dropout_ratio = 0, BN = 0):
        layer = input_layer
        if BN == 0:
            layer = BatchNormalization()(layer)
            
        
        if is_first:
            layer = Conv2D(filter_size, 3, padding='same', input_shape=(
                self.INPUT_SIZE, self.INPUT_SIZE, 3))(layer)
        else:
            layer = MaxPooling2D(2)(layer)
            layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = Conv2D(filter_size, 3, padding='same')(layer)
        
        if BN == 1:
            layer = BatchNormalization()(layer)
        
        layer = Activation(activation='relu')(layer)
        
        if add_drop_layer:
            layer = Dropout(dropout_ratio)(layer)
        print(layer.shape)
        return layer

    def __add_decode_layers(self, filter_size, input_layer, concat_layer, add_drop_layer=False, dropout_ratio = 0):
        layer = UpSampling2D(2)(input_layer)
        layer = concatenate([layer, concat_layer])
        layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)
        if add_drop_layer:
            layer = Dropout(dropout_ratio)(layer)
        print(layer.shape)
        return layer

    def model(self):
        return self.MODEL
