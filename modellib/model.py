import tensorflow as tf
import keras
import numpy as np
from keras import layers as kl


class Res_Block(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        bneck_channels = out_channels // 4

        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(bneck_channels, kernel_size=1,
                               strides=1, padding='valid', use_bias=False)

        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(bneck_channels, kernel_size=3,
                               strides=1, padding='same', use_bias=False)

        self.bn3 = kl.BatchNormalization()
        self.av3 = kl.Activation(tf.nn.relu)
        self.conv3 = kl.Conv2D(out_channels, kernel_size=1,
                               strides=1, padding='valid', use_bias=False)

        self.shortcut = self._scblock(in_channels, out_channels)
        self.add = kl.Add()

    # Shortcut Connection
    def _scblock(self, in_channels, out_channels):
        if in_channels != out_channels:
            self.bn_sc1 = kl.BatchNormalization()
            self.conv_sc1 = kl.Conv2D(out_channels, kernel_size=1,
                                      strides=1, padding='same', use_bias=False)
            return self.conv_sc1
        else:
            return lambda x: x

    def call(self, x, training):
        out1 = self.conv1(self.av1(self.bn1(x, training=training)))
        out2 = self.conv2(self.av2(self.bn2(out1, training=training)))
        out3 = self.conv3(self.av3(self.bn3(out2, training=training)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])
        return out4
    

class miniResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim,layer_num=5):
        self.init_input_shape = input_shape
        super().__init__()
        self._kl = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(256, kernel_size=3, strides=(1, 1),
                      padding="same", activation="relu"),
            [Res_Block(256, 256) for _ in range(layer_num)],
            kl.GlobalAveragePooling2D(),
            kl.Dense(512, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]

    def call(self, x, training=True, isDebug=False):
        try:
            assert (self.init_input_shape == x.shape[1:])
        except AssertionError:
            raise AssertionError(f"Seems like input shape differs from init one.\n",
                                 f"init shape:{self.init_input_shape}",
                                 f"input shape:{x.shape[1:]}")
        for layer in self._kl:
            if isinstance(layer, list):
                for _layer in layer:
                    x = _layer(x, training=training)
                    if isDebug:
                        print(_layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
            else:
                if type(layer) == kl.BatchNormalization:
                    x = layer(x, training=training)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
                else:
                    x = layer(x)
                    if isDebug:
                        print(layer.name, x.shape, np.min(np.array(x)), np.max(
                            np.array(x)), np.mean(np.array(x)), np.std(np.array(x)))
        return x


