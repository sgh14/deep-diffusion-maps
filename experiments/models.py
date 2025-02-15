from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


class EncoderHead(Layer):
    def __init__(self, n_components, use_bn=False, **kwargs):
        super(EncoderHead, self).__init__(**kwargs)
        self.n_components = n_components
        self.use_bn = use_bn
        self.batch_normalization = BatchNormalization() if use_bn else None
        self.dense = None  # Initialize in build

    def build(self, input_shape):
        """This method initializes the weights once the input shape is known."""
        self.dense = Dense(self.n_components, activation='linear', use_bias=not self.use_bn)
        super().build(input_shape)  # Important: Call the parent build method

    def call(self, inputs, training=False):
        if self.use_bn:
            inputs = self.batch_normalization(inputs, training=training)
        return self.dense(inputs)
    

def build_encoder(input_shape, units, n_components, activation='relu', use_bn=False):
    encoder = Sequential([
        Input(shape=input_shape),
        Dense(units, activation=activation),
        Dense(units//2, activation=activation),
        Dense(units//4, activation=activation),
        EncoderHead(n_components, use_bn)
    ], name='encoder')

    return encoder


class ConvBlock2D(Layer):
    def __init__(self, filters, dropout=0.0, **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)
        self.conv1 = Conv2D(
            filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.conv2 = Conv2D(
            filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.dropout = Dropout(dropout)
        self.maxpool = MaxPool2D(pool_size=(2, 2))


    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(inputs=x, training=training)

        return x


def build_conv_encoder(input_shape, filters, n_components, zero_padding=(0, 0), dropout=0.0, use_bn=False):
    encoder = Sequential([
        Input(shape=input_shape),
        ZeroPadding2D(zero_padding),
        ConvBlock2D(filters, dropout=dropout),
        ConvBlock2D(filters*2, dropout=dropout),
        ConvBlock2D(filters*4, dropout=dropout),
        Flatten(),
        Dense(16*n_components, activation='relu'),
        EncoderHead(n_components, use_bn)
    ], name='encoder')

    return encoder


class ConvBlock1D(Layer):
    def __init__(self, filters, dropout=0.0, **kwargs):
        super(ConvBlock1D, self).__init__(**kwargs)
        self.conv1 = Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.conv2 = Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.dropout = Dropout(dropout)
        self.maxpool = MaxPool1D(pool_size=2)


    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(inputs=x, training=training)

        return x


# def build_seq_encoder(input_shape, filters, n_components, zero_padding=0, dropout=0.0, use_bn=False):
#     encoder = Sequential([
#         Input(shape=input_shape),
#         ZeroPadding1D(zero_padding),
#         ConvBlock1D(filters, dropout=dropout),
#         ConvBlock1D(filters*2, dropout=dropout),
#         ConvBlock1D(filters*4, dropout=dropout),
#         Flatten(),
#         EncoderHead(n_components, use_bn)
#     ], name='encoder')

#     return encoder


def build_seq_encoder(input_shape, units, n_components, dropout=0.0, use_bn=False):
    encoder = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units, return_sequences=True, dropout=dropout)),
        Bidirectional(LSTM(units * 2, return_sequences=True, dropout=dropout)),
        Bidirectional(LSTM(units * 4, return_sequences=False, dropout=dropout)),  # No return_sequences for final layer
        EncoderHead(n_components, use_bn)
    ], name='encoder')

    return encoder