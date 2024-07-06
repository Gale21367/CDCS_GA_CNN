import keras.layers
from random import randint

class Block:
    __slots__ = ('type', 'index', 'layerList1', 'layerList2')

    def __init__(self, type, index, layerList1, layerList2):
        self.type = type                                       # 0 -> lớp ban đầu; 1 -> các lớp giữa; 2 -> lớp cuối cùng
        self.index = index                                     # chỉ số của khối trong tất cả các khối
        self.layerList1 = layerList1                           # Các lớp convolutional
        self.layerList2 = layerList2                           # Các lớp pooling và dropout

    def get_layers(self):
        return self.layerList1 + self.layerList2

    def get_size(self):
        return len(self.get_layers())


class Convolutional:
    # __slots__ = ('name', 'filters', 'padding', 'filter_size', 'stride_size', 'input_shape')

    def __init__(self, filters, padding, filter_size, stride_size, input_shape):
        self.name = 'Conv2D'
        self.filters = filters
        self.padding = padding
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.input_shape = input_shape

    def build_layer(self, model):
        model.add(keras.layers.Conv2D(filters=self.filters,
                                       kernel_size=self.filter_size,
                                       strides=self.stride_size,
                                       padding=self.padding,
                                       activation='relu',
                                       kernel_initializer='he_uniform',
                                       input_shape=self.input_shape))

    def mutate_parameters(self):
        mutation = randint(0, 4)
        print("Đột biến lớp", self.name, ":")
        if mutation == 0 and self.filters >= 32:
            print("--> thay đổi self.filters từ ", self.filters, " thành ", end="")
            self.filters = int(self.filters / 2)
            print(self.filters)
        elif mutation == 1 and self.filters >= 32:
            print("--> thay đổi self.filters từ ", self.filters, " thành ", end="")
            self.filters = int(self.filters / 2)
            print(self.filters)
        elif mutation == 2 and self.filters <= 512:
            print("--> thay đổi self.filters từ ", self.filters, " thành ", end="")
            self.filters *= 2
            print(self.filters)
        elif mutation == 3 and self.filters <= 512:
            print("--> thay đổi self.filters từ ", self.filters, " thành ", end="")
            self.filters *= 2
            print(self.filters)
        elif mutation == 4:
            if self.padding == 'valid':
                print("--> thay đổi self.padding từ ", self.padding, " thành ", end="")
                self.padding = 'same'
                print(self.padding)
            else:
                print("--> thay đổi self.padding từ ", self.padding, " thành ", end="")
                self.padding = 'valid'
                print(self.padding)


class Pooling:
    __slots__ = ('name', 'pool_size', 'stride_size', 'padding')

    def __init__(self, pool_size, stride_size, padding):
        self.name = 'MaxPooling2D'
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.padding = padding

    def build_layer(self, model):
        if self.name == 'MaxPooling2D':
            model.add(keras.layers.MaxPooling2D(self.pool_size, self.stride_size, self.padding))
        elif self.name == 'AveragePooling2D':
            model.add(keras.layers.AveragePooling2D(self.pool_size, self.stride_size, self.padding))

    def mutate_parameters(self):
        print("Đột biến lớp", self.name, ":")
        mutation = randint(0, 1)
        if mutation == 0:
            if self.padding == 'valid':
                print("--> thay đổi self.padding từ ", self.padding, " thành ", end="")
                self.padding = 'same'
                print(self.padding)
            else:
                print("--> thay đổi self.padding từ ", self.padding, " thành ", end="")
                self.padding = 'valid'
                print(self.padding)
        elif mutation == 1:
            if self.name == 'MaxPooling2D':
                print("--> thay đổi self.name từ ", self.name, " thành ", end="")
                self.name = 'AveragePooling2D'
                print(self.name)
            else:
                print("--> thay đổi self.name từ ", self.name, " thành ", end="")
                self.name = 'MaxPooling2D'
                print(self.name)


class FullyConnected:
    __slots__ = ('name', 'units', 'num_classes')

    def __init__(self, units, num_classes):
        self.name = "FullyConnected"
        self.units = units
        self.num_classes = num_classes

    def build_layer(self, model):
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.units, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))

    def mutate_parameters(self):
        print("Đột biến lớp", self.name, ":")
        mutation = randint(0, 2)
        if mutation == 0:
            print("--> thay đổi self.units từ ", self.units, " thành ", end="")
            self.units *= 2
            print(self.units)
        elif mutation == 1:
            print("--> thay đổi self.units từ ", self.units, " thành ", end="")
            self.units *= 2
            print(self.units)
        elif mutation == 2:
            print("--> thay đổi self.units từ ", self.units, " thành ", end="")
            self.units /= 2
            print(self.units)


class Dropout:
    __slots__ = ('name', 'rate')

    def __init__(self, rate):
        self.name = "Dropout"
        self.rate = rate

    def build_layer(self, model):
        model.add(keras.layers.Dropout(self.rate))

    def mutate_parameters(self):
        print("Đột biến lớp", self.name, ":")
        mutation = randint(0, 3)
        if mutation == 0 and self.rate <= 0.85:
            print("--> thay đổi self.rate từ ", self.rate, " thành ", end="")
            self.rate = self.rate + 0.10
            print(self.rate)
        elif mutation == 1 and self.rate <= 0.90:
            print("--> thay đổi self.rate từ ", self.rate, " thành ", end="")
            self.rate = self.rate + 0.05
            print(self.rate)
        elif mutation == 2 and self.rate >= 0.15:
            print("--> thay đổi self.rate từ ", self.rate, " thành ", end="")
            self.rate = self.rate - 0.10
            print(self.rate)
        elif mutation == 3 and self.rate >= 0.10:
            print("--> thay đổi self.rate từ ", self.rate, " thành ", end="")
            self.rate = self.rate - 0.05
            print(self.rate)
