import tensorflow as tf
import os

from keras.callbacks import Callback
from utilities import save_network, load_network
from keras.models import Sequential
from topology import Convolutional, Pooling, Dropout, Block
from random import randint, choice
from copy import deepcopy


class Network:
    __slots__ = ('name', 'block_list', 'fitness', 'model')

    def __init__(self, it):
        self.name = 'parent_' + str(it) if it == 0 else 'net_' + str(it)
        self.block_list = []
        self.fitness = None
        self.model = None

    def build_model(self):
        model = Sequential()                                # Tạo mô hình Sequential
        for block in self.block_list:
            for layer in block.get_layers():                # Xây dựng mô hình
                try:
                    layer.build_layer(model)
                except:
                    print("\nBỎ QUA CÁ THỂ, TẠO MỘT CÁ THỂ MỚI\n")
                    return -1
        return model

    def train_and_evaluate(self, model, dataset):
        print("Đang huấn luyện", self.name)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(dataset['x_train'],
                            dataset['y_train'],
                            batch_size=dataset['batch_size'],
                            epochs=dataset['epochs'],
                            validation_data=(dataset['x_test'], dataset['y_test']),
                            shuffle=True)

        self.model = model                                    # Lưu mô hình
        self.fitness = history.history['val_loss'][-1]        # Đánh giá fitness

        print("SUMMARY OF", self.name)
        print(model.summary())
        print("FITNESS: ", self.fitness)

        model.save(self.name + '.h5')                       # Lưu mô hình
        save_network(self)                                  # Lưu cấu trúc, mô hình và fitness

    def asexual_reproduction(self, it, dataset):

        # Nếu cá thể đã tồn tại, tải nó lên
        if os.path.isfile('net_' + str(it) + '.h5'):
            print("\n-------------------------------------")
            print("Đang tải cá thể net_" + str(it))
            print("--------------------------------------\n")
            individual = load_network('net_' + str(it))
            model = tf.keras.models.load_model(individual.name + '.h5')
            print("SUMMARY OF", individual.name)
            print(model.summary())
            print("FITNESS: ", individual.fitness)
            return individual

        # Nếu không, tạo cá thể bằng cách đột biến từ cha mẹ
        individual = Network(it)

        print("\n-------------------------------------")
        print("\nKhởi tạo cá thể", individual.name)
        print("--------------------------------------\n")

        individual.block_list = deepcopy(self.block_list)           # Sao chép danh sách layer từ cha mẹ

        print("----->Đột biến mạnh")
        individual.block_mutation(dataset)                          # Đột biến khối
        individual.layer_mutation(dataset)                          # Đột biến lớp
        individual.parameters_mutation()                            # Đột biến tham số

        model = individual.build_model()

        if model == -1:
            return self.asexual_reproduction(it, dataset)

        individual.train_and_evaluate(model, dataset)

        return individual

    def block_mutation(self, dataset):
        print("Đột biến khối")

        print([(block.index, block.type) for block in self.block_list])

        # Danh sách khối chứa tất cả các khối có loại = 1
        bl = [block.index for block in self.block_list if block.type == 1]

        if len(bl) == 0:
            print("Tạo một khối mới với hai lớp Convolutional và một lớp Pooling")
            self.block_list[1].index = 2
            layerList1 = [
                Convolutional(filters=pow(2, randint(5, 8)),
                              filter_size=(3, 3),
                              stride_size=(1, 1),
                              padding='same',
                              input_shape=dataset['x_train'].shape[1:]),
                Convolutional(filters=pow(2, randint(5, 8)),
                              filter_size=(3, 3),
                              stride_size=(1, 1),
                              padding='same',
                              input_shape=dataset['x_train'].shape[1:])
            ]
            layerList2 = [
                Pooling(pool_size=(2, 2),
                        stride_size=(2, 2),
                        padding='same')
            ]
            b = Block(1, 1, layerList1, layerList2)
            self.block_list.insert(1, b)
            return

        block_idx = randint(1, max(bl))         # Chọn ngẫu nhiên một khối trong tất cả các khối có loại = 1
        block_type_idx = randint(0, 1)          # 1 -> Conv2D; 0 -> Pooling hoặc Dropout
        mutation_type = randint(0, 1)           # 1 -> xóa; 0 -> thêm

        # Danh sách các lớp của khối đã chọn
        layerList = self.block_list[block_idx].layerList1 if block_type_idx else self.block_list[block_idx].layerList2
        length = len(layerList)

        if mutation_type:                                       # Xóa
            if length == 1:
                del self.block_list[block_idx]
            elif block_type_idx:
                pos = randint(0, length - 1)
                print("Xóa một lớp Conv2D tại", pos)
                del layerList[pos]
            else:
                pos = randint(0, length - 1)
                print("Xóa một lớp Pooling/Dropout tại", pos)
                del layerList[pos]
        else:                                                   # Thêm
            if block_type_idx:
                print("Chèn một lớp Convolutional")
                layer = Convolutional(filters=pow(2, randint(5, 8)),
                                      filter_size=(3, 3),
                                      stride_size=(1, 1),
                                      padding='same',
                                      input_shape=dataset['x_train'].shape[1:])
                layerList.insert(randint(0, length - 1), layer)
            else:
                if randint(0, 1):                               # 1 -> Pooling; 0 -> Dropout
                    print("Chèn một lớp Pooling")
                    layer = Pooling(pool_size=(2, 2),
                                    stride_size=(2, 2),
                                    padding='same')
                    layerList.insert(randint(0, length - 1), layer)
                else:
                    print("Chèn một lớp Dropout")
                    rate = choice([0.15, 0.25, 0.35, 0.50])
                    layer = Dropout(rate=rate)
                    layerList.insert(randint(0, length - 1), layer)

    def layer_mutation(self, dataset):
        print("Đột biến lớp")

        # Chọn ngẫu nhiên một khối trong tất cả các khối có loại = 1
        bl = [block.index for block in self.block_list if block.type == 1]

        if len(bl) == 0:
            return

        block_idx = randint(1, max(bl))
        block_type_idx = randint(0, 1)      # 1 -> Conv2D; 0 -> Pooling hoặc Dropout

        # Danh sách các lớp của khối đã chọn
        layerList = self.block_list[block_idx].layerList1 if block_type_idx else self.block_list[block_idx].layerList2

        if len(layerList) == 0:
            if block_type_idx:
                layer = Convolutional(filters=pow(2, randint(5, 8)),
                                      filter_size=(3, 3),
                                      stride_size=(1, 1),
                                      padding='same',
                                      input_shape=dataset['x_train'].shape[1:])
                self.block_list[block_idx].layerList1.append(layer)
                return
            else:
                layer = Pooling(pool_size=(2, 2),
                                stride_size=(2, 2),
                                padding='same')
                self.block_list[block_idx].layerList2.append(layer)

        idx = randint(0, len(layerList) - 1)
        layer = layerList[idx]

        if layer.name == 'Conv2D':
            print("Tách lớp Conv2D tại index", idx)
            layer.filters = int(layer.filters * 0.5)
            layerList.insert(idx, deepcopy(layer))
        elif layer.name == 'MaxPooling2D' or layer.name == 'AveragePooling2D':
            print("Thay đổi lớp Pooling tại index", idx, "với lớp Conv2D")
            del layerList[idx]
            conv_layer = Convolutional(filters=pow(2, randint(5, 8)),
                                       filter_size=(3, 3),
                                       stride_size=(2, 2),
                                       padding=layer.padding,
                                       input_shape=dataset['x_train'].shape[1:])
            layerList.insert(idx, conv_layer)

    def parameters_mutation(self):
        print("Đột biến thông số")
        for block in self.block_list:
            for layer in block.get_layers():
                if randint(0, 1):
                    layer.mutate_parameters()
