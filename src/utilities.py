import pickle
import matplotlib.pyplot as plt
import time
from keras.datasets import cifar10
from keras.utils import to_categorical

def load_dataset(batch_size, num_classes, epochs):
    print("Đang tải dữ liệu CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("Dữ liệu đã được tải thành công.")

    print("Đang chuyển đổi kiểu dữ liệu...")
    x_train = x_train.astype('float32')  # chuyển từ số nguyên sang số thực
    x_test = x_test.astype('float32')

    print("Đang chuẩn hóa dữ liệu về khoảng 0-1...")
    x_train /= 255  # chuẩn hóa về khoảng 0-1
    x_test /= 255

    print("Đang thực hiện mã hóa one-hot cho nhãn...")
    y_train = to_categorical(y_train, num_classes)  # mã hóa one-hot cho nhãn
    y_test = to_categorical(y_test, num_classes)

    dataset = {
        'batch_size': batch_size,
        'num_classes': num_classes,
        'epochs': epochs,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return dataset

def save_network(network):
    object_file = open(network.name + '.obj', 'wb')
    pickle.dump(network, object_file)

def load_network(name):
    object_file = open(name + '.obj', 'rb')
    return pickle.load(object_file)

def order_indexes(self):
    i = 0
    for block in self.block_list:
        block.index = i
        i += 1

def plot_training(history):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    filename = 'loss_plot_' + str(int(time.time())) + '.png'
    plt.savefig(filename)
    plt.show()

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    filename = 'acc_plot_' + str(int(time.time())) + '.png'
    plt.savefig(filename)
    plt.show()


def plot_statistics(stats):
    plt.figure(figsize=[8, 6])
    plt.plot([s[0] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][0]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['BestFitness', 'InitialFitness'], fontsize=18)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('FitnessValue', fontsize=16)
    plt.title('Fitness Curve', fontsize=16)
    filename = 'fitness_plot_' + str(int(time.time())) + '.png'
    plt.savefig(filename)
    plt.close() 

    plt.figure(figsize=[8, 6])
    plt.plot([s[1] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][1]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['BestParamsNum', 'InitialParamsNum'], fontsize=18)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('ParamsNum', fontsize=16)
    plt.title('Parameters Curve', fontsize=16)
    filename = 'params_plot_' + str(int(time.time())) + '.png'
    plt.savefig(filename)
    plt.close() 
