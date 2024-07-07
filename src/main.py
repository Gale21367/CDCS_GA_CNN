import tensorflow as tf
import os
from network import Network
from inout import compute_parent
from random import randint, sample
from utilities import order_indexes, plot_training, plot_statistics, load_network
from copy import deepcopy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)      # Tắt các thông báo từ Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def initialize_population(population_size, dataset):
    print("----->Khởi tạo Quần thể")
    daddy = compute_parent(dataset)                                 # Tải cá thể cha mẹ từ đầu vào
    population = [daddy]
    for it in range(1, population_size):
        population.append(daddy.asexual_reproduction(it, dataset))

    # Sắp xếp quần thể theo thứ tự tăng dần dựa trên độ thích nghi (fitness)
    return sorted(population, key=lambda cnn: cnn.fitness)


def selection(k, population, num_population):
    if k == 0:                                              # Lựa chọn bảo toàn
        print("----->Lựa chọn Elitism")
        return population[0], population[1]
    elif k == 1:                                            # Lựa chọn giải đấu
        print("----->Lựa chọn Tournament")
        i = randint(0, num_population - 1)
        j = i
        while j < num_population - 1:
            j += 1
            if randint(1, 100) <= 50:
                return population[i], population[j]
        return population[i], population[0]
    else:                                                   # Lựa chọn tỷ lệ
        print("----->Lựa chọn Proportionate")
        cum_sum = 0
        for i in range(num_population):
            cum_sum += population[i].fitness
        perc_range = []
        for i in range(num_population):
            count = 100 - int(100 * population[i].fitness / cum_sum)
            for j in range(count):
                perc_range.append(i)
        i, j = sample(range(1, len(perc_range)), 2)
        while i == j:
            i, j = sample(range(1, len(perc_range)), 2)
        return population[perc_range[i]], population[perc_range[j]]


def crossover(parent1, parent2, it):
    print("----->Sự Giao thoa")
    child = Network(it)

    first, second = None, None
    if randint(0, 1):
        first = parent1
        second = parent2
    else:
        first = parent2
        second = parent1

    child.block_list = deepcopy(first.block_list[:randint(1, len(first.block_list) - 1)]) \
                       + deepcopy(second.block_list[randint(1, len(second.block_list) - 1):])

    order_indexes(child)                            # Sắp xếp các chỉ số của các khối

    return child


def genetic_algorithm(num_population, num_generation, num_offspring, dataset, resume_from_checkpoint=None):
    print("Thuật toán Di truyền")

    population = initialize_population(num_population, dataset)

    if resume_from_checkpoint:
        # Tải mô hình từ checkpoint nếu có
        print(f"Tiếp tục từ điểm kiểm soát: {resume_from_checkpoint}")
        model = tf.keras.models.load_model(resume_from_checkpoint)
        # Cập nhật cá thể ban đầu với mô hình đã tải
        population[0].model = model
        population[0].train_and_evaluate(model, dataset)

    print("\n-------------------------------------")
    print("Quần thể Ban đầu:")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)
    print("--------------------------------------\n")

    # Để in thống kê về độ thích nghi và số lượng tham số của cá thể tốt nhất
    stats = [(population[0].fitness, population[0].model.count_params())]

    for gen in range(1, num_generation + 1):

        '''
            k là tham số lựa chọn:
                k = 0 -> Lựa chọn elitism
                k = 1 -> Lựa chọn tournament
                k = 2 -> Lựa chọn proportionate
        '''
        k = randint(0, 2)

        print("\n------------------------------------")
        print("Thế hệ", gen)
        print("-------------------------------------")

        for c in range(num_offspring):

            print("\nTạo Ra Con cái", c)

            parent1, parent2 = selection(k, population, num_population)                 # Lựa chọn
            print("Chọn", parent1.name, "và", parent2.name, "để sinh sản")

            child = crossover(parent1, parent2, c + num_population)                     # Giao thoa
            print("Đã tạo ra Con cái")

            print("----->Đột biến Nhẹ")
            child.layer_mutation(dataset)                                               # Đột biến
            child.parameters_mutation()
            print("Đã đột biến Con cái")

            model = child.build_model()                                                 # Đánh giá

            while model == -1:
                child = crossover(parent1, parent2, c + num_population)
                child.block_mutation(dataset)
                child.layer_mutation(dataset)
                child.parameters_mutation()
                model = child.build_model()

            child.train_and_evaluate(model, dataset)

            if child.fitness < population[-1].fitness:                                  # Tiến hóa quần thể
                print("----->Tiến hóa: Con cái", child.name, "với độ thích nghi", child.fitness, "thay thế cha mẹ ", end="")
                print(population[-1].name, "với độ thích nghi", population[-1].fitness)
                name = population[-1].name
                population[-1] = deepcopy(child)
                population[-1].name = name
                population = sorted(population, key=lambda net: net.fitness)
            else:
                print("----->Tiến hóa: Con cái", child.name, "với độ thích nghi", child.fitness, "bị loại bỏ")

        stats.append((population[0].fitness, population[0].model.count_params()))

        # Lưu checkpoint sau mỗi thế hệ
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'generation_{gen}.h5')
        population[0].model.save(checkpoint_path)

    print("\n\n-------------------------------------")
    print("Quần thể Cuối cùng")
    print("-------------------------------------\n")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)

    print("\n-------------------------------------")
    print("Thống kê")
    for i in range(len(stats)):
        print("Cá thể tốt nhất ở thế hệ", i + 1, "có độ thích nghi", stats[i][0], "và số lượng tham số", stats[i][1])
    print("-------------------------------------\n")

    # Vẽ biểu đồ về độ thích nghi và số lượng tham số của cá thể tốt nhất ở mỗi vòng lặp
    plot_statistics(stats)

    return population[0]


def main():
    '''
        dataset chứa các siêu tham số để tải dữ liệu và tập dữ liệu:
            dataset = {
                'batch_size': batch_size,
                'num_classes': num_classes,
                'epochs': epochs,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test
            }
    '''
    dataset = load_dataset(batch_size, num_classes, epochs)
    resume_from_checkpoint = None  # Đặt lại thành đường dẫn của checkpoint nếu bạn muốn tiếp tục từ checkpoint

    # Vẽ biểu đồ về mô hình tốt nhất thu được
    optCNN = genetic_algorithm(num_population, num_generation, num_offspring, dataset, resume_from_checkpoint)

    # Vẽ biểu đồ về sự mất mát và độ chính xác trong quá trình huấn luyện và xác thực
    model = optCNN.build_model()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(dataset['x_train'],
                        dataset['y_train'],
                        batch_size=dataset['batch_size'],
                        epochs=num_epoch,
                        validation_data=(dataset['x_test'], dataset['y_test']),
                        shuffle=True)
    optCNN.model = model                                        # model
    optCNN.fitness = history.history['val_loss'][-1]            # fitness

    print("\n\n-------------------------------------")
    print("CNN ban đầu đã được tiến hóa thành công trong các cá thể", optCNN.name)
    print("-------------------------------------\n")
    daddy = load_network('parent_0')
    model = tf.keras.models.load_model('parent_0.h5')
    print("\n\n-------------------------------------")
    print("Tóm tắt của CNN ban đầu")
    print(model.summary())
    print("Fitness của CNN ban đầu:", daddy.fitness)

    print("\n\n-------------------------------------")
    print("Tóm tắt của cá nhân tiến hóa")
    print(optCNN.model.summary())
    print("Fitness của cá nhân tiến hóa:", optCNN.fitness)
    print("-------------------------------------\n")

    plot_training(history)


if __name__ == '__main__':
    main()
