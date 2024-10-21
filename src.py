# cdir into script's path (to have a controllable relative pathing)
from collections.abc import Iterable
import os
from os.path import abspath, dirname
from typing import ByteString, Dict, List, Union

from numpy._typing import NDArray

os.chdir(dirname(abspath(__file__)))

# тека із усіма батчами даних
DATASET_FOLDER = "./cifar-10-batches-py"
# імена файлів-тренувальних батчів
DATASET_BATCHES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]
# ім'я тестового файлу для остаточної перевірки
DATASET_TEST_BATCH = "test_batch"
CATEGORIES = 10
# кількість сусідів, яку слід використовувати
K = 5

import numpy as np

_load_batch_cahe = dict()


def load_batch(batch_name: str) -> Dict[ByteString, Union[List[int], List[List[int]]]]:
    """
    Завантажує масив даних за потреби. Кешує результат
    """
    res = _load_batch_cahe.get(batch_name)
    if res is None:
        import pickle

        with open(f"{DATASET_FOLDER}/{batch_name}", "rb") as f:
            res = pickle.load(f, encoding="bytes")
            _load_batch_cahe[batch_name] = res
    return res


def metric(x: Iterable[int], y: Iterable[int]) -> float:
    return float(sum(map(lambda t: abs(int(t[0]) - int(t[1])), zip(x, y))))


class KNearestNeighbor(object):

    data: List[List[int]]
    labels: List[str]

    def __init__(self):
        """
        Метод потрiбний для створення об’єкту класу
        """
        self.data = list()
        self.labels = list()

    def train(self, X, y):
        """
        Запам’ятовує» тренувальнi зображення X та їх мiтки y
        """
        self.data.extend(X)
        self.labels.extend(y)

    def predict(self, X, k=1) -> List[int]:
        """
        Повертає мiтки для усiх зображень X за мiтками k найближчих сусiдiв
        """
        import heapq

        # для пошуку найближчих сусідів використовуватимемо бінарну кучу
        closest = list()
        heapq.heapify(closest)
        # масив для результатів
        result = list()
        # розраховані відстані до усіх тренувальних зображень
        distances = self.compute_distances_one_loop(X)

        for dists in distances:
            closest.clear()  # прибираємо можливо-присутні елементи
            for d, l in zip(dists[:k], self.labels[:k]):
                # згідно з документацією heapq, на неї можна пушити тупли --
                # в такому разі сортування відбувається за першим елементом тупла.
                # першим елементом кучі є найменший елемент. оскільки метою є
                # прибирання "найдальших" забражень, відстань до зображення можна
                # помножити на -1 (таким чином першим елементом завжди буде найбільше зображення)
                heapq.heappush(closest, (-d, l))
            for d, l in zip(dists[k:], self.labels[k:]):
                # решта зображень додаються до кучі, після чого з неї знімається найдальше зображення
                heapq.heappushpop(closest, (-d, l))

            # підраховуємо кількості сусідів, що вказують на кожну категорію
            freqs = [0] * CATEGORIES
            for _i in range(k):
                c = heapq.heappop(closest)
                freqs[c[1]] += 1
            # як результат, записуємо категорію із найбільшою кількістю сусідів
            result.append(max(freqs))
        return result

    def compute_distances_two_loops(self, X: List[List[int]]) -> List[List[float]]:
        """
        Обчислює вiдстанi мiж кожним тестовим зображення в X та кожним тренувальним зображенням, який класифiкатор «запам’ятав», використовуючи вкладений цикл над навчальними та тестовими даними. Обчислення з допомогою двох циклiв – неефективне рiшення.
        """
        result = list()
        for m in self.data:
            line = list()
            for x in X:
                line.append(metric(m, x))
            result.append(line)
        return result

    def compute_distances_one_loop(self, X: List[List[int]]) -> List[List[float]]:
        """
        Обчислює вiдстанi мiж кожною тестовим зображення в X та кожним тренувальним зображенням, який класифiкатор «запам’ятав», використовуючи вкладений лише один цикл тестовими даними.
        """
        result = list()
        for x in X:
            result.append(
                np.float64(
                    np.sum(
                        np.absolute(np.subtract(self.data, x)),
                        axis=0,
                    )
                )
            )
        return result

    def compute_distances_no_loops(self, X: List[List[int]]) -> List[List[float]]:
        """
        Вiдстанi обчислюються без явного використання циклiв.
        """
        return np.sum(np.absolute(np.subtract(self.data, X)), axis=2)


# виконуємо крос-валідацію методом k-згортки
validation_results = list()
for validation_batch in range(len(DATASET_BATCHES)):
    # створюємо нову модель
    model = KNearestNeighbor()
    # використовуаємо усі батчі окрім одного для "тренування"
    for training_batch in range(len(DATASET_BATCHES)):
        if training_batch == validation_batch:
            continue
        batch = load_batch(DATASET_BATCHES[training_batch])
        images = batch[b"data"]
        labels = batch[b"labels"]
        model.train(images, labels)
    # завантажуємо останній батч та проводимо крос-валідацію
    batch = load_batch(DATASET_BATCHES[validation_batch])
    images = batch[b"data"]
    labels = batch[b"labels"]
    predicted_labels = model.predict(images, K)
    total_images = len(labels)
    correct_images = 0
    for expected, actual in zip(labels, predicted_labels):
        if expected == actual:
            correct_images += 1
    # save the result
    validation_results.append((correct_images, total_images))

# print the results
correct_images = 0
total_images = 0
for k, res in enumerate(validation_results):
    correct_images += res[0]
    total_images += res[1]
    print(
        f"For batch {k} there were {res[0]}/{res[1]} ({float(res[0])/float(res[1])*100}%) correct predictions"
    )
print(
    f"{correct_images}/{total_images} ({float(correct_images)/float(total_images)*100}%) overall\n"
)
