
import os

import numpy as np

from cifar.utils import unpickle, reshape_data

here = os.path.dirname(__file__)



cifar_train = os.path.join(here, "train")
cifar_test = os.path.join(here, "test")
cifar_meta = os.path.join(here, "meta")


meta = unpickle(cifar_meta)

class_names = [each.decode("utf-8") for each in meta[b'fine_label_names']]

_train_data_dict = unpickle(cifar_train)
_test_data_dict = unpickle(cifar_test)

train_data = reshape_data(_train_data_dict)
test_data = reshape_data(_test_data_dict)

train_label = np.array(_train_data_dict[b'fine_labels'])
test_label = np.array(_test_data_dict[b'fine_labels'])