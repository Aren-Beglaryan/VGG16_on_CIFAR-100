# VGG16_on_CIFAR-100
Implementing VGG16 model on Cifar-100 dataset with PyTorch


### Project structure
```
-cifar
    -model
        -device.py  //definition of torch device//
        -vgg.py   //definition of vgg model//
    
    -resources //data folder //
        -meta
        -test
        -train
    
    -dataset.py   //definition of custom torch dataset//
    -loaders.py   //definition of train,test data loaders//
    -utils.py   //utilities//
-main.py   //implementation of the project//

```

To run the project's implementation do the following steps.

```
We highly recommend to use virtual environments like `conda` or `python venv`
```

1. git clone the project
2. download the data from https://www.cs.toronto.edu/~kriz/cifar.html and put the `train`, `test` and `meta` as mentioned in the structure
3. install the requirements
4. run main.py with or without your modifications.
