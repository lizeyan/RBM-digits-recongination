# RBM-digits-recongination 

Use Restricted Boltzmann Machine to recognize hand-written digits.

RBM's implementation is based on [echen's code](https://github.com/echen/restricted-boltzmann-machines).

*Note that I place all origin train and test data in this repo.*

## Run

``` bash
python3 run.py
```

## Change log
### 2017-6-3
- RBM(1024, 0.001, 10)->SVC(8000), 0.71
- RBM(1024, 0.001, 10)->Logistic(8000), 0.76
### 2017-5-31
- 直接用random forest就可以基本上稳稳地达到100%了，数据集好像太弱了吧
- 发现是给的数据集中，测试集的数据被包含在训练集里了。已经通过脚本剔除掉了训练集中的测试数据
