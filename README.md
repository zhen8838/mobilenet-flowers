# mobilenet-flowers
利用Google预定义的mobilenet v1网络构建一个花朵分类器(K210预备)

# 使用

##### 1.    首先需要安装`Tensorflow`的`Models/reserach/slim`模块
下载`models`模块,在`bashrc`中写入环境变量.
```sh
git clone https://github.com/tensorflow/models/
vi ~/.bashrc
```
添加如下(**需要修改为你所对应的地址**):
```sh
export PYTHONPATH="~/models/research/slim:$PYTHONPATH"
```

##### 2.    下载本项目

```sh
git clone git@github.com:zhen8838/mobilenet-flowers.git
```

##### 3.    下载数据集以及预处理
我使用的是`slim`教程中所用的`Flowers`数据集.首先下载解压数据集:
```sh
wget http://download.tensorflow.org/example_images/flower_photos.tgz
tar -zxvf flower_photos.tgz
```
接下来修改代码进行预处理:
```sh
cd mobilenet-flowers
vi Globals.py
```
修改`Globals.py`中第4行的`DATA_DIR`(**必须要用绝对路径**)
```python
import os


DATA_DIR = '/media/zqh/Datas/DataSet/flower_photos' #修改为你对应的路径
CLASS_NUM = 5
SEED = 3
```
保存退出后执行:
```sh
python3 prepare_data.py
```
执行结束后检查目录,出现以下即完成:
```sh
ls /home/zqh/flower_photos/*.csv
/home/zqh/flower_photos/Label.csv  /home/zqh/flower_photos/train.csv
/home/zqh/flower_photos/test.csv
```
    
##### 4.  测试数据集
执行
```sh
python3 tf_test.py
```
这时候可能会出现准确率非常低的情况,应该是由于我制作数据集的过程中`shuffle`的缘故.没有关系,我们可以自己重新训练.
如果需要加载自行训练的权重,那么需要修改`tf_test.py`文件第20行`RESTORE_CKPT_PATH`为新的权重文件夹:
```python
if __name__ == "__main__":
tf.reset_default_graph()
# =========== define the ckpt path===================
# NOTE modfiy to your path
RESTORE_CKPT_PATH = 'log/train/save_18:40:14'
TEST_IMG_NUM = 100
# ===================== end =============================
```


##### 5.  训练数据集
执行:
```sh
python3 tf_train.py
```    
结束后查看保存的模型文件,其中带`save`的文件为模型文件(**`save_18:40:14`为默认使用权重文件**):
```sh
ls log/train/
0.50_18:40:14  0.50_19:14:35  save_18:40:14  save_19:14:35
```

# TODO

##### 1.    转化模型
##### 2.    移植到K210