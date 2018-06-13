###初识numpy
#### 简介
NumPy系统是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多。
#### 从一张图片认识numpy
```
import numpy as np
import matplotlib.pyplot as plt  # matplotlib用于画图，也可以读取图片
```
```
cat = plt.imread('cat.jpg')  # 读取本地图片
```
```
display(cat, cat.shape) 
```
```
cat.shape  # 查看照片形状
```
```
plt.imshow(cat)  # 显示照片
```
#### 生成一张随机图片
```
im = np.random.randint(0, 255, size=(456, 730, 3))
im = im.astype(np.float64)
plt.imshow(im)
```
![这里写图片描述](https://img-blog.csdn.net/20180612203203478?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc4MjA1MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### ndarray
numpy中最重要的一个形式叫ndarray n 表示的是n个 d dimension 维度 array 数组。
ndarray类具有6个参数：
* shape：数组的形状
* dtype：数据类型
* buffer：对象暴露缓冲区接口
* offset：数组数据的偏移量
* strides：数据步长
创建ndarray的5种方法：
1. 从 Python 数组结构列表，元组等转换
2. 使用 np.arange、np.ones、np.zeros 等 numpy 原生方法
3. 存储空间读取数组
4. 通过使用字符串或缓冲区从原始字节创建数组
5. 使用特殊函数，如 random

#### 从列表或元组转换创建ndarray
```
l = list('123456')
np.array(l)
```
#### arange方法创建ndarray
```
nd = np.arange(0, 150, step=5, dtype=np.float32)
```
#### linespace方法创建ndarray
linspace方法也可以像arange方法一样，创建数值有规律的数组。linspace用于在指定的区间内返回间隔均匀的值.
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
1. start：序列的起始值。
2. stop：序列的结束值。
3. num：生成的样本数。默认值为50。
4. endpoint：布尔值，如果为真，则最后一个样本包含在序列内。
5. retstep：布尔值，如果为真，返回间距。
6. dtype：数组的类型。

```
nd = np.linspace(0,150,num=151)
```
#### noes方法创建ndarray
```
nd = np.ones(shape=(456, 730, 3))
```
#### zeros方法创建ndarray
zeros 方法和上面的 ones 方法非常相似，不同的地方在于，这里全部填充为 0。zeros 方法和 ones 是一致的。
numpy.zeros(shape, dtype=None, order='C')
```
nd = np.zeros(shape=(456, 730, 3))
```

#### full方法创建ndarray
numpy.full用于创建一个自定义形状的数组，可以自己指定一个值，该值填满整个矩阵。
numpy.full(shape,fill_value=num)
```
nd = np.full(shap, fill_value=125)
```

#### eye方法创建ndarray
numpy.eye 用于创建一个二维数组，其特点是k 对角线上的值为 1，其余值全部为0。
numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
```
nd = np.eye(5, 5)
nd
```
![这里写图片描述](https://img-blog.csdn.net/20180612210110479?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc4MjA1MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

