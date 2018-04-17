# 02 Variables

In the last session, we learned about the difference between tf.Variables and
tf.Tensor. In this chapter, we are going to review in-depth use of tf.Variable
and torch.autograd.Variable.


# 1. Creating a Variable
## **[TensorFlow]**
tf.get_variable() function or tf.Variable
Unlike tf.Tensor objects, a
tf.Variable exists outside the context of a single session.run call.

And
Variable in TensorFlow is one of the most important concept that you should
always well aware of to build your own networks.

```python
import tensorflow as tf
import numpy as np
```

### **Method 1:** tf.get_variable()  

To get the Tensorflow Variable, we can
use get_variable function. There are many arguments for get_variable function
but usually [name], [dtype], [initializer]. Without dtype, it automatically sets
the dtype as "tf.float32". (This is different from python3 numpy, which uses
float64 as a default dtype)

```python
tf_variable = tf.get_variable('tensorflow_variable', [1, 2, 3])
tf_variable_int = tf.get_variable('tensorflow_int_var', [1, 2, 3], dtype=tf.int32)
tf_variable_intialized = tf.get_variable('tensorflow_var_init', [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
    
```

Since Tensorflow variable has "name" propoerty, if you declare a Tensorflow
variable with same name, you get an error.

However, tf.Variable can be initialized with tf.constant, which is tf.Tensor
object.

```python
tf_variable_constintialized = tf.get_variable('tensorflow_var_init_const', dtype=tf.int32, initializer=tf.constant([1,2]))
print('tf_variable has the type of :', type(tf_variable_constintialized), 'and the shape of :', tf_variable_constintialized.shape)
```

### **Method 2:** tf.Variable

We can also use tf.Variable() to create
TensorFlow variables. This method is similar to pytorch.

```python
tf_weights = tf.Variable(tf.random_normal([256, 100], stddev=0.35), name="tf_weights")
tf_biases = tf.Variable(tf.zeros([100]), name="tf_biases")
weprint('tf_biases has type of :', type(tf_biases), 'and shape of', tf_biases.shape)
```

## **[PyTorch]** Creating PyTorch Variable - torch.autograd.Variable

```python
import torch
```

As we learned from the last chapter, Pytorch is based on the concept of
"Variable".

```python
x_np = np.array([1, 2])
x = torch.autograd.Variable(torch.from_numpy(x_np).type(torch.FloatTensor), requires_grad=True)
print('Torch Variable x: ', x)
```

However, most of the time, Pytorch users use the following convention for the
torch.autograd.Variable()

```python
# I recommand you to use this convention so that you can read other people's codes.
from torch.autograd import Variable
x = Variable(torch.randn(3, 2).type(torch.FloatTensor), requires_grad=False)
print('Torch Variable x: ', x)
```

### The concept of Pytorch Variable
Unlike tf.Variable, torch.Variable needs a
lot more explanations. 

PyTorch's Variable contains three different entities as
below

**torch.autograd.Variable**
> **data**: Raw data Variable contains inside
the variable.

> **grad**: Gradient obtained from Autograd feature in PyTorch.
> **creator**: Variable remembers how the variable is created and what operation
it has gone through. 
(Creator does not exists as a real variable in the
torch.autograd.Variable.)


Unlike TensorFlow, PyTorch Variable contains the
history of the Variable itself to enable Autograd feature. When the a variable
is declared, .grad and .grad_fn contain None.

```python
print('x.data:', x.data)
print('x.grad:', x.grad)
print('x.grad:', x.grad_fn)
```

However, if the Variables go through some mathmatical operation and we use
.backward() function to use Autograd feature, we can see what is inside the
variables .data, .grad and .grad_fn. 

".grad_fn" variable contains the gradiemt
function that has automatically assigned to the operation. 

We will discuss
about this in detail in computation chapter. Here, make sure you understand
torch.autograd.Variable contains the following variables.

```python
x = Variable(torch.randn(3, 2).type(torch.FloatTensor), requires_grad=True)
y = x * 2
z = y.mean()
print('y, z contains :', y, '\n', z)
z.backward()
print('After backward() x.data:', x.data)
print('After backward() x.grad:', x.grad)
```

Also, after excuting .backward() function, .grad_fn variables are assigned with
gradient function.

```python
print('x.grad_fn:', x.grad_fn)
print('y.grad_fn:', y.grad_fn)
print('z.grad_fn:', z.grad_fn)
```

Here, x is not assigned with grad_fn because we started the operation from x.
