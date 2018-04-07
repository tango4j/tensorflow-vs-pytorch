# **01 Tensor**

TensorFlow and PyTorch both are based on the concept "Tensor". 

However, the
term "Variable" in each framework is used in different way.

**TensorFlow**

IF
you want to declare mutable variable (weight and bias): use **tf.Variable**  
IF
you want to declare immutable variable (a constant that will never change): use
**tf.constant**

**PyTorch**

IF you want to calculate matrix with torch
framework: use **torch.FloatTensor**  
IF you want to use autograd and get
gradient value: use **torch.autograd.Variable**

Let's get into details.

# 1. The Concept of Tensor


## **[TensorFlow]** Tensors and special type of tensors

```python
import tensorflow as tf
import numpy as np
```
**(1) What is TensorFlow "Tensor" ?**  
The concept of "tensor" in Tensorflow is very confusing for beginners.
When it says "tf.Tensor", that means "Class Tensor". In addition, there are some special type of tensors.

Unfortunately, TensorFlow's official website [Tensorflow Programmer's Guide -Tensor](https://www.tensorflow.org/programmers_guide/tensors) explains this in
very confusing way. 

Official guide from TensorFlow's website says these three
tensors are the most commonly used special type tensors:  


**The most frequently used data types in TensorFlow:**  
> **tf.Variable**  
> **tf.constant**   
> **tf.placeholder**   
> **tf.SparseTensor**


However, tf.Variable is not internally categorized as "Tensor" according to
the class structure that we can see with "type" command in python.

```python
tf_var = tf.Variable([1, 2])
tf_const = tf.constant([1, 2])
tf_ph = tf.placeholder(tf.float32, shape=(2, 2))
tf_spts = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

print('Type of tf.Variable is: ', type(tf_var))
print('Type of tf.constant is: ', type(tf_const))
print('Type of tf.placeholder is: ', type(tf_ph))
print('Type of tf.SparseTensor is: ', type(tf_spts))
```

Threrefore, the following description would be way less confusing.  

* tf.Variable creates **Variable**.   
* tf.constant creates **Tensor**.
* tf.placeholder creates **Tensor**.
* tf.SparseTensor creates **SparseTensor** (which is similar to Tensor).

**(2) Special type Tensors:**  
There are more special type tensors other than above three. For example, regular type of tensors such as:

> **tf.zeros**   
> **tf.ones**  

**(3) Convention for Tensor dimension**  
The following dimension is usually used for batch image source. Dimension index: 

> Batch x height x width x color-channel

*Example 4* -dimensional tensor

```python
my_image = tf.zeros([10, 512, 256, 3])
print('The shape of my_image:', my_image.shape)
print('The type of my_image:', type(my_image))
```

These are just regular tensor and you can define special type tensor with
regular ones.

```python
tf_var0 = tf.Variable(tf.zeros((2,2)))
print('The shape of tf_var0:', tf_var0.shape)
```
In this case, type is: *tensorflow.python.ops.variables.Variable* 
**(4) Numpy to tf.Variable**   
Or, you can directly convert numpy into tf.Varialbe (which is tf.Tensor)

```python
tf_var1 = tf.Variable(np.zeros((2,2)))
print('The shape of tf_var1:', tf_var1.shape)
```
Also in this case, just as in (3), data type is: *tensorflow.python.ops.variables.Variable*. 

**(5) Direct declaration**   

If we want to directly declare tensorflow Tensor, we have to specify 

*  shape
*  datatype

to declare a special type tensors. (which is tf.Variable, tf.constant, tf.placeholder... etc)

```python
tf_tensor_ones = tf.ones([3, 4, 4], dtype=tf.float64)
print('Directly declare a Tensor: ', tf_tensor_ones)
```
Since ones and zeros are Tensors, in this case, data type is: *tensorflow.python.framework.ops.Tensor*.

--------- 
Okay. Now that we know what are the most commonly used TensorFlow Variables.  

Note that this is very different from PyTorch, since PyTorch does not have concept of placeholder or constant.  

Therefore, if you want to get the hang of Tenforflow you should know what are the differences between these variables, and their use cases.  

Let's find out.

### Difference between special tensors and tf.Variable (TensorFlow)
**(1)tf.Variable **  

- tf.Variable is **NOT** actually tensor, but rather it
should be classified as **Variable** to avoid confusion.
- tf.Variable is the
only type that can be modified.
- tf.Variable is designed for weights and bias(≠
tf.placeholder). Not for feeding data.
- tf.Variable is stored separately, and
may live on a parameter server, **not in the graph**. 
- tf.Variable should
always be initialized before run.
- Usually declared by [initial value],
[dtype], [name]. (There are more arguments...)

```python
mymat = tf.Variable([[7],[11]], tf.int16, name='cat') 
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
```

**(2)tf.constant**  

- tf.constant holds values that cannot be changed.
-
tf.constant is also designed for weights and bias, but fixed value. 
-
tf.constant value is stored in the graph and its value is replicated wherever
the graph is loaded.
- tf.constant does not need to be initialized. (≠
tf.Variable)

```python
const_tensor = tf.constant([[7],[11]], tf.int16, name='cat') 
```

**(3)tf.placeholder**  

- tf.placeholder is designed to store values to be fed,
such as images.
- tf.placeholder will produce an error if evaluated. Its value
must be fed using the feed_dict optional argument to Session.run(),
Tensor.eval(), or Operation.run().
- tf.placeholder is usually declared with
[dtype], [data shape]

```python
placeholder_tensor = tf.placeholder(tf.float32, shape=(2, 2))
```

## **[PyTorch]** Torch tensor and torch.Variable

```python
import torch
```

Unlike Tensorflow, the tensor command itself determines the data type.  
Then,
we feed the python list.

> torch.cuda.FloatTensor 
> torch.cuda.DoubleTensor
>
torch.cuda.HalfTensor
> torch.cuda.ByteTensor 
> torch.cuda.CharTensor 
>
torch.cuda.ShortTensor
> torch.cuda.IntTensor 
> torch.cuda.LongTensor

If you
want to use CPU, you can remove cuda. Ex: torch.FloatTensor

```python
cpu_tensor = torch.FloatTensor([[1,2], [3,4]])
gpu_tensor = torch.cuda.FloatTensor([[1, 2], [3, 4]]) 
print('CPU tensor:', cpu_tensor)
print('GPU tensor:', gpu_tensor)
```

Unlike tf.Variable, PyTorch's Variable functions differently. Since PyTorch is
based on "Autograd" which enables Define-by-Run type of programming with
PyTorch. We will deal with this again later.

```python
x_np = np.array([1, 2])
x = torch.autograd.Variable(torch.from_numpy(x_np).type(torch.FloatTensor), requires_grad=True)
print('Torch Variable x: ', x)

```

We will study about torch.Variable in detail. In this chapter, make sure to
understand the difference between torch.Tensor and torch.Variable. The following

```python
print('Type of torch.Varialbe: ', type(x))
print('Type of torch.Tensor: ', type(cpu_tensor))
```

Because of the forementioned reason, PyTorch's Variable contains three different
entities as below

**torch.autograd.Variable**
> **data**: Raw data Variable contains inside the
variable.

> **grad**: Gradient obtained from Autograd feature in PyTorch.

>
**creator**: Variable remembers how the variable is created and what operation
it has gone through. 
(*Creator does not exists as a real variable in the
torch.autograd.Variable.)

Unlike TensorFlow, PyTorch Variable contains the history of the Variable itself
to enable Autograd feature. When the a variable is declared, .grad and .grad_fn
contain None.

```python
print('x.data:', x.data)
print('x.grad:', x.grad)
print('x.grad:', x.grad_fn)
```

However, if the Variables go through some mathmatical operation and we use
.backward() function to use Autograd feature.  
Then we can see what is inside
the variables .data, .grad and .grad_fn. ".grad_fn" variable contains the
gradiemt function that has automatically assigned to the operation.


We will
discuss about this in detail later. Here, make sure you understand
torch.autograd.Variable contains the following variables.

```python
y = x * 2
z = y.mean()
print('y, z contains :', y, '\n', z)
z.backward()
print('After backward() x.data:', x.data)
print('After backward() x.grad:', x.grad)
```

```python
print('x.grad_fn:', x.grad_fn)
print('y.grad_fn:', y.grad_fn)
print('z.grad_fn:', z.grad_fn)
```

# 2. Tensor-Numpy Conversion
## **[TensorFlow]** tf.convert_to_tensor or .eval()

### Numpy to tf.Tensor

Numpy automatically sets the datatype as float64. 

However, TensorFlow uses
float32 as a default. 
To convert a tf.Tensor from numpy array, use 

>
**tf.convert_to_tensor()**

function as below.

```python
numpy_array = np.asarray([[1, 2, 3], [4, 5, 6]], np.float32)
tensorflow_tensor = tf.convert_to_tensor(numpy_array, np.float32)
```

### tf.Tensor to Numpy

If you want to convert tf.Tensor to numpy array, you can evaluate the tensorflow
tensor.

```python
tf.InteractiveSession()
evaluated_tensor = tensorflow_tensor.eval()
print('The Source Numpy Array:\n', numpy_array)
print('Tensorflow Tensor Value:\n', tensorflow_tensor)
print('Tensorflow Tensor Type:\n', type(tensorflow_tensor))
print('Evaluated Tensor:\n', evaluated_tensor)
```

Or you can create a sesssion rather than using interactive session.

```python
tensorflow_tensor = tf.convert_to_tensor(numpy_array, np.float32)
sess = tf.Session()
with sess.as_default():
    print('The original tensor before conversion: ', type(tensorflow_tensor))
    print('The type of converted tensor: ', type(tensorflow_tensor.eval()))
```

## **[PyTorch]** .numpy() or torch.from_numpy()

### Numpy to torch.Tensor

In PyTorch, the conversion is much simpler. Use 

> **torch.from_numpy() **
function to get a torch.Tensor.

```python
numpy_array = np.asarray([[1, 2, 3], [4, 5, 6]])
torch_for_numpy = torch.from_numpy(numpy_array)
print('The Source Numpy Array:\n', numpy_array)
print('Torch Tensor Value:\n', torch_for_numpy)
print('Torch Tensor Type:\n', type(numpy_array))

```

### torch.Tensor to Numpy

From torch.Tensor to numpy is even simpler than other way around.

```python
numpy_again = torch_for_numpy.numpy()
print('The value of numpy_again: \n', numpy_again)
print('The type of numpy_again: \n', type(numpy_again))
```

As above, we've got numpy array again.

<Caveat!> This is only the case for torch.Tensor. If you want to convert
torch.Variable to numpy, you need to use Variable.data.numpy(). And if it is
cuda() variable, you need to use cpu(). We will cover this later.

# 3. Indentifying The Dimension

## **[TensorFlow]** .shape or tf.rank() followed by .eval()

The dimension is called "Rank" in Tensorflow. However, like numpy .shape in-
place variable, tensorflow also supports .shape variable.

```python
my_image = tf.zeros([10, 512, 256, 3])
shape = my_image.shape
print('Dimension(shape) of my_image: ', shape)
```

However, after graph run, the variable "rank" will hold the rank value. To
obtain "rank" value from the tensorflow variable, we should evaluate it through
session.

```python
rank_var =tf.rank(my_image)
tf.InteractiveSession()
rank_var_evaluated = rank_var.eval()
print("Printing 'rank_var' does not show rank of the variable.")
print('rank_var:', rank_var) 
print('\nBut after graph run (evaluation) it obtains rank value.')
print('Evaluated rank_var :', rank_var_evaluated)
```

## **[PyTorch]** .shape or .size()

As in the tensorflow example,

```python
numpy_array = np.asarray([[1, 2, 3], [4, 5, 6]])
torch_for_numpy = torch.from_numpy(numpy_array)
print('If you print it, it shows the dimension - The value of torch_for numpy: \n', torch_for_numpy)

```

But sometimes we want to get a numerical output. Thus:

```python
print('The dim of torch_for_numpy: \n', list(torch_for_numpy.size()))
print('Get the numbers directly as a python list!: \n', torch_for_numpy.size()[0], ' and ', torch_for_numpy.size()[1])

```

In addition, in PyTorch you could also use the inplace variable ".shape" to view
the dimension of Tensor, just like in TensorFlow.

```python
the_size = torch_for_numpy.shape
print('What is in the shape?:', list(torch_for_numpy.shape))
```

# 4. Shaping the Tensor Variables

## **[TensorFlow]** tf.reshape

Sometimes you need to reshape the tensor (e.g. Convolutional layer to full
connected layer).

```python
tf_tensor0 = tf.ones([2, 3, 4])  # 3D Tensor variable filled with ones.
print('The shape of tf_tensor0: %s' %tf_tensor0.shape)
```

Then we feed the number 12 = 3 × 4 to the shape vector to get reshaped Tensor.

```python
tf_tensor1 = tf.reshape(tensor0, [2, 12])  # Make "tensor0" matrix into 2 by 12 matrix.
print('The shape of tf_tensor1: %s' %tf_tensor1.shape)
```

Furthermore, reshape function supports automatic dimension shaping. Instead of
punching in the dimensions, you could put '-1' and let the function figure out
the rest of dimension.

```python
tf_tensor2 = tf.reshape(tensor0, [2, -1])  # Make "tensor1" matrix into 4 by 6 matrix.
print('The shape of tf_tensor2: %s' %tf_tensor2.shape)
```

Or you can specitfy multiple dimension and then set it '-1' to handle rest of
the dimension.

```python
tf_tensor3 = tf.reshape(tensor0, [2, 2, -1])  # Make "tensor2" matrix into 4 by 6 matrix.
print('The shape of tf_tensor3: %s' %tf_tensor3.shape)
```

## **[PyTorch]** .view() function

PyTorch provides with .view() function for reshaping the Tensor.

```python
torch_tensor0 = torch.ones([2, 3, 4])
print('The shape of torch_tensor0:', torch_tensor0.shape)
```

The usage of .view() is very analgous to tf.reshape() in TensorFlow.

```python
torch_tensor1 = torch_tensor0.view(2, -1)
print('The shape of torch_tensor1:', torch_tensor1.shape)
```

Also, view_as() function copies the shape of the input Tensor and shape the
corresponding Tensor.

```python
torch_tensor2 = torch_tensor0.view_as(torch_tensor1)
print('The shape of torch_tensor2:', torch_tensor2.shape)
```

# 5. Datatype Conversion

# 6. Printing Variables
