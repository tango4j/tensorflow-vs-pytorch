# tensorflow-vs-pytorch

This repository aims for comparative analysis of TensorFlow vs PyTorch, for those who want to learn TensorFlow while already familiar with PyTorch or vice versa.

The whole content was written in Ipython Notebook then converted into MarkDown. Ipython Notebooks in main directory cotains the same content.

### TABLE OF CONTENTS

[**01. Tensor**](https://github.com/tango4j/tensorflow-vs-pytorch#01-tensor)  

[1. The Concept of Tensor](https://github.com/tango4j/tensorflow-vs-pytorch#01-tensor)  
[2. Tensor Numpy Conversion](https://github.com/tango4j/tensorflow-vs-pytorch#2-tensor-numpy-conversion)  
[3. Indentifying The Dimension](https://github.com/tango4j/tensorflow-vs-pytorch#3-indentifying-the-dimension)  
[4. Shaping the Tensor Variables](https://github.com/tango4j/tensorflow-vs-pytorch#4-shaping-the-tensor-variables)  

[**02. Variable**](https://github.com/tango4j/tensorflow-vs-pytorch#02-variables-)

[1. Creating a Variable](https://github.com/tango4j/tensorflow-vs-pytorch#1-creating-a-variable)

[**03. Computation of data**](https://github.com/tango4j/tensorflow-vs-pytorch#03-computaion-of-data)

[1. Dynamic Graph and Static Graph](https://github.com/tango4j/tensorflow-vs-pytorch#1-dynamic-graph-and-static-graph)


- There are a few distinct differences between Tensorflow and Pytorch when it comes to data compuation.

|               | TensorFlow      | PyTorch        |
|---------------|-----------------|----------------|
| Framework     | Define-and-run  | Define-by-run  |
| Graph         | Static | Dynamic|
| Debug         | Non-native debugger (tfdbg) |pdb(ipdb) Python debugger|

**How "Graph" is defined in each framework?**

#**TensorFlow:** 

- Static graph.

- Once define a computational graph and excute the same graph repeatedly.

- Pros: 

    (1) Optimizes the graph upfront and makes better distributed computation.
    
    (2) Repeated computation does not cause additional computational cost.


- Cons: 

    (1) Difficult to perform different computation for each data point.
    
    (2) The structure becomes more complicated and harder to debug than dynamic graph. 


#**PyTorch:** 

- Dynamic graph.

- Does not define a graph in advance. Every forward pass makes a new computational graph.

- Pros: 

    (1) Debugging is easier than static graph.
    
    (2) Keep the whole structure concise and intuitive. 
    
    (3) For each data point and time different computation can be performed.
    
    
- Cons: 

    (1) Repetitive computation can lead to slower computation speed. 
    
    (2) Difficult to distribute the work load in the beginning of training.


# **01 Tensor**

Both TensorFlow and PyTorch  are based on the concept "Tensor". 

However, the term "Variable" in each framework is used in different way.

**TensorFlow**

If you want to declare mutable variable (weight and bias): use **tf.Variable**  
If you want to declare immutable variable (a constant that will never change): use

**tf.constant**

**PyTorch**

If you want to calculate matrix with torch framework: use **torch.FloatTensor**  
If you want to use autograd and get gradient value: use **torch.autograd.Variable**

Let's get into details.

# 1. The Concept of Tensor

## **[TensorFlow]** Tensors and special type of tensors

### Basics for TensorFlow Tensors

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

These are TensorFlow **Tensors**.

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

### Difference Between Special Tensors and tf.Variable (TensorFlow)  
**(1) tf.Variable:**   

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

**(2) tf.constant:**  

- tf.constant holds values that cannot be changed (=Immutable).
- tf.constant is also designed for weights and bias, but fixed value. 
- tf.constant value is stored in the graph and its value is replicated wherever
the graph is loaded.
- tf.constant does not need to be initialized. (≠
tf.Variable)

```python
const_tensor = tf.constant([[7],[11]], tf.int16, name='cat') 
```

**(3) tf.placeholder:**   

- tf.placeholder is designed to store values to be fed, such as images.
- tf.placeholder will produce an error if evaluated. Its value
must be fed using the feed_dict optional argument to Session.run(), Tensor.eval(), or Operation.run().
- tf.placeholder is usually declared with [dtype], [data shape]

```python
placeholder_tensor = tf.placeholder(tf.float32, shape=(2, 2))
```

## **[PyTorch]** Torch tensor and torch.Variable

```python
import torch
```
### Basics for PyTorch Tensors.

**(1) PyTorch Tensor.**    

Unlike Tensorflow, the tensor command itself determines the data type.  
Then, we feed the python list.

> torch.cuda.FloatTensor  
> torch.cuda.DoubleTensor   
> torch.cuda.HalfTensor   
> torch.cuda.ByteTensor    
> torch.cuda.CharTensor    
> torch.cuda.ShortTensor   
> torch.cuda.IntTensor    
> torch.cuda.LongTensor   

c.f.) In TensorFlow, this would be: 

```python
x = tf.constant([[1, 2, 3]], dtype=tf.float32, name="B")
y = tf.Variable([[7],[11]], tf.int16, name='cat') 
```

Usually, PyTorch Tensor is defined as below.
```python
x = torch.IntTensor(2, 4).zero_()
y = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print('PyTorch tensor x:', x)
print('PyTorch tensor y:', y)
```

If you want to use GPU, you can put cuda. before FloatTensor Ex: torch.cuda.FloatTensor

```python
cputensor = torch.FloatTensor([[1,2], [3,4]])
gputensor = torch.cuda.FloatTensor([[1, 2], [3, 4]]) 
print('CPU tensor:', cputensor)
print('GPU tensor:', gputensor)
```
**(2) PyTorch's dynamic graph feature**   

Unlike TensorFlow's tf.Variable, PyTorch's Variable functions differently. This is because PyTorch is based on "Autograd" which enables Define-by-Run type of computational graph. We will deal with this again later.

```python
x_np = np.array([1, 2])
x = torch.autograd.Variable(torch.from_numpy(x_np).type(torch.FloatTensor), requires_grad=True)
print('Torch Variable x: ', x)

```

We will study about torch.Variable in detail. In this chapter, make sure to
understand the difference between torch.Tensor and torch.Variable. The following

```python
print('Type of torch.Varialbe: ', type(x))
print('Type of torch.Tensor: ', type(cputensor))   

```

**(3) What does torch.autograd.Variable contain?**   

Because of the aforementioned reasons, PyTorch's Variable contains three different
entities as below

**torch.autograd.Variable:**
> **data**: Raw data Variable contains inside the
variable.

> **grad**: Gradient obtained from Autograd feature in PyTorch.

> **creator**: Variable remembers how the variable is created and what operation
it has gone through. (Creator does not exists as a real variable in the torch.autograd.Variable.)

Unlike TensorFlow, PyTorch Variable (not graph) contains the history of the Variable itself
to enable Autograd feature. When PyTorch Variable is declared, .grad and .grad_fn
contain None.

```python
print('x.data:', x.data)
print('x.grad:', x.grad)
print('x.grad:', x.grad_fn)
```

**(4) Backpropagation with dynamic graph**  

However, if the Variables go through some mathematical operation and we use
.backward() function to use Autograd feature.  

Then we can see what is inside the variables .data, .grad and .grad_fn. ".grad_fn" variable contains the gradient function that has automatically assigned to the operation.

We will discuss about this in detail later. Here, make sure you understand
torch.autograd.Variable contains the following variables.

```python
y = x * 2
z = y.mean()
print('y, z contains :', y, '\n', z)
z.backward()
print('After backward() x.data:', x.data)
print('After backward() x.grad:', x.grad)

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

Or you can create a session rather than using interactive session.

```python
tensorflow_tensor = tf.convert_to_tensor(numpy_array, np.float32)
sess = tf.Session()
with sess.as_default():
    print('The original tensor before conversion: ', type(tensorflow_tensor))
    print('The type of converted tensor: ', type(tensorflow_tensor.eval()))
```

## **[PyTorch]** .numpy() or torch.from_numpy()

### Numpy to torch.Tensor

In PyTorch, the conversion is much simpler. Use **torch.from_numpy()**
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

### .shape variable in TensorFlow

Like numpy .shape variable, tensorflow also supports .shape variable.

```python
my_image = tf.zeros([10, 512, 256, 3])
shape = my_image.shape
print('Dimension(shape) of my_image: ', shape)
```
### tf.rank function   

The dimension is called "Rank" in Tensorflow. To obtain "rank" value from the tensorflow variable, we should evaluate it through
session.

```python
rank_var = tf.rank(my_image)
tf.InteractiveSession()
rank_var_evaluated = rank_var.eval()
print("Printing 'rank_var' does not show rank of the variable.")
print('rank_var:', rank_var) 
print('\nBut after graph run (evaluation) it obtains rank value.')
print('Evaluated rank_var :', rank_var_evaluated)
```

However, PyTorch does not have the concept of rank_var.eval() type routine for shape checking.

## **[PyTorch]** .shape or .size()

### Automatically Displayed PyTorch Tensor Dimension 
PyTorch Tensor automatically prints the shape of tensor.

```python
numpy_array = np.asarray([[1, 2, 3], [4, 5, 6]])
torch_for_numpy = torch.from_numpy(numpy_array)
print('If you print it, it shows the dimension - The value of torch_for numpy: \n', torch_for_numpy)

```
if you print it, it shows the dimension - The value of torch_for numpy: 
 
  1  2  3   
  4  5  6
[torch.LongTensor of size 2x3]

### .shape variable in PyTorch 

But sometimes we want to get a numerical output. Thus:

```python
print('The dim of torch_for_numpy: \n', list(torch_for_numpy.size()))
print('Get the numbers directly as a python list!: \n', torch_for_numpy.size()[0], ' and ', torch_for_numpy.size()[1])

```
In addition, in PyTorch you could also use the class variable ".shape" to view
the dimension of Tensor, just like in numpy and TensorFlow.

```python
the_size = torch_for_numpy.shape
print('What is in the shape?:', list(torch_for_numpy.shape))
```

# 4. Shaping the Tensor Variables

## **[TensorFlow]** tf.reshape

### Reshape tf.Tensor with tf.reshape
Sometimes you need to reshape the tensor (e.g. Convolutional layer to full connected layer).

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

### Handling the Rest of Dimension with "-1"
Or you can specitfy multiple dimension and then set it '-1' to handle rest of
the dimension.

```python
tf_tensor3 = tf.reshape(tensor0, [2, 2, -1])  # Make "tensor2" matrix into 4 by 6 matrix.
print('The shape of tf_tensor3: %s' %tf_tensor3.shape)
```

## **[PyTorch]** .view() function

### Reshape PyTorch Tensor with .view()
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
### Copy the Dimension of other PyTorch Tensor .view_as()
Also, view_as() function copies the shape of the input Tensor and shape the
corresponding Tensor.

```python
torch_tensor2 = torch_tensor0.view_as(torch_tensor1)
print('The shape of torch_tensor2:', torch_tensor2.shape)
```

# 5. Datatype Conversion

# 6. Printing Variables   

# **02 Variables **

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

### ** - Method 1:** tf.get_variable()  

To get the Tensorflow Variable, we can
use get_variable function. There are many arguments for get_variable function
but usually [name], [dtype], [initializer]. Without dtype, it automatically sets
the dtype as "tf.float32". (This is different from python3 numpy, which uses
float64 as a default dtype)

```python
if 'tf_variable' not in locals(): # To prevent overwriting problem.
    tf_variable = tf.get_variable('tensorflow_variable', [1, 2, 3])
    tf_variable_int = tf.get_variable('tensorflow_int_var', [1, 2, 3], dtype=tf.int32)
    tf_variable_intialized = tf.get_variable('tensorflow_var_init', [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
    
```

Since Tensorflow variable has "name" propoerty, if you declare a Tensorflow
variable with same name, you get an error.

However, tf.Variable can be initialized with tf.constant, which is tf.Tensor
object.

```python
if 'tf_variable_constintialized' not in locals():
    tf_variable_constintialized = tf.get_variable('tensorflow_var_init_const', dtype=tf.int32, initializer=tf.constant([1,2]))
print('tf_variable has the type of :', type(tf_variable_constintialized), 'and the shape of :', tf_variable_constintialized.shape)
```

### ** - Method 2:** tf.Variable

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
(*Creator does not exists as a real variable in the
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
# 03 **Computaion of Data**

```python
import tensorflow as tf
import torch
import numpy as np
```

There are a few distinct differences between Tensorflow and Pytorch when it
comes to data compuation.


|               | TensorFlow      | PyTorch        |
|---------------|-----------------|----------------|
| Framework     | Define-
and-run  | Define-by-run  |
| Graph         | Static | Dynamic|
| Debug
| Non-native debugger (tfdbg) |pdb(ipdb) Python debugger|

**How "Graph" is defined in each framework?**

**TensorFlow:** 

- Static graph.
- Once define a computational graph and excute the same graph repeatedly.

-
Pros: 

    (1) Optimizes the graph upfront and makes better distributed
computation.
    
    (2) Repeated computation does not cause additional
computational cost.


- Cons: 

    (1) Difficult to perform different
computation for each data point.
    
    (2) The structure becomes more
complicated and harder to debug than dynamic graph. 

**PyTorch:** 

- Dynamic
graph.

- Does not define a graph in advance. Every forward pass makes a new
computational graph.

- Pros: 

    (1) Debugging is easier than static graph.
(2) Keep the whole structure concise and intuitive. 
    
    (3) For each data
point and time different computation can be performed.
    
    
- Cons:
(1) Repetitive computation can lead to slower computation speed. 
    
    (2)
Difficult to distribute the work load in the beginning of training.

# 1. Dynamic Graph and Static Graph

## **[TensorFlow]** Graph and session

# tf.Graph: 

What is tf.Graph?

* tf.Graph should be defined before add
operations and tensors, otherwise we use default graph.

* tf.Graph is needed
whenever there are multiple models in one file.

* tf.Graph contains two
informations. 
  
    (1) **Graph Structure**: Nodes(Operations) and
Edges(Tensors) of the graph.
  
    (2) **Graph Collections**: Store all the
collections of metadata. Use tf.add_to_collection and tf.get_collection to
access thses collections. 
    
    
* If we do not specify tf.Graph, TF
automatically defines default graph which we cannot see in the code.

* Node:
**tf.Operation** - Edge: **tf.Tensor**

* Each and every tf.Operation and
tf.Tensor is added to tf.Graph instacne.

*Example 1)*

```python
tf_graph = tf.Graph()
with tf_graph.as_default():
    x = tf.constant([1, 2], shape = [1,2])
    y = tf.constant([3, 4], shape = [2,1])
    z = tf.matmul(x, y)    
```

In the above example:
    - tf.constant() is a tf.Operation that creates 42.0,
adds it to a tf.Graph and returns a tf.Tensor.
    - tf.matmul() is a
tf.Operation that calculates multiplication of x and y, adds it to a tf.Graph
and returns a Tensor.

# tf.Session: 

What is tf.Session?

* tf.Session incorporates operations and
tensors. tf.Session also excute and evaluate the operations and tensors.

*
tf.Session takes three arguments, which are all optional
  
    (1) **target**:
The excution engine to connect to.
        
    (2) **graph**: tf.Graph that
session wants to launch. If not specified, automatically links default graph.
(3) **config**: A ConfigProto protocol buffer with configuration options.
* Unlike tf.Graph, tf.Session should be placed before the operations. 

*
tf.Session.run() function excutes the given operation.

*Example 2)*

```python
with tf.Session(graph=tf_graph) as sess:
  initialize = tf.global_variables_initializer()
  sess.run(initialize)
  print(sess.run(z))
```

However, Example 2 can be excuted without specifying a graph instance.

*Example 3)*

```python
x1 = tf.constant([1, 2], shape = [1,2])
y1 = tf.constant([3, 4], shape = [2,1])
z1 = tf.matmul(x1, y1)
with tf.Session() as sess:
  initialize = tf.global_variables_initializer()
  sess.run(initialize)
  print(sess.run(z1))
```

# tf.InteractiveSession:

TensorFlow supports tf.InteractiveSession() that enables more convenient form of
session. 

Use tf.Tensor.eval() to obtain the result.

```python
sess = tf.InteractiveSession()
x2 = tf.constant([1, 2], shape = [1,2])
y2 = tf.constant([3, 4], shape = [2,1])
z2 = tf.matmul(x2, y2)
print(z2.eval())
```

## **[PyTorch]** Dynamic Graph

Unlike TensorFlow, PyTorch does not require graph instance or session instance.

*Example 4)*

```python
a = torch.from_numpy(np.asarray([[1,2]]))
b = torch.from_numpy(np.asarray([[3,4]]).T)
c = torch.matmul(a, b)
print(c)
```

Since there are no graph instance and session instance, the code is much simpler
for the same operation.

This operation can be also computed with torch.autograd.Variable. With
torch.autograd.Variable, we can compute gradients!

```python
from torch.autograd import Variable
```

```python
a1 = Variable(torch.from_numpy(np.asarray([[1,2]])), requires_grad=True)
b1 = Variable(torch.from_numpy(np.asarray([[3,4]]).T), requires_grad=True)
c1 = torch.matmul(a1, b1)
print(c1)
```

We can calculate the gradient as below.

```python
c1.backward(retain_graph=True)
print(a1.grad, b1.grad)
```
