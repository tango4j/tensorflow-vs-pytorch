# tensorflow-vs-pytorch

A comparative study of TensorFlow vs PyTorch.

This repository aims for comparative analysis of TensorFlow vs PyTorch, for those who want to learn TensorFlow while already familiar with PyTorch or vice versa.


## Important Updates

**TensorFlow**  

[Eager Excution(Oct 17, 2018)](https://www.tensorflow.org/guide/eager) (Dynamic graph)
 Tensorflow also launches a dynamic graph framework which enables define by run.

**Pytorch**

[Pytorch 4.0 Migraction (Apr 22, 2018)](https://pytorch.org/blog/pytorch-0_4_0-migration-guide). Variable is merged into Tensor. 
Currently, torch.Variable returns torch.tensor and torch.tensor can function as old torch.Variable.


## vs. Table

|               | TensorFlow                                           | PyTorch                                            |
|---------------|------------------------------------------------------|----------------------------------------------------|
|  Numpy to tensor | [**Numpy to tf.Tensor**](https://github.com/tango4j/tensorflow-vs-pytorch#numpy-to-tftensor) <br/> tf.convert_to_tensor(numpy_array, np.float32)  | [**Numpy to torch.Tensor**](https://github.com/tango4j/tensorflow-vs-pytorch#numpy-to-torchtensor) <br/> torch.from_numpy(numpy_array) |
| Tensor to Numpy  | [**tf.Tensor to Numpy**](https://github.com/tango4j/tensorflow-vs-pytorch#tftensor-to-numpy) <br/> tensorflow_tensor.eval()  <br/>  tf.convert_to_tensor(numpy_array, np.float32)  |  [**torch.Tensor to Numpy**](https://github.com/tango4j/tensorflow-vs-pytorch#torchtensor-to-numpy) <br/> torch_for_numpy.numpy() |

## Table of Contents

[**01. Tensor**](https://github.com/tango4j/tensorflow-vs-pytorch#01-tensor)   

> [**1. The Concept of Tensor**](https://github.com/tango4j/tensorflow-vs-pytorch#01-tensor)   
>[[TensorFlow] - Tensors and special type of tensors](https://github.com/tango4j/tensorflow-vs-pytorch#tensorflow-tensors-and-special-type-of-tensors)  

>> [(1) What is TensorFlow "Tensor" ?](https://github.com/tango4j/tensorflow-vs-pytorch#1-what-is-tensorflow-tensor-)   
>> [(2) Special type Tensors](https://github.com/tango4j/tensorflow-vs-pytorch#2-special-type-tensors)   
>> [(3) Convention for Tensor dimension](https://github.com/tango4j/tensorflow-vs-pytorch#3-convention-for-tensor-dimension)     
>> [(4) Numpy to tf.Variable](https://github.com/tango4j/tensorflow-vs-pytorch#4-numpy-to-tfvariable)   
>> [(5) Direct declaration](https://github.com/tango4j/tensorflow-vs-pytorch#5-direct-declaration)   
>> [(6) Difference Between Special Tensors and tf.Variable (TensorFlow)](https://github.com/tango4j/tensorflow-vs-pytorch#difference-between-special-tensors-and-tfvariable-tensorflow)   

>[[PyTorch] - Torch tensor and torch.Variable](https://github.com/tango4j/tensorflow-vs-pytorch#pytorch-torch-tensor-and-torchvariable)   
>[Basics for PyTorch Tensors.](https://github.com/tango4j/tensorflow-vs-pytorch#basics-for-pytorch-tensors)   

>>[(1) PyTorch Tensor](https://github.com/tango4j/tensorflow-vs-pytorch#1-pytorch-tensor)   
>>[(2) PyTorch's dynamic graph feature](https://github.com/tango4j/tensorflow-vs-pytorch#2-pytorchs-dynamic-graph-feature)   
>>[(3) What does torch.autograd.Variable contain?](https://github.com/tango4j/tensorflow-vs-pytorch#3-what-does-torchautogradvariable-contain)   
>>[(4) Backpropagation with dynamic graph](https://github.com/tango4j/tensorflow-vs-pytorch#4-backpropagation-with-dynamic-graph)   

>[**2. Tensor Numpy Conversion**](https://github.com/tango4j/tensorflow-vs-pytorch#2-tensor-numpy-conversion)   

>[[TensorFlow] tf.convert_to_tensor or .eval()](https://github.com/tango4j/tensorflow-vs-pytorch#tensorflow-tfconvert_to_tensor-or-eval)  
>> [Numpy to tf.Tensor](https://github.com/tango4j/tensorflow-vs-pytorch#numpy-to-tftensor)   
>> [tf.Tensor to Numpy](https://github.com/tango4j/tensorflow-vs-pytorch#tftensor-to-numpy) |

>[[PyTorch] .numpy() or torch.from_numpy()](https://github.com/tango4j/tensorflow-vs-pytorch#pytorch-numpy-or-torchfrom_numpy)   
>>[Numpy to torch.Tensor](https://github.com/tango4j/tensorflow-vs-pytorch#numpy-to-torchtensor)   
>>[torch.Tensor to Numpy](https://github.com/tango4j/tensorflow-vs-pytorch#torchtensor-to-numpy)    

> [**3. Indentifying The Dimension**](https://github.com/tango4j/tensorflow-vs-pytorch#3-indentifying-the-dimension)    

> [[TensorFlow] .shape or tf.rank() followed by .eval()](https://github.com/tango4j/tensorflow-vs-pytorch#tensorflow-shape-or-tfrank-followed-by-eval)   
>> [.shape variable in TensorFlow](https://github.com/tango4j/tensorflow-vs-pytorch#shape-variable-in-tensorflow)   
>> [tf.rank function](https://github.com/tango4j/tensorflow-vs-pytorch#tfrank-function)   

>[[PyTorch] .shape or .size()](https://github.com/tango4j/tensorflow-vs-pytorch#pytorch-shape-or-size)   
>>[Automatically Displayed PyTorch Tensor Dimension](https://github.com/tango4j/tensorflow-vs-pytorch#automatically-displayed-pytorch-tensor-dimension)   
>>[.shape variable in PyTorch](https://github.com/tango4j/tensorflow-vs-pytorch#shape-variable-in-pytorch)   

>[**4. Shaping the Tensor Variables**](https://github.com/tango4j/tensorflow-vs-pytorch#4-shaping-the-tensor-variables)   

>[[TensorFlow] tf.reshape](https://github.com/tango4j/tensorflow-vs-pytorch#tensorflow-tfreshape)   
>>[Reshape tf.Tensor with tf.reshape](https://github.com/tango4j/tensorflow-vs-pytorch#reshape-tftensor-with-tfreshape)   
>>[Handling the Rest of Dimension with "-1"](https://github.com/tango4j/tensorflow-vs-pytorch#handling-the-rest-of-dimension-with--1-1)   

>[[PyTorch].view() function](https://github.com/tango4j/tensorflow-vs-pytorch#pytorch-view-function)   
>> [Reshape PyTorch Tensor with .view()](https://github.com/tango4j/tensorflow-vs-pytorch#reshape-pytorch-tensor-with-view)   
>> [Handling the Rest of Dimension with "-1"](https://github.com/tango4j/tensorflow-vs-pytorch#handling-the-rest-of-dimension-with--1-1)   
>> [Copy the Dimension of other PyTorch Tensor .view_as()](https://github.com/tango4j/tensorflow-vs-pytorch#copy-the-dimension-of-other-pytorch-tensor-view_as)   

> [**5. Shaping the Tensor Variables**](https://github.com/tango4j/tensorflow-vs-pytorch#4-shaping-the-tensor-variables)   

> [**6. Datatype Conversion**](https://github.com/tango4j/tensorflow-vs-pytorch#5-datatype-conversion)    

> [**7. Printing Variables**](https://github.com/tango4j/tensorflow-vs-pytorch#6-printing-variables)   

[**02. Variable**](https://github.com/tango4j/tensorflow-vs-pytorch#02-variables-)   
>[**1. Creating a Variable**](https://github.com/tango4j/tensorflow-vs-pytorch#1-creating-a-variable)   
>[[TensorFlow]](https://github.com/tango4j/tensorflow-vs-pytorch#tensorflow)  
>>[Method 1: tf.get_variable()](https://github.com/tango4j/tensorflow-vs-pytorch#method-1-tfget_variable)  
>>[Method 2: tf.Variable](https://github.com/tango4j/tensorflow-vs-pytorch#method-2-tfvariable)   

>[[PyTorch] Creating PyTorch Variable - torch.autograd.Variable](https://github.com/tango4j/tensorflow-vs-pytorch#pytorch-creating-pytorch-variable---torchautogradvariable)   
>>[The concept of Pytorch Variable](https://github.com/tango4j/tensorflow-vs-pytorch#the-concept-of-pytorch-variable)   

[**03. Computation of data**](https://github.com/tango4j/tensorflow-vs-pytorch#03-computaion-of-data)   
>[1. Tensorflow VS PyTorch Comparison](https://github.com/tango4j/tensorflow-vs-pytorch#1-tensorflow-vs-pytorch-comparison)   
>[2. Dynamic Graph and Static Graph](https://github.com/tango4j/tensorflow-vs-pytorch#1-dynamic-graph-and-static-graph)   

- There are a few distinct differences between Tensorflow and Pytorch when it comes to data compuation.

|               | TensorFlow      | PyTorch        |
|---------------|-----------------|----------------|
| Framework     | Define-and-run  | Define-by-run  |
| Graph         | Static          | Dynamic        |
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


- There are a few distinct differences between Tensorflow and Pytorch when it comes to data compuation.

