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


