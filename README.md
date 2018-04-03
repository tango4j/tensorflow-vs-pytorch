# tensorflow-vs-pytorch

This repository aims for comparative analysis of TensorFlow vs PyTorch, for those who want to learn TensorFlow while already familiar with PyTorch or vice versa.

[**01. Tensor**](https://github.com/tango4j/tensorflow-vs-pytorch/blob/master/01_tensor.ipynb)  

[1.The Concept of Tensor](https://render.githubusercontent.com/view/ipynb?commit=59e8df5d44e2a7578e78024f1313985ac81d4d30&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f74616e676f346a2f74656e736f72666c6f772d76732d7079746f7263682f353965386466356434346532613735373865373830323466313331333938356163383164346433302f30315f74656e736f722e6970796e62&nwo=tango4j%2Ftensorflow-vs-pytorch&path=01_tensor.ipynb&repository_id=127955188&repository_type=Repository#1.-The-Concept-of-Tensor)  
[2.Tensor Numpy Conversion](https://render.githubusercontent.com/view/ipynb?commit=59e8df5d44e2a7578e78024f1313985ac81d4d30&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f74616e676f346a2f74656e736f72666c6f772d76732d7079746f7263682f353965386466356434346532613735373865373830323466313331333938356163383164346433302f30315f74656e736f722e6970796e62&nwo=tango4j%2Ftensorflow-vs-pytorch&path=01_tensor.ipynb&repository_id=127955188&repository_type=Repository#2.-Tensor-Numpy-Conversion)  
[3.Indentifying The Dimension](https://render.githubusercontent.com/view/ipynb?commit=59e8df5d44e2a7578e78024f1313985ac81d4d30&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f74616e676f346a2f74656e736f72666c6f772d76732d7079746f7263682f353965386466356434346532613735373865373830323466313331333938356163383164346433302f30315f74656e736f722e6970796e62&nwo=tango4j%2Ftensorflow-vs-pytorch&path=01_tensor.ipynb&repository_id=127955188&repository_type=Repository#3.-Indentifying-The-Dimension)  
[4.Shaping the Tensor Variables](https://render.githubusercontent.com/view/ipynb?commit=59e8df5d44e2a7578e78024f1313985ac81d4d30&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f74616e676f346a2f74656e736f72666c6f772d76732d7079746f7263682f353965386466356434346532613735373865373830323466313331333938356163383164346433302f30315f74656e736f722e6970796e62&nwo=tango4j%2Ftensorflow-vs-pytorch&path=01_tensor.ipynb&repository_id=127955188&repository_type=Repository#4.-Shaping-the-Tensor-Variables)  

[02. Variable](https://github.com/tango4j/tensorflow-vs-pytorch/blob/master/02_variable.ipynb)

[03. Computation of data](https://github.com/tango4j/tensorflow-vs-pytorch/blob/master/03_computation_of_data.ipynb)


- There are few distinct differences between Tensorflow and Pytorch when it comes to data compuation.


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
    
   
