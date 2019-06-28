# 03 **Computaion of Data**

# 1. Tensorflow VS PyTorch Comparison 
There are a few distinct differences between Tensorflow and Pytorch when it
comes to data compuation.

|               | TensorFlow      | PyTorch        |
|---------------|-----------------|----------------|
| Framework     | Define-and-run  | Define-by-run  |
| Graph         | Static          | Dynamic        |
| Debug         | Non-native debugger (tfdbg) |pdb(ipdb) Python debugger|

**How "Graph" is defined in each framework?**

### TensorFlow 

- Static graph.

- Once define a computational graph and excute the same graph repeatedly.

- Pros: 

(1) Optimizes the graph upfront and makes better distributed
computation.
    
(2) Repeated computation does not cause additional
computational cost.


- Cons: 

(1) Difficult to perform different
computation for each data point.
    
(2) The structure becomes more
complicated and harder to debug than dynamic graph. 

### PyTorch 

- Dynamic graph.

- Does not define a graph in advance. Every forward pass makes a new
computational graph.

- Pros: 
(1) Debugging is easier than static graph(Tensorflow, etc.)
(2) Keep the whole structure concise and intuitive. 
(3) For each data point and time different computation can be performed.
    
    
- Cons:
(1) Repetitive computation can lead to slower computation speed. 
(2) Difficult to distribute the work load in the beginning of training.

# 2. Dynamic Graph and Static Graph

## **[TensorFlow]** Graph and session

### tf.Graph: 

What is tf.Graph?

* tf.Graph should be defined before add
operations and tensors, otherwise we use default graph.

* tf.Graph is needed
whenever there are multiple models in one file.

* tf.Graph contains two informations. 
  
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

### tf.Session: 

What is tf.Session?

* tf.Session incorporates operations and
tensors. tf.Session also excute and evaluate the operations and tensors.

* tf.Session takes three arguments, which are all optional
  
(1) **target**:
The excution engine to connect to.
        
(2) **graph**: tf.Graph that
session wants to launch. If not specified, automatically links default graph.

(3) **config**: A ConfigProto protocol buffer with configuration options.
* Unlike tf.Graph, tf.Session should be placed before the operations. 

* tf.Session.run() function excutes the given operation.

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

### tf.InteractiveSession:

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
