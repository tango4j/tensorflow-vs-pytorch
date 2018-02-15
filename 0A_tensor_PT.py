import torch

# --------------------------------------------------------
# 1. Tensor
# --------------------------------------------------------

# (1) Tensor
# Unlike Tensorflow, the tensor command itself determines the data type.  
# Then, we feed the python list.

# torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor
# torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor
# torch.cuda.IntTensor, torch.cuda.LongTensor

# If you want to use CPU, you can remove cuda. Ex: torch.FloatTensor
CPU_tensor = torch.FloatTensor([[1,2], [3, 4]])
GPU_tensor = torch.cuda.FloatTensor([[1, 2], [3, 4]])

# Also, you could first declare the shape and fill it with in-place functions.
fill_zero_in_the_tensor = torch.FloatTensor(2, 2).zero_()
print 'fill_zero_in_the_tensor: ', fill_zero_in_the_tensor

# For the most cases, torch Tensor usually behaves like numpy array. 

# (2) Variable

# --------------------------------------------------------
# Convert Numpy Array to Tensor
# --------------------------------------------------------
# numpy_array = np.asarray([[1, 2, 3], [4, 5, 6]], np.float32)
# tensorflow_tensor = tf.convert_to_tensor(numpy_array, np.float32)
# tf.InteractiveSession()
# evaluated_tensor = tensorflow_tensor.eval()
# print('Numpy Array: ', numpy_array)
# print('Tensorflow Tensor: ', tensorflow_tensor)
# print('Evaluated Tensor: ', evaluated_tensor)


# --------------------------------------------------------
# Dimension Check
# --------------------------------------------------------
# shape = my_image.shape
# print('Dimension(shape) of my_image: ', shape)

# # However, after graph run, the variable "rank" will hold the rank value.
# rank_var =tf.rank(my_image)
# rank_var_evaluated = rank_var.eval()
# print("Printing 'rank_var' does not show rank of the variable.")
# print('rank_var:', rank_var) 
# print('But after graph run (evaluation) it obtains rank value.')
# print('Evaluated rank_var :', rank_var_evaluated)

# --------------------------------------------------------
# Shaping Tensor Variable 
# --------------------------------------------------------
# tensor0 = tf.ones([2, 3, 4])  # 3D Tensor variable filled with ones.
# print('The shape of tensor0: %s' %tensor0.shape)

# tensor1 = tf.reshape(tensor0, [2, 12])  # Make "tensor0" matrix into 2 by 12 matrix.
# print('The shape of tensor1: %s' %tensor1.shape)

# # If you want to let it automatically figure out the rest of the shape, do:
# tensor2 = tf.reshape(tensor1, [4, -1])  # Make "tensor1" matrix into 4 by 6 matrix.
# print('The shape of tensor2: %s' %tensor2.shape)

# tensor3 = tf.reshape(tensor2, [2, 2, -1])  # Make "tensor2" matrix into 4 by 6 matrix.
# print('The shape of tensor3: %s' %tensor3.shape)










other
