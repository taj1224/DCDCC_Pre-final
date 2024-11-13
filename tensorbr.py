import tensorflow as tf 
 
tensor1 = tf.constant([[1, 2, 3, 4], 
                       [5, 6, 7, 8], 
                       [9, 10, 11, 12]]) 
print("Shape of tensor1:", tensor1.shape) 
 
tensor2 = tf.constant([1, 2, 3, 4]) 
print("Shape of tensor2:", tensor2.shape) 
 
try: 
    result = tensor1 + tensor2 
    print("Result of tensor1 + tensor2:\n", result) 
except Exception as e: 
    print("Error:", e)
