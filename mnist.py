import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from keras.preprocessing import image 
 
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
  
x_train = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1) 
 

x_train = x_train.astype('float32') / 255 
x_test = x_test.astype('float32') / 255 
 

y_train = tf.keras.utils.to_categorical(y_train, 10) 
y_test = tf.keras.utils.to_categorical(y_test, 10) 
 

cnn = tf.keras.models.Sequential() 

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', 
input_shape=[28, 28, 1])) 
 

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) 
 

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) 

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) 
 

cnn.add(tf.keras.layers.Flatten()) 

cnn.add(tf.keras.layers.Dense(units=64, activation='relu')) 
 
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax')) 
 
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
trained_model = cnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2) 
 
test_image = x_test[0].reshape(1, 28, 28, 1) 
 
results = cnn.predict(test_image) 
 
class_indices = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 
8: 'Eight', 9: 'Nine'} 
predicted_class = np.argmax(results) 

print("The model predicts:", class_indices[predicted_class]) 

plt.imshow(x_test[0].reshape(28, 28), cmap='gray') 
plt.title(f"Predicted: {class_indices[predicted_class]}") 
plt.show()
