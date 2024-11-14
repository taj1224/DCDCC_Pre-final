from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from keras.datasets import imdb

top_words=5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

max_words=500
x_train=sequence.pad_sequences(x_train,maxlen=max_words)
x_test=sequence.pad_sequences(x_test,maxlen=max_words)

model=Sequential()
model.add(Embedding(top_words,100,input_length=max_words))
model.add(GRU(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(x_train,y_train,epochs=3,batch_size=64)
scores=model.evaluate(x_test,y_test,verbose=0)
print("Accuracy: %.2f%%"%(scores[1]*100))

y_predict=model.predict(x_test)

y_predict_binary=(y_predict>0.5).astype(int)
print(y_predict_binary)
