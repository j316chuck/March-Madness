from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 65, activation = 'relu', input_dim = 100))
model.compile(loss = 'categorical_crosentropy', optimizer = 'sgd', metrix = ['accuracy'])
model.fit(x_train, y_train, epochs = 5, batch_size = 32)
model.predict(x_test, batch_size = 128)
