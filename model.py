
import pandas as pd
import numpy as np

df = pd.read_csv("musk_csv.csv")

x = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)

import keras
from keras.layers import Dense
from keras.activations import relu
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import BatchNormalization

classifier = Sequential()

classifier.add(Dense(units = 256, kernel_initializer='uniform', activation=relu,input_dim = 166))
classifier.add(Dense(units = 128, kernel_initializer='uniform', activation=relu))
classifier.add(Dense(units = 256, kernel_initializer='uniform', activation=relu))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 128, kernel_initializer='uniform', activation=relu))
classifier.add(Dense(units = 64, kernel_initializer='uniform', activation=relu))
classifier.add(Dense(units = 128, kernel_initializer='uniform', activation=relu))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 64, kernel_initializer='uniform', activation=relu))
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint,TensorBoard
checkpoints=[]
checkpoints.append(ModelCheckpoint('checkpoints/weights_alpha_.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
checkpoints.append(TensorBoard(log_dir='tensorboard_logs/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

history = classifier.fit(x=X_train,y=y_train,batch_size=300,epochs=30,validation_data=(X_test, y_test),callbacks = checkpoints)


from matplotlib import pyplot as plt
# "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix as cm
y_pred = classifier.predict(X_test)
for j in range(len(y_pred)):
    if (y_pred[j]>0.5):
        y_pred[j] = 1
    else:
        y_pred[j] = 0
y_pred =np.squeeze(y_pred)
y_pred = y_pred.astype('int64')
cm = cm(y_test,y_pred)

precision = cm[1,1]/(cm[1,1]+cm[0,1])
recall = cm[1,1]/(cm[1,1]+cm[1,0])

f1_score = (2*precision*recall)/(precision+recall)