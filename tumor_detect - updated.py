# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam,Nadam, SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.metrics import confusion_matrix
import keras


# define cnn model
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(1, activation='sigmoid'))
 # compile model
 opt = SGD(learning_rate=0.001, momentum=0.5)
 model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.AUC() ])
 return model

# define model
model = define_model()

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

# prepare iterators
#train_it = datagen.flow_from_directory('C:/Users/SHUVANKAR/Documents/paper/Training/notumor',
# class_mode='binary', batch_size=64, target_size=(200, 200))
train_it = datagen.flow_from_directory('C:/Users/SAHIL/OneDrive/Tumor using ml/archive (6)/Training',
 class_mode='binary', batch_size=64, target_size=(200, 200))

test_it = datagen.flow_from_directory('C:/Users/SAHIL/OneDrive/Tumor using ml/archive (6)/Testing',
 class_mode='binary', batch_size=64, target_size=(200, 200))
# fit model
history = model.fit(train_it, steps_per_epoch=len(train_it),
 validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)

# evaluate model
_,acc,precison,rcall,aucc = model.evaluate(test_it,batch_size=128)

#print(len(model.evaluate(test_it)))

# plot diagnostic learning curves
#def summarize_diagnostics(history):
# plot loss
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
pyplot.show()                
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='green', label='train')
pyplot.plot(history.history['precision'], color='red', label='train')
pyplot.plot(history.history['recall'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
pyplot.show()

pyplot.close()