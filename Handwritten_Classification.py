import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from  keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


#Data from MNIST dataset(Subset of NIST)
(X_train, Y_train),(X_test, Y_test)=mnist.load_data()

#Sample Graph from the dataset
plt.imshow(X_train[0], cmap='gray' )
plt.title('Class'+str(Y_train[0]))

feature_train=X_train.reshape(X_train.shape[0],28,28,1)
feature_test=X_test.reshape(X_test.shape[0],28,28,1)

feature_train=feature_train.astype("float32")
feature_test=feature_test.astype("float32")

#Data noramalization for the Neural Networks
feature_test/=255
feature_train/=255

#Data Noramlization for the output class
target_train= np_utils.to_categorical(Y_train,10)
target_test= np_utils.to_categorical(Y_test,10)

#Model is being initiated
model= Sequential()

#Making Dense Network
model.add(Conv2D(32, (3,3),input_shape=(28,28,1)))
model.add(Activation('relu'))

#Function to make the algo run faster by making appropriate changes
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

#Model Flateening for having input
model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Dropout is being used to reduce overfitting
model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#Invoke this when not using data augmentation for the reducing the overfiting 
#model.fit(feature_train,target_train,batch_size=126, epochs=2, validation_data=(feature_test,target_test))


train_generator=ImageDataGenerator(rotation_range=7, width_shift_range=0.05, height_shift_range=0.07, shear_range=0.2,zoom_range=0.04)

test_generator=ImageDataGenerator()

train_generator=train_generator.flow(feature_train,target_train,batch_size=70)
test_generator=test_generator.flow(feature_test,target_test,batch_size=70)

model.fit_generator(train_generator,steps_per_epoch=60000//70, epochs=5, validation_data=test_generator, validation_steps=10000//70)
score=model.evaluate(feature_test,target_test)
print("Test Accuracy: %.2f" %score[1])

