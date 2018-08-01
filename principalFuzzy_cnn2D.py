from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(5, 12, 1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=256, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen = ImageDataGenerator(rescale=1./255,
#                                         rotation_range=7,
#                                         horizontal_flip=True,
#                                         shear_range=0.2,
#                                         height_shift_range=0.07,
#                                         zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./FuzzyBase/train', 
                                                    color_mode='grayscale',
                                                    target_size=(5, 12), 
                                                    batch_size=10, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('./FuzzyBase/validation', 
                                                        color_mode='grayscale',
                                                        target_size=(5, 12), 
                                                        batch_size=10, class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('label batch shape:', labels_batch.shape)
    break

history = model.fit_generator(train_generator, steps_per_epoch=120, epochs=15, 
                              validation_data=validation_generator, validation_steps=72)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
