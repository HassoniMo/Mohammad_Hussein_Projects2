# Steps 1 to 4 

# HUSSEIN MOHAMMAD - 501098569
# AER850 Section 01 Project 2

'''STEP 1 - DATA PROCESSING'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
IMG_SIZE = (500, 500)
BATCH_SIZE = 32

# Define paths
train_dir = './Data/Train'
val_dir = './Data/Valid'
test_dir = './Data/Test'

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)


'''STEP 2 & 3 - NEURAL NETWORK ARCHITECTURE DESIGN and HYPERPARAMETER ANALYSIS'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

# Build the model with Input layer
model = Sequential([
    # Input layer
    Input(shape=(500, 500, 3)),

    # Convolutional layers with Batch Normalization
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and fully connected layers
    Flatten(),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    
    # Output layer
    Dense(3, activation='softmax')  # 3 classes: crack, missing-head, paint-off
])

model.summary()

# Compile the model with a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Reduced learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


'''STEP 4 - MODEL EVALUATION'''
import matplotlib.pyplot as plt

# Train the model with callbacks
history = model.fit(
    train_generator,
    epochs=25,  
    validation_data=validation_generator,
)

model.save('model.keras')

# Plotting accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
