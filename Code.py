import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1. Dataset Setup
if not os.path.exists('Brain-Tumor-Classification-DataSet'):
    !git clone https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet.git

train_dir = 'Brain-Tumor-Classification-DataSet/Training'

# 2. Strategic Split (The key to high validation accuracy)
# We use 80% for training and 20% for validation from the SAME high-quality source.
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # This creates the stable validation set
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'   # 80% of images
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation' # 20% of images
)

# 3. Model Architecture
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x) # Moderate dropout to maintain accuracy
output = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# 4. Training (Stage 1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Training Stage 1: Establishing Baseline ---")
model.fit(train_gen, epochs=10, validation_data=val_gen)

# 5. Fine-Tuning (Stage 2: Pushing to 95%+)
print("\n--- Training Stage 2: Fine-Tuning for Max Validation ---")
base_model.trainable = True
for layer in base_model.layers[:-20]: # Unfreeze last 20 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping ensures we save the version with the HIGHEST validation accuracy
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(train_gen, epochs=15, validation_data=val_gen, callbacks=[callback])

# 6. Save
model.save('max_val_brain_model.keras')
print("\nSuccess! Model saved with optimized validation accuracy.")
