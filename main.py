import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# مسیر دیتاست
dataset_path = 'F:/animal_detector/raw-img'  # این مسیر رو به مسیر واقعی پوشه dataset خودت تغییر بده

# اندازه تصاویر
image_size = (128, 128)
batch_size = 32

# آماده‌سازی داده‌ها
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ساخت مدل
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# کامپایل مدل
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# آموزش مدل
model.fit(train_data, validation_data=val_data, epochs=5)

# ذخیره مدل
model.save('animal_model.keras')

# ذخیره دیکشنری کلاس‌ها
with open('class_names.pkl', 'wb') as f:
    pickle.dump(train_data.class_indices, f)

print("✅ آموزش مدل به پایان رسید و فایل‌ها ذخیره شدند.")