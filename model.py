#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tensorflow as tf


# In[30]:


import matplotlib.pyplot as plt
from keras.models import Sequential


# In[31]:


from tensorflow.keras import models, layers


# In[32]:


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=25


# In[33]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[34]:


class_names = dataset.class_names
class_names


# In[35]:


plt.figure(figsize=(20, 20))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")


# In[36]:


len(dataset)


# In[37]:


train_size = 0.8
len(dataset)*train_size


# In[38]:


train_ds = dataset.take(400)
len(train_ds)


# In[39]:


test_ds = dataset.skip(400)
len(test_ds)


# In[40]:


val_size=0.1
len(dataset)*val_size


# In[41]:


val_ds = test_ds.take(50)
len(val_ds)


# In[42]:


test_ds = test_ds.skip(50)
len(test_ds)


# In[43]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[44]:


resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])


# In[45]:


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])


# In[46]:


model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256,activation="relu"),
    layers.Dense(len(class_names))
])


# In[47]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[48]:


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = EPOCHS,
    verbose = 1,
    batch_size=BATCH_SIZE
)


# In[49]:


scores = model.evaluate(test_ds)


# In[50]:


history


# In[51]:


history.history.keys()


# In[52]:


history.history['accuracy']


# In[53]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[54]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc,label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


# In[55]:


import numpy as np
def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class


# In[56]:


plt.figure(figsize = (30,30))
for images,labels in test_ds.take(1):
    for i in range(10):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        
        predicted_class = predict(model,images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual : {actual_class},\n Predicted: {predicted_class}")
        
        plt.axis("off")


# In[ ]:




