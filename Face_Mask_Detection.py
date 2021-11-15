#!/usr/bin/env python
# coding: utf-8

# In[25]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input


# In[26]:


INIT_LR = 1e-4
EPOCHS = 10
BS = 32
dataset = "M/dataset"


# In[27]:


args={}
args["dataset"]=dataset


# In[28]:


images = ['With_Mask','Without_Mask'] 
data = [1915,1918] 
plt.pie(data, labels = images)
plt.show()


# In[29]:


x = ['With_Mask','Without_Mask']
y = [1915,1918] 
plt.barh(x, y)

for index, value in enumerate(y):
    plt.text(value, index, str(value))


# In[30]:


imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
	# extracting the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	# loading the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	# updating the data and labels lists, respectively
	data.append(image)
	labels.append(label)
# converting the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)


# In[31]:


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partitioning the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# constructing the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


# In[32]:


labels


# In[33]:


baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# constructing the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# placing the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False


# In[34]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# In[35]:


print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# training the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(X_train, Y_train, batch_size=BS),
	steps_per_epoch=len(X_train) // BS,
	validation_data=(X_test, Y_test),
	validation_steps=len(X_test) // BS,
	epochs=EPOCHS)


# In[44]:


predIdxs = model.predict(X_test, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# showing a nicely formatted classification report
print(classification_report(Y_test.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# serializeing the model to disk
print("[INFO] saving mask detector model...")
model.save('model11.h5')


# In[45]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# showing the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))


# In[46]:


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


# In[47]:


from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
img = image.load_img('M/dataset/without_mask/0_0_huangtingting_0008.jpg', target_size=(224, 224))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
New_pred = np.argmax(classes, axis=1)
if New_pred==[1]:
  print('Prediction: NO MASK')
else:
  print('Prediction: MASK')


# In[48]:


img = image.load_img('M/dataset/with_mask/0_0_0.jpg', target_size=(224, 224))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
New_pred = np.argmax(classes, axis=1)
if New_pred==[1]:
  print('    Prediction: NO MASK')
else: 
  print('    Prediction: MASK')


# In[49]:


img = image.load_img('M/dataset/with_mask/0_0_0 copy 15.jpg', target_size=(224, 224))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
New_pred = np.argmax(classes, axis=1)
if New_pred==[1]:
  print('    Prediction: NO MASK')
else:
  print('    Prediction: MASK')


# In[50]:


img = image.load_img('M/dataset/without_mask/0_0_caizhuoyan_0027.jpg', target_size=(224, 224))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
New_pred = np.argmax(classes, axis=1)
if New_pred==[1]:
  print('    Prediction: NO MASK')
else:
  print('    Prediction: MASK')


# In[53]:


img = image.load_img('M/dataset/without_mask/0_0_chenhaomin_0117.jpg', target_size=(224, 224))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
New_pred = np.argmax(classes, axis=1)
if New_pred==[1]:
  print('    Prediction: NO MASK')
else:
  print('    Prediction: MASK')


# In[ ]:





# In[ ]:




