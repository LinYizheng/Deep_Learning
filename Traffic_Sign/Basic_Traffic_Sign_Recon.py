'''
Apr 5, 2017
Author: Hong San Wong
Email: hswong1@uci.edu


 The Follow code should perform Traffic Sign recognition with CNN

'''


from __future__ import print_function

import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load training and testing datasets.
'''
ROOT_PATH = "~/"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")
'''


train_data_dir = "/home/user/san/Deep_Learning/Traffic_Sign/datasets/BelgiumTS/Training"
test_data_dir = "/home/user/san/Deep_Learning/Traffic_Sign/datasets/BelgiumTS/Testing"

images, labels = load_data(train_data_dir)


'''
images: a list of images, each image is represented by a numpy arrays
labels: list of label (range from 0 to 61: i.e. 62 classes)
'''

# Print Dataset
print("Unique labels:  {0}\nTotal Images: {1}".format(len(set(labels)),len(images)))

def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15,15))
    i=1
    for label in unique_labels:
        image = images[labels.index(label)]
        plt.subplot(8,8,i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label,labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


display_images_and_labels(images, labels)



def display_label_images(images, label):
    limit = 24 #Show max of 24 images
    plt.figure(figsize=(15,5))
    i=1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3,8,i) # 3 rows, 8 images in a row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

display_label_images(images,32)

# Print size of image
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape,image.min(),image.max()))


# Resize image (Convert to 32*32. (0-255) to (0-1))
images32 = [skimage.transform.resize(image,(32,32)) for image in images]
display_images_and_labels(images32,labels)

for image in images32[:5]:
    print("shape: {0}, min:{1}, max:{2}".format(image.shape, image.min(), image.max()))


labels_a = np.array(labels)
images_a = np.array(images32)
print("labels:",labels_a.shape,"\nimages: ",images_a.shape)


# Create a graph to hold a model
graph = tf.Graph()

with graph.as_default():
    #Placeholder
    images_ph = tf.placeholder(tf.float32, [None,32,32,3])
    labels_ph = tf.placeholder(tf.int32,[None])

    # Flatten input from: [None,H,W,Channels] to [None, H*W*Channel] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    #Fully Connected layer. Generate logits of size [None, 62]
    #logits = tf.contrib.fully_connected(images_flat,62,tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(images_flat,62,tf.nn.relu)

    predicted_labels = tf.argmax(logits,1)

    # Cross-Entropy Loss Function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels_ph))

    # Train Op
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()




# Training
session = tf.Session(graph = graph)
# Step1. Initialize (We don't care what's being return. So we don't hold on to it)
_ = session.run([init])

for i in range(201):
    _, loss_value = session.run([train, loss], feed_dict = {images_ph: images_a, labels_ph: labels_a})

    if i % 10 == 0:
        print("Loss: ", loss_value)


# ================================= USE THE MODEL =====================================
# Pick 10 random sample
sample_indexes = random.sample(range(len(images32)),10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = session.run([predicted_labels], feed_dict={images_ph: sample_images})[0]

print(sample_labels)
print(predicted)

# Visualize Predicted Result
fig = plt.figure(figsize=(10,10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5,2,1+i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40,10,"Truth:              {0}\nPrediction:  {1}".format(truth,prediction),fontsize=12,color=color)
    plt.imshow(sample_images[i])



# ==================================== EVALUATION ========================================
# Load test data
test_images, test_labels = load_data(test_data_dir)

test_images32 = [skimage.transform.resize(image,(32,32)) for image in test_images]
display_images_and_labels(test_images32, test_labels)

# Run prediction against the full test_set
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))

session.close()





