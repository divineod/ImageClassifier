# Import required python libraries
import tensorflow as tf
import re
import numpy as np
import configparser
import time
import math
import pickle
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Read required initial paths from the config file
config = configparser.RawConfigParser()
config.optionxform = lambda option: option
config.read(r'/Users/divinefavourodion/Documents/ImageClassifier/config.ini')

training_data_path = dict(config.items('data'))['GCS_PATH']  # Path for saving file
model_path = dict(config.items('classifier'))['model']

sleep_time = 5 # Define time for breaks in between code segments

Flower_Classes = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium', 'tiger lily',
           'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',  # 00 - 09
           'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower',
           'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',  # 10 - 19
           'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy',
           'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',  # 20 - 29
           'carnation', 'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya',
           'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',  # 30 - 39
           'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold',
           'buttercup', 'daisy', 'common dandelion',  # 40 - 49
           'petunia', 'wild pansy', 'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff', 'gaura', 'geranium',
           'orange dahlia', 'pink-yellow dahlia',  # 50 - 59
           'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
           'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy',  # 60 - 69
           'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus',
           'toad lily', 'anthurium',  # 70 - 79
           'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ',
           'watercress', 'canna lily',  # 80 - 89
           'hippeastrum ', 'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
           'mexican petunia', 'bromelia', 'blanket flower',  # 90 - 99
           'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose']



IMAGE_SIZE = [192, 192]
BATCH_SIZE = 16 # Define training batch size
image_192 = training_data_path + '/192x192_images'
image_224 = training_data_path + '/224x224_images'
image_331 = training_data_path + '/331x331_images'
image_512 = training_data_path + '/512x512_images'
AUTO = tf.data.experimental.AUTOTUNE  # Allow TensorFlow to determine the optimal number of parallel calls based on the available hardware and input data



### Retraining functions
def get_filename(image_size):
    training_filenames = tf.io.gfile.glob(image_size + '/train/*.tfrec')
    validation_filenames = tf.io.gfile.glob(image_size + '/val/*.tfrec')
    test_filenames = tf.io.gfile.glob(image_size + '/test/*.tfrec')
    return training_filenames, validation_filenames, test_filenames


## Functions for reading the tfrec images

def decode_image(image_data): # Function to convert jpeg image to tensor object
    image = tf.image.decode_jpeg(image_data, channels=3) # convert the jpeg binary into a tensor object
    image = tf.cast(image, tf.float32) / 255.0 # Cast the tensor into type float32
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # Reshape to fit the input layer of the DL model
    return image # Return the image matrix


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label  # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(feature):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string indicates bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means scalar element
    }
    feature = tf.io.parse_single_example(feature, UNLABELED_TFREC_FORMAT)
    image = decode_image(feature['image'])
    id_num = feature['id']
    return image, id_num  # returns a dataset of image(s)


def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable the order in which the tfrec dataset is read in

    dataset = tf.data.TFRecordDataset(filenames,
                                      num_parallel_reads=AUTO)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order)  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset



## Function to augment the training data

def data_augment_train(image, label):
    image = tf.image.resize(image, [192, 192])
    image = tf.image.random_flip_left_right(image)
    return image, label

def data_augment_test_val(image, label):
    image = tf.image.resize(image, [192, 192])
    return image, label


## Functions to fetch, shuffle and batch train, test, and validation sets
def get_training_dataset():
    dataset = load_dataset(training_data_unprocessed, labeled=True) # Load the training data
    dataset = dataset.map(data_augment_train, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048) # shuffle the data
    dataset = dataset.batch(BATCH_SIZE) # Using a batch size of 16
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(valid_data_unprocessed, labeled=True, ordered=ordered) # Load thr validation data
    dataset = dataset.map(data_augment_test_val, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE) # Batch size of 16
    dataset = dataset.cache() # Using cache memory to speed up training process
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(test_data_unprocessed, labeled=False, ordered=ordered)
    dataset = dataset.map(data_augment_test_val, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE) # Batch size of 16
    dataset = dataset.prefetch(AUTO)
    return dataset



# Function to count the number items in each tfrec file
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

## Functions to display single or image batches
def batch_to_numpy_images_and_labels(data): # Function to convert (image,label) tuple to numpy objects
    images, labels = data
    numpy_images = images.numpy() # Convert image to numpy object
    numpy_labels = labels.numpy() # Convert label to numpy object
    if numpy_labels.dtype == object:  # binary string in this case
        numpy_labels = [None for _ in enumerate(numpy_images)]     # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label): # Function to print if the image was correctly classified or not by the model
    if correct_label is None:
        return Flower_Classes[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(Flower_Classes[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                Flower_Classes[correct_label] if not correct else ''), correct


def display_one_flower(image, title, subplot, red=False, titlesize=16): # Function to display a single flower
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0: # Check if it has a title
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize / 1.2), color='red' if red else 'black',
                  fontdict={'verticalalignment': 'center'}, pad=int(titlesize / 1.5))
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_batch_of_images(databatch, predictions=None): # Function to display a batch of images
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)] # Check for image labels in the batch, replace with list of None dtypes if none

    # auto-squaring: this will drop data that does not fit into square
    # or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    FIGSIZE = 13.0 # Plot figure size
    SPACING = 0.1 # Spacing between subplots
    subplot = (rows, cols, 1) # Define subplots for the image batch

    # Determine how to shape the figure based of row-column arrangement of subplots
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display images
    for i, (image, label) in enumerate(zip(images[:rows * cols], labels[:rows * cols])):
        title = '' if label is None else Flower_Classes[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE * SPACING / max(rows,
                                                    cols) * 40 + 3  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

## Function to display training and validation (loss and accuracy) curves
def display_curves(training, validation, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'])

# Learning Rate Schedule for fine-tuning #
def exponential_lr(epoch,
                   start_lr = 0.00001, min_lr = 0.00001, max_lr = 0.00005,
                   rampup_epochs = 5, sustain_epochs = 0,
                   exp_decay = 0.8):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        # linear increase from start to rampup_epochs
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) /
                  rampup_epochs * epoch + start_lr)
        # constant max_lr during sustain_epochs
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        # exponential decay towards min_lr
        else:
            lr = ((max_lr - min_lr) *
                  exp_decay**(epoch - rampup_epochs - sustain_epochs) +
                  min_lr)
        return lr
    return lr(epoch,
              start_lr,
              min_lr,
              max_lr,
              rampup_epochs,
              sustain_epochs,
              exp_decay)

### Get unproccessed datasets
training_data_unprocessed = get_filename(image_192)[0]
test_data_unprocessed = get_filename(image_192)[1]
valid_data_unprocessed = get_filename(image_192)[2]


### Display the number of images in the dataset
train_image_num = count_data_items(training_data_unprocessed ) # Number of images in the training set
val_image_num = count_data_items(test_data_unprocessed) # Number of images in the test set
test_image_num = count_data_items(valid_data_unprocessed) # Number of images in the validation set
print('---------------------------------------------------------------')
print('The Dataset contains: {} training images, {} validation images, {} unlabeled test images'.format(train_image_num, val_image_num, test_image_num))

# Pre-processed datasets
train_data = get_training_dataset()
val_data = get_validation_dataset()
test_data = get_test_dataset()

### Display the dimensions of the train, test and validation datasets
print('---------------------------------------------------------------')
print("Training data shape:")
for image, id_num in train_data.take(1):
    print(image.numpy().shape, id_num.numpy().shape)
time.sleep(sleep_time) # wait 5 seconds
print("Validation data shape:")
for image, id_num in val_data.take(1):
    print(image.numpy().shape, id_num.numpy().shape)
time.sleep(sleep_time) # wait 5 seconds
print("Test data shape:")
for image, id_num in test_data.take(1):
    print(image.numpy().shape, id_num.numpy().shape)


### Model training

## Define the pre-trained classifier
LR_EPOCHS = 50
pretrained_model = tf.keras.applications.resnet50.ResNet50(
    weights='imagenet', # Use imagenet weights
    include_top=False , # Remove input layer
    input_shape=[*IMAGE_SIZE, 3], # Input layer of classifier corresponding to training data dimensions
    pooling='avg', # Utilizing average pooling
) # RESNET pretrained classifier trained on imagenet weights
pretrained_model.trainable = True

predictions = tf.keras.layers.Dense(len(Flower_Classes), activation='softmax')(pretrained_model.input) # Final output layer of CNN
model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)

model.compile(
    optimizer='nadam',
    loss = 'sparse_categorical_crossentropy', # Using sparse_categorical_crossentropy as loss function
    metrics=['sparse_categorical_accuracy'], # "" "" as key metric
)# Final defined model

print('Training RESNET 50...')
print('Using imagenet weights...')
print('The model summary:')
model.summary()

##Print the learning rate schedule

lr_callback = tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=True)
rng = [i for i in range(LR_EPOCHS)]
y = [exponential_lr(x) for x in rng]
plt.figure(figsize=(15,7))
plt.plot(rng, y)
plt.title("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
plt.xlabel('Learning rate')
plt.ylabel('Epochs')
plt.show()

## Train the classifier
Epochs = 30 # Number of epochs for training
Steps_per_epoch = train_image_num // BATCH_SIZE

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=Epochs,
    steps_per_epoch=Steps_per_epoch,
    callbacks=[lr_callback],
)

## Plot the training and validation curves

display_curves(
    history.history['loss'],
    history.history['val_loss'],
    'loss',
    211,
)
display_curves(
    history.history['sparse_categorical_accuracy'],
    history.history['val_sparse_categorical_accuracy'],
    'accuracy',
    212,
)


### Evaluate the retrained model






