import matplotlib.pyplot as plt
import tensorflow as tf

##############################################################################################################
def test_helper_file(some_input):
    print(some_input)

##############################################################################################################
def plot_loss_curves(history):
    """
    Returns separate loss curves for training and val. metrics
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()


##############################################################################################################
# create a function to import image and resize it and shape it to the model
def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename, turns into tensor, reshape
    """
    # read in the image
    img = tf.io.read_file(filename)
    # decode the read file into a tensor
    img = tf.image.decode_image(img)
    # resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # rescale the image
    img = img / 255.

    return img


##############################################################################################################
def pred_and_plot(model, filename, class_names):
    """
    Imports an image at filename, makes a prediction,
    plots the image with predicted class as the title
    """
    # import
    img = load_and_prep_image(filename)

    # make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    print(pred)
    print(tf.argmax(pred[0]))

    # logic for multiclass vs binary
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]

    # plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);
    
