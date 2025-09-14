# Step 1 : import necessary libraries and frameworks
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)


# Step 2: Load and preprocess dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(" Raw training shape:", x_train.shape)
print(" Raw test shape:", x_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

print(" Training shape after preprocess:", x_train.shape)
print(" Test shape after preprocess:", x_test.shape)

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# Step 3: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

print("\n Model built. Summary:\n")
model.summary()


# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("\n Model compiled with Adam optimizer")


# Step 5: Train the model
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.1,
                    verbose=2)


# Step 6: Evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n Test accuracy: {acc:.4f}, loss: {loss:.4f}")



# Step 7: Make predictions on first 2 test images
imgs = x_test[:2]
true_labels = y_test[:2]
probs = model.predict(imgs)
preds = np.argmax(probs, axis=1)

# Plot images with true vs predicted labels 
plt.figure(figsize=(6,3))
for i in range(2):
    plt.subplot(1, 2, i+1) 
    plt.imshow(imgs[i].reshape(28,28), cmap="gray")
    plt.axis("off")

    pred_label = class_names[preds[i]]
    true_label = class_names[true_labels[i]]
    confidence = probs[i][preds[i]] * 100

# Green title if correct, red if wrong
    color = "green" if pred_label == true_label else "red"
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}\n({confidence:.2f}%)", color=color)

plt.tight_layout()
plt.savefig("predictions.png")

plt.show()



# print predictions in text
for i, p in enumerate(preds):
    confidence = probs[i][p] * 100
    print(f" Image {i}: predicted -> {class_names[p]} "
          f"(confidence {confidence:.2f}%)")


# Step 8: Save the model
model.save("fashion_cnn_notebook.keras")
print("Model saved as fashion_cnn_notebook.keras")

