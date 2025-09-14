# Step 1 : import necessary libraries and frameworks
install.packages("keras3")
install.packages("tensorflow")

ibrary(tensorflow)
install_tensorflow()

library(keras3)
library(tensorflow)


# Step 2: Load & preprocess dataset
fashion <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion

x_train <- x_train / 255
x_test  <- x_test / 255

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test  <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

class_names <- c("T-shirt/top","Trouser","Pullover","Dress","Coat",
                 "Sandal","Shirt","Sneaker","Bag","Ankle boot")


# Step 3: Build CNN model (6 layers)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)


# Step 4: Compile
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# Step 5: Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 64,
  validation_split = 0.1
)

# Step 6: Evaluate the model
scores <- model %>% evaluate(x_test, y_test, verbose = 2)
cat(sprintf("Test accuracy: %.4f, loss: %.4f\n", scores["accuracy"], scores["loss"]))


# Step 7: Predictions on first 2 images
imgs <- x_test[1:2,,,,drop = FALSE] 
true_labels <- y_test[1:2]
probs <- model %>% predict(imgs)
preds <- apply(probs, 1, which.max) - 1   

# Plot predictions 
par(mfrow=c(1,2))
for (i in 1:2) {
  img <- matrix(imgs[i,,,1], nrow=28, byrow=TRUE)
  image(1:28, 1:28, img[28:1,], col=gray.colors(255), axes=FALSE, main="")

  pred_label <- class_names[preds[i]+1]
  true_label <- class_names[true_labels[i]+1]
  confidence <- round(probs[i,preds[i]+1]*100,2)

  title(
    main = sprintf("Pred: %s\nTrue: %s\n(%.2f%%)", pred_label, true_label, confidence),
    col.main = ifelse(pred_label==true_label, "green", "red")
  )
}

# Step 8: Save model
save_model(model, "fashion_cnn.keras", overwrite = TRUE)
model <- load_model("fashion_cnn.keras")