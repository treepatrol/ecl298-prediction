### Lab: Deep Learning
### Note: this lab is slightly different from that in the August 2021 printing of ISLRII
### and from the on-line version. Here we use the newer keras3 package 


# first run this (takes a while)
# install.packages(c("keras3"))


## A Single Layer Network on the Hitters Data

###
library(ISLR2)
Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

###
lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(abs(lpred - Salary)))

###
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary

###
library(glmnet)
cvfit <- cv.glmnet(x[-testid, ], y[-testid], type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))

###
library(keras3)
## reticulate::use_condaenv(condaenv = "r-tensorflow")
modnn <- keras_model_sequential(input_shape = ncol(x)) |>
  layer_dense(units = 50, activation = "relu") |>
  layer_dropout(rate = 0.4) |> 
  layer_dense(units = 1)

compile(modnn, loss = "mse", optimizer = optimizer_rmsprop(),
        metrics = list("mean_absolute_error"))

### this one takes a while (1500 epochs!)
history <- fit(modnn, x[-testid, ], y[-testid], epochs = 1500, 
               batch_size=32, validation_data=list(x[testid, ], y[testid]))


npred <- predict(modnn, x[testid, ])
mean(abs(y[testid] - npred))


## A Multilayer Network on the MNIST Digit Data

###
mnist <- dataset_mnist()

# illustrate with terra
library(terra)
show_digit <- \(i) {
  plot(rast(mnist$train$x[i,,]), legend=FALSE,
       axes=FALSE, col=gray(255:0/255), mar=c(0,0,0,0))
  text(2, 26, mnist$train$y[i], cex=2)
}
par(mfrow=c(5,5), mar=c(0,0,0,0))
for (i in 1:25) show_digit(i)

x_train <- mnist$train$x
g_train <- mnist$train$y
x_test <- mnist$test$x
g_test <- mnist$test$y
dim(x_train)
dim(x_test)

###
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
y_train <- to_categorical(g_train, 10)
y_test <- to_categorical(g_test, 10)

###
x_train <- x_train / 255
x_test <- x_test / 255

###
modelnn <- keras_model_sequential(input_shape = 784) |>
  layer_dense(units = 256, activation = "relu") |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 10, activation = "softmax")

summary(modelnn)

compile(modelnn, loss = "categorical_crossentropy",
        optimizer = optimizer_rmsprop(), metrics = "accuracy")

system.time(
  history <- fit(modelnn, x_train, y_train, epochs = 30, 
                 batch_size = 128, validation_split = 0.2)
)
plot(history, smooth = FALSE)

accuracy <- function(pred, truth) {
  mean(drop(as.numeric(pred)) == drop(truth))
}  

predict(modelnn, x_test) |> 
  op_argmax(-1, zero_indexed = T) |> accuracy(g_test)


###
modellr <- keras_model_sequential(input_shape = 784)  |>
  layer_dense(units = 10, activation = "softmax")

summary(modellr)

compile(modellr, loss = "categorical_crossentropy",
        optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

fit(modellr, x_train, y_train, epochs = 30,
    batch_size = 128, validation_split = 0.2)

predict(modellr, x_test) |> op_argmax(-1, zero_indexed = T) |> accuracy(g_test)


## 10.9.3 Convolutional Neural Networks

cifar100 <- dataset_cifar100()
names(cifar100)
x_train <- cifar100$train$x

par(mar = c(0, 0, 0, 0), mfrow = c(5, 5))
for (i in 1:25) {
  terra::plotRGB(terra::rast(x_train[i,,, ]))
  terra::halo(3,30, cifar100$train$y[i], cex=1.5)
}
# elepants
index <- which(cifar100$train$y == 31)[1:25]
for (i in index) {
  terra::plotRGB(terra::rast(x_train[i,,, ]))
}

g_train <- cifar100$train$y
x_test <- cifar100$test$x
g_test <- cifar100$test$y
dim(x_train)
range(x_train[1,,, 1])
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(g_train, 100)
dim(y_train)

model <- keras_model_sequential(input_shape = c(32, 32, 3)) |>
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                padding = "same", activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 128, kernel_size = c(3, 3),
                padding = "same", activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 256, kernel_size = c(3, 3),
                padding = "same", activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_flatten() |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 512, activation = "relu") |>
  layer_dense(units = 100, activation = "softmax")
summary(model)

compile(model, loss = "categorical_crossentropy",
        optimizer = optimizer_rmsprop(), 
        metrics = c("accuracy"))

# this takes a while to run  
history <- fit(model, x_train, y_train, epochs = 30,
               batch_size = 128, validation_split = 0.2)

predict(model, x_test) |> 
  op_argmax(-1, zero_indexed = T) |> accuracy(g_test)


## 10.9.4 Using Pretrained CNN Models

###
burl <- "https://www.statlearning.com/s/book_images.zip"
if (!file.exists("book_images")) {
  download.file(burl, basename(burl), mode="wb")
  unzip("book_images.zip")
}
image_files <- list.files("book_images", pattern=".jpg$", full.names = TRUE)
num_images <- length(image_files)

x <- array(dim = c(num_images, 224, 224, 3))
for (i in 1:num_images) {
  img <- image_load(image_files[i], target_size = c(224, 224))
  x[i,,, ] <- image_to_array(img)
}

x <- imagenet_preprocess_input(x)
model <- application_resnet50(weights = "imagenet")
summary(model)

pred6 <- predict(model, x) |>
  imagenet_decode_predictions(top = 3)
names(pred6) <- basename(image_files)
print(pred6)


## 10.9.5 IMDb Document Classification
max_features <- 10000
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
x_train[[1]][1:12]

word_index <- dataset_imdb_word_index()
decode_review <- function(text, word_index) {
  word <- names(word_index)
  idx <- unlist(word_index, use.names = FALSE)
  word <- c("<PAD>", "<START>", "<UNK>", "<UNUSED>", word)
  idx <- c(0:3, idx + 3)
  words <- word[match(text, idx, 2)]
  paste(words, collapse = " ")
}

decode_review(x_train[[1]][1:12], word_index)

library(Matrix)
one_hot <- function(sequences, dimension) {
  seqlen <- sapply(sequences, length)
  n <- length(seqlen)
  rowind <- rep(1:n, seqlen)
  colind <- unlist(sequences)
  sparseMatrix(i = rowind, j = colind,
               dims = c(n, dimension))
}

x_train_1h <- one_hot(x_train, 10000)
x_test_1h <- one_hot(x_test, 10000)
dim(x_train_1h)
nnzero(x_train_1h) / (25000 * 10000)

set.seed(3)
ival <- sample(seq(along = y_train), 2000)

library(glmnet)
fitlm <- glmnet(x_train_1h[-ival, ], y_train[-ival],
                family = "binomial", standardize = FALSE)
classlmv <- predict(fitlm, x_train_1h[ival, ]) > 0
acclmv <- apply(classlmv, 2, accuracy,  y_train[ival] > 0)

par(mar = c(4, 4, 4, 4), mfrow = c(1, 1))
plot(-log(fitlm$lambda), acclmv)

###
model <- keras_model_sequential(input_shape = 10000) |>
  layer_dense(units = 16, activation = "relu") |>
  layer_dense(units = 16, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")

compile(model, optimizer = "rmsprop",
        loss = "binary_crossentropy", metrics = "accuracy")

history <- fit(model, x_train_1h[-ival, ], y_train[-ival],
               epochs = 20, batch_size = 512,
               validation_data = list(x_train_1h[ival, ], y_train[ival]))

###
history <- fit(model, x_train_1h[-ival, ], y_train[-ival], epochs = 20,
               batch_size = 512, validation_data = list(x_test_1h, y_test))



## 10.9.6 Recurrent Neural Networks

### Sequential Models for Document Classification

wc <- sapply(x_train, length)
median(wc)
sum(wc <= 500) / length(wc)
maxlen <- 500
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
dim(x_train)
dim(x_test)
x_train[1, 490:500]
###

model <- keras_model_sequential() |>
  layer_embedding(input_dim = 10000, output_dim = 32) |>
  layer_lstm(units = 32) |>
  layer_dense(units = 1, activation = "sigmoid")

compile(model, optimizer = "rmsprop",
        loss = "binary_crossentropy", metrics = "accuracy")

history <- fit(model, x_train, y_train, epochs = 10,
               batch_size = 128, validation_data = list(x_test, y_test))

predy <- predict(model, x_test) > 0.5
mean(abs(y_test == as.numeric(predy)))


### Time Series Prediction

library(ISLR2)
xdata <- data.matrix(
  NYSE[, c("DJ_return", "log_volume","log_volatility")]
)
istrain <- NYSE[, "train"]
xdata <- scale(xdata)

lagm <- function(x, k = 1) {
  n <- nrow(x)
  pad <- matrix(NA, k, ncol(x))
  rbind(pad, x[1:(n - k), ])
}

arframe <- data.frame(log_volume = xdata[, "log_volume"],
                      L1 = lagm(xdata, 1), L2 = lagm(xdata, 2),
                      L3 = lagm(xdata, 3), L4 = lagm(xdata, 4),
                      L5 = lagm(xdata, 5))

arframe <- arframe[-(1:5), ]
istrain <- istrain[-(1:5)]
arfit <- lm(log_volume ~ ., data = arframe[istrain, ])
arpred <- predict(arfit, arframe[!istrain, ])
V0 <- var(arframe[!istrain, "log_volume"])
1 - mean((arpred - arframe[!istrain, "log_volume"])^2) / V0

###
arframed <- data.frame(day = NYSE[-(1:5), "day_of_week"], arframe)
arfitd <- lm(log_volume ~ ., data = arframed[istrain, ])
arpredd <- predict(arfitd, arframed[!istrain, ])
1 - mean((arpredd - arframe[!istrain, "log_volume"])^2) / V0

###
n <- nrow(arframe)
xrnn <- data.matrix(arframe[, -1])
xrnn <- array(xrnn, c(n, 3, 5))
xrnn <- xrnn[,, 5:1]
xrnn <- aperm(xrnn, c(1, 3, 2))
dim(xrnn)

###
model <- keras_model_sequential(input_shape = list(5, 3)) |>
  layer_simple_rnn(units = 12, dropout = 0.1, recurrent_dropout = 0.1) |>
  layer_dense(units = 1)

compile(model, optimizer = optimizer_rmsprop(), loss = "mse")

history <- fit(model, xrnn[istrain,, ], arframe[istrain, "log_volume"],
               batch_size = 64, epochs = 200,
               validation_data =
                 list(xrnn[!istrain,, ], arframe[!istrain, "log_volume"]))

kpred <- predict(model, xrnn[!istrain,, ])
1 - mean((kpred - arframe[!istrain, "log_volume"])^2) / V0

###
model <- keras_model_sequential(input_shape = c(5, 3)) |>
  layer_flatten() |>
  layer_dense(units = 1)

x <- model.matrix(log_volume ~ . - 1, data = arframed)
colnames(x)

arnnd <- keras_model_sequential(input_shape = ncol(x)) |>
  layer_dense(units = 32, activation = 'relu') |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 1)

compile(arnnd, loss = "mse",
        optimizer = optimizer_rmsprop())

history <- fit(arnnd, x[istrain, ], arframe[istrain, "log_volume"], 
               epochs = 100, batch_size = 32, validation_data =
                 list(x[!istrain, ], arframe[!istrain, "log_volume"]))

npred <- predict(arnnd, x[!istrain, ])
1 - mean((arframe[!istrain, "log_volume"] - npred)^2) / V0

