import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# =================================================================================
# PARTE 1: Carregamento dos dados e pré-processamento
# =================================================================================

# --- Carregando o conjunto de dados MNIST no Keras ---
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digits = test_images.copy() # Criando a variável digits para a função compare_prediction

# --- Covertendo os tensores do MNIST em matrizes e normalizando os valores na escala 0-250 ---
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255


# =================================================================================
# PARTE 2: Treinamento do modelo baseado em uma rede neural densamente conectada
# =================================================================================

import keras
from keras import layers

# --- Parâmetros do treinamento ---
batch_size = 256
epochs = 480

# --- Arquitetura da rede ---
model = keras.Sequential(
    [
      layers.Dense(1024, activation="relu"),
      layers.Dropout(0.6),
      layers.Dense(512, activation="relu"),
      layers.Dropout(0.4),
      layers.Dense(256, activation="relu"),
      layers.Dense(10, activation="softmax"),
    ]
)

# --- Técnicas de regularização ---
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=40, # se não houver redução da função loss no conjunto de validação após 40 épocas, o treinamento é interrompido
    restore_best_weights=True # Restaura os pesos para àqueles encontrados na época que atingiu o menor val_loss
)

model_checkpoint = ModelCheckpoint(
    filepath=f"mnist_dense_b{batch_size}e{epochs}_last.keras", # Salvando o modelo 
    monitor="val_loss",
    save_best_only=True, # Se o treinamento for executado para todas as épocas, restaura os pesos da época com menor val_loss
)

# --- Compilando o modelo ---
optimizer = Adam(learning_rate=0.0005) # Especificando a taxa de aprendizado
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy", # Função de perda para mais de dois labels
    metrics=["accuracy"], # Métrica para avaliar as classificações do conjunto de treinamento e validação
)

# --- Iniciando o treinamento ---
history = model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2, # Separando 20% dos dados do treinamento para a validação do modelo
    callbacks = [model_checkpoint, early_stop] # -> early_stop: opcional caso queira interromper o treinamento
)


# --- Carregando o modelo salvo em model_checkpoint ---
model = keras.models.load_model(f"mnist_dense_b{batch_size}e{epochs}_last.keras")


# ===================================================================================
# PARTE 3: Avaliação do modelo
# ===================================================================================

# --- Avaliando o modelo com os dados em test_images ---
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

# --- Gráficos  ---
history_dict = history.history # Acessando a perda e acurácia armazenados no objeto history

# Plotando o gráfico da perda
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r--", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("[MNIST] Training and validation loss")
plt.xlabel("Epochs")
plt.xticks()
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotando o gráfico da acurácia
loss_values = history_dict["accuracy"]
val_loss_values = history_dict["val_accuracy"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r--", label="Training accuracy")
plt.plot(epochs, val_loss_values, "b", label="Validation accuracy")
plt.title("[MNIST] Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks()
plt.ylabel("Accuracy")
plt.legend()
plt.show()


import random

def compare_prediction(idx):
    image_predict = test_images[idx]
    prediction = model.predict(image_predict[np.newaxis, :])
    predicted_digit = np.argmax(prediction)
    actual_digit = test_labels[idx]

    print(f"Model Prediction: {predicted_digit}")
    print(f"Actual Label: {actual_digit}")

    digit = digits[idx]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()

digit = random.randint(0, 9999)
compare_prediction(digit)