import tensorflow as tf
import keras.layers as L
from keras.models import Sequential, Model

from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler


def get_reconstruction_segment(model, values, start, end):
    """
    Автокодировщик model получает на вход сигнал values и
    возвращает реконструированные (декодированные) значения
    для заданного сегмента [start, end].
    Поскольку автокодировщик требует постоянное число сэмплов на входе, то для
    последнего набора данных берется сегмент [end-WINWOW_SIZE, end].
    """
    num = int((end - start) / WINDOW_SIZE)
    data = []
    left, right = 0, WINDOW_SIZE

    for i in range(num):
        result = model.predict(np.array(values[left:right]).reshape(1, -1))
        data = np.r_[data, result[0]]
        left += WINDOW_SIZE
        right += WINDOW_SIZE

    if left < end:
        result = model.predict(np.array(values[end - WINDOW_SIZE:end]).reshape(1, -1)).reshape(-1, 1)
        data = np.r_[data, result[-end + left:].squeeze()]

    return np.array(data)


def draw_reconstruction_segment(xs, ys, ys_hat, start, end, c='g'):
    """
    Рисует истинные и предсказанные значения для заданного сегмента
    """
    plt.figure(figsize=(20, 5))
    plt.plot(xs[start:end], ys[start:end], c='b', alpha=0.7, label="Исходный сигнал")
    plt.scatter(xs[start:end], ys_hat[start:end], c=c, label="Восстановленный сигнал")

    plt.legend()
    plt.show()

def draw_losses(aeLoss, cxLoss, czLoss, cx_g_Loss, cz_g_Loss):

    plt.figure(figsize=(20, 10))
    plt.plot(aeLoss, label="aeLoss")
    plt.plot(cxLoss, label="cxLoss")
    plt.plot(czLoss, label="czLoss")
    plt.plot(cx_g_Loss, label="cx_g_Loss")
    plt.plot(cz_g_Loss, label="cz_g_Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


LATENT_VECTOR_SIZE = 10
WINDOW_SIZE = 100


def init_model():
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

    E = Sequential()
    E.add(L.InputLayer(WINDOW_SIZE))
    E.add(L.Dense(WINDOW_SIZE * 2, activation='relu'))
    E.add(L.Dropout(0.3))
    E.add(L.Dense(WINDOW_SIZE, activation='relu'))
    E.add(L.Dropout(0.3))
    E.add(L.Dense(LATENT_VECTOR_SIZE))

    G = Sequential()
    G.add(L.Dense(128, input_dim=LATENT_VECTOR_SIZE))
    G.add(L.LeakyReLU(0.2))
    G.add(L.Dropout(0.3))
    G.add(L.Dense(256))
    G.add(L.LeakyReLU(0.2))
    G.add(L.Dropout(0.3))
    G.add(L.Dense(512))
    G.add(L.LeakyReLU(0.2))
    G.add(L.Dropout(0.3))
    G.add(L.Dense(WINDOW_SIZE, activation='tanh'))
    G.compile(loss='binary_crossentropy', optimizer=adam)

    Cx = Sequential()
    Cx.add(L.Dense(1024, input_dim=WINDOW_SIZE))
    Cx.add(L.LeakyReLU(0.2))
    Cx.add(L.Dropout(0.3))
    Cx.add(L.Dense(512, input_dim=WINDOW_SIZE))
    Cx.add(L.LeakyReLU(0.2))
    Cx.add(L.Dropout(0.3))
    Cx.add(L.Dense(256))
    Cx.add(L.LeakyReLU(0.2))
    Cx.add(L.Dropout(0.3))
    Cx.add(L.Dense(1, activation='sigmoid'))
    Cx.compile(loss='binary_crossentropy', optimizer=adam)

    Cz = Sequential()
    Cz.add(L.Dense(1024, input_dim=LATENT_VECTOR_SIZE))
    Cz.add(L.LeakyReLU(0.2))
    Cz.add(L.Dropout(0.3))
    Cz.add(L.Dense(512))
    Cz.add(L.LeakyReLU(0.2))
    Cz.add(L.Dropout(0.3))
    Cz.add(L.Dense(256))
    Cz.add(L.LeakyReLU(0.2))
    Cz.add(L.Dropout(0.3))
    Cz.add(L.Dense(1, activation='sigmoid'))
    Cz.compile(loss='binary_crossentropy', optimizer=adam)

    ae_input = L.Input(WINDOW_SIZE)
    ae_code = E(ae_input)
    ae_reconstruction = G(ae_code)
    ae_model = Model(inputs=ae_input, outputs=ae_reconstruction)
    ae_model.compile(loss='mse', optimizer=adam)

    cx_gan_input = L.Input(LATENT_VECTOR_SIZE)
    cx_gan_code = G(cx_gan_input)
    cx_gan_output = Cx(cx_gan_code)
    cx_gan_model = Model(inputs=cx_gan_input, outputs=cx_gan_output)
    cx_gan_model.compile(loss='binary_crossentropy', optimizer=adam)

    cz_gan_input = L.Input(WINDOW_SIZE)
    cz_gan_code = E(cz_gan_input)
    cz_gan_output = Cz(cz_gan_code)
    cz_gan_model = Model(inputs=cz_gan_input, outputs=cz_gan_output)
    cz_gan_model.compile(loss='binary_crossentropy', optimizer=adam)

    return E, G, Cx, Cz, ae_model, cx_gan_model, cz_gan_model


def train_model(X_train, epochs=1, batch_size=128):
    batchCount = int(X_train.shape[0] / batch_size)

    for epoch in range(1, epochs + 1):
        print("-" * 10, "Epoch: {}, batchCount {}".format(epoch, batchCount), "-" * 10)

        for _ in range(batchCount):

            # обучение дискриминатора Cx
            Cx.trainable = True
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            fake = G.predict(np.random.normal(0, 1, size=(batch_size, LATENT_VECTOR_SIZE)))
            X = []
            for i in idxs:
                X.append(X_train[i:i + WINDOW_SIZE])
            X = np.r_[X, fake]
            labels = np.r_[np.ones(shape=batch_size) * 0.95, np.zeros(shape=batch_size)]
            cx_loss = Cx.train_on_batch(X, labels)

            # обучение генератора cx_gan_model обманывать дискриминатор Cx
            Cx.trainable = False
            labels = np.ones(shape=batch_size)
            X = np.random.normal(0, 1, size=(batch_size, LATENT_VECTOR_SIZE))
            cx_g_loss = cx_gan_model.train_on_batch(X, labels)

            # обучение дискриминатора Cz
            Cz.trainable = True
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            fake = np.array([X_train[i:i + WINDOW_SIZE] for i in idxs])
            fake = E.predict(fake)
            X = np.random.normal(0, 1, size=(batch_size, LATENT_VECTOR_SIZE))
            X = np.r_[X, fake]
            labels = np.r_[np.ones(shape=batch_size) * 0.95, np.zeros(shape=batch_size)]
            cz_loss = Cz.train_on_batch(X, labels)

            # обучение генератора cx_gan_model обманывать дискриминатор Cz
            Cz.trainable = False
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            X = []
            for i in idxs:
                X.append(X_train[i:i + WINDOW_SIZE])
            X = np.array(X)
            labels = np.ones(shape=batch_size)
            cz_g_loss = cz_gan_model.train_on_batch(X, labels)

            # обучение автокодировщика AE
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            X = []
            for i in idxs:
                X.append(X_train[i:i + WINDOW_SIZE])
            E.trainable = True
            G.trainable = True
            X = np.array(X)
            ae_loss = ae_model.train_on_batch(X, X)

        # оценка ошибок и периодическое сохранение картинок
        aeLoss.append(ae_loss)
        cxLoss.append(cx_loss)
        czLoss.append(cz_loss)
        cx_g_Loss.append(cx_g_loss)
        cz_g_Loss.append(cz_g_loss)

        if epoch % 4 == 0:
            print("Эпоха {}".format(epoch))
            print("ae_loss {}, cx_loss {}, cz_loss {}, cx_g_loss {}, cz_g_loss {}".format(ae_loss,
                                                                                          cx_loss,
                                                                                          cz_loss,
                                                                                          cx_g_loss,
                                                                                          cz_g_loss))


# простой сигнал без аномалий, чтобы посмотреть насколько хорошо он будет коррелировать восстановиленный сигнал с исходным
xs = np.linspace(0, 1000, 10000)
ys1 = 0.5*np.sin(xs) + 0.5*np.sin(0.8*xs)
label1 = np.zeros(10000)

aeLoss = []
cxLoss = []
czLoss = []
cx_g_Loss = []
cz_g_Loss = []

E, G, Cx, Cz, ae_model, cx_gan_model, cz_gan_model = init_model()


train_model(ys1, 4, 128)
r1 = get_reconstruction_segment(ae_model, ys1, 0, len(ys1))

train_model(ys1, 16, 128)
r2 = get_reconstruction_segment(ae_model, ys1, 0, len(ys1))

train_model(ys1, 80, 128)
r3 = get_reconstruction_segment(ae_model, ys1, 0, len(ys1))



draw_losses(aeLoss, cxLoss, czLoss, cx_g_Loss, cz_g_Loss)

l, r = 0, 1000
plt.figure(figsize=(20,5))
plt.plot(xs[l:r], ys1[l:r], c='b', label='Истинные значения')
plt.scatter(xs[l:r], r1[l:r], c='r', label='4 итерации')
plt.scatter(xs[l:r], r2[l:r], c='y', label='16 итераций')
plt.scatter(xs[l:r], r3[l:r], c='g', label='80 итераций')
plt.legend()
plt.show()