import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import defaultdict

def get_mapping(series):
    occurances = defaultdict(int)
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i

    return mapping




def get_data():
    data = pd.read_csv("data/ratings.csv")

    mapping_work = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_work)

    mapping_users = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_users)

    percentil_80 = np.percentile(data["timestamp"], 80)

    print(percentil_80)

    print(np.mean(data["timestamp"]<percentil_80))

    print(np.mean(data["timestamp"]>percentil_80))

    cols = ["userId", "movieId", "rating"]

    train = data[data.timestamp<percentil_80][cols]

    print(train.shape)

    test = data[data.timestamp>=percentil_80][cols]

    print(test.shape)

    max_user = max(data["userId"].tolist() )
    max_work = max(data["movieId"].tolist() )


    return train, test, max_user, max_work




def get_model_1(max_work, max_user):
    dim_embedddings = 30
    bias = 3
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model


def get_model_2(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = concatenate([o, u_bis, w_bis])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_model_3(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = Dense(10, activation="relu")(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])

