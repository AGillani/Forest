import numpy as np
import math
from Smooth import smooth
import pandas

def pre_data(batch_size, seq_len):
    '''
    Prepare data for training and testing

    :batch_size: batch size for training/testing, type: int
    :seq_len: sequence length, type: int
    :return: (encoder input, expected decoder output), type: tuple, shape: [seq_len, batch_size, out_dim]
    '''
    # converting mean values from the CSV file into a smoothened numpy array
    data_dir = "/home/amna/Downloads/convertcsv.csv"
    df = pandas.read_csv(data_dir, converters={"value_mean": float})
    df = df["value_mean"]
    df = df.values
    # df = np.reshape(df, (-1, 424))
    df = smooth(df, window_len=5)

    X_batch = []
    Y_batch = []

    for _ in range(batch_size):
        offset = np.random.random_sample() * math.pi
        t = np.linspace(offset, offset + 4 * math.pi, 2 * seq_len)
        # seq_1 = np.sin(t)
        seq_1 = df
        seq_2 = np.cos(t)

        x1 = seq_1[:seq_len]
        y1 = seq_1[seq_len:]
        x2 = seq_2[:seq_len]
        y2 = seq_2[seq_len:]

        X = np.array([x1])  # size: [out_dim, seq_len]

        Y = np.array([y1])

        X = X.T  # size: [seq_len, out_dim]
        Y = Y.T


        X_batch.append(X)
        Y_batch.append(Y)



    X_batch = np.array(X_batch)  # size: [batch_size, seq_len, out_dim]
    Y_batch = np.array(Y_batch)
    print ("X batch", X_batch.shape)
    print("Y batch", Y_batch.shape)
    X_batch = np.transpose(X_batch, (1, 0, 2))  # size: [seq_len, batch_size, out_dim]
    Y_batch = np.transpose(Y_batch, (1, 0, 2))


    return X_batch, Y_batch