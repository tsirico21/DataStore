from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing

def perf_classification_DGCNN(sg_graphs, graph_labels, lr, batch):
    generator = PaddedGraphGenerator(graphs=sg_graphs)
    
    k = 35
    
    layer_sizes = [32,32,32,1]
    
    model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=['relu','relu','relu','relu'],
        k=k,
        bias=False,
        generator=generator,
    )
    
    x_inp, x_out = model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=128, activation="relu")(x_out)
    x_out = Dropout(rate=0.5)(x_out)

    predictions = Dense(units=1, activation="relu")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(
        optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=["acc"],
    )
    
    train_graphs, test_graphs = model_selection.train_test_split(
        graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
    )
    
    gen = PaddedGraphGenerator(graphs=sg_graphs)

    train_gen = gen.flow(
        list(train_graphs.index - 1),
        targets=train_graphs.values,
        batch_size=10,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        list(test_graphs.index - 1),
        targets=test_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )
    
    return model, train_gen, test_gen

def binary_classification_DGCNN(sg_graphs, graph_labels, lr, batch):
    generator = PaddedGraphGenerator(graphs=sg_graphs)
    
    k = 35
    
    layer_sizes = [32,32,32,1]
    
    model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=['tanh','tanh','tanh','tanh'],
        k=k,
        bias=False,
        generator=generator,
    )
    
    x_inp, x_out = model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=128, activation="relu")(x_out)
    x_out = Dropout(rate=0.5)(x_out)

    predictions = Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(
        optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
    )
    
    train_graphs, test_graphs = model_selection.train_test_split(
        graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
    )
    
    gen = PaddedGraphGenerator(graphs=sg_graphs)

    train_gen = gen.flow(
        list(train_graphs.index - 1),
        targets=train_graphs.values,
        batch_size=batch,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        list(test_graphs.index - 1),
        targets=test_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )
    
    return model, train_gen, test_gen