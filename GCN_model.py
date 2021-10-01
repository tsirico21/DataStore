import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph
from sklearn import model_selection
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def binary_classification_GCN(sg_graphs, graph_labels):
    generator = PaddedGraphGenerator(graphs=sg_graphs)

    gc_model = GCNSupervisedGraphClassification(
        layer_sizes = [64,64],
        activations=['relu','relu'],
        generator=generator,
        dropout=0.5,
    )
    
    x_inp, x_out = gc_model.in_out_tensors()

    predictions = Dense(units=32, activation='relu')(x_out)
    predictions = Dense(units=16, activation='relu')(predictions)
    predictions = Dense(units=1, activation='sigmoid')(predictions)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=['acc'])
    
    es = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=25, restore_best_weights=True
    )
    
def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen

    
    