import tensorflow as tf
from tensorflow.keras import models, Model
from stellargraph.mapper import PaddedGraphGenerator

def load_model(path):
    model = models.load_model(path)
    return model

def predict_graph_classification(sg_graphs,model)