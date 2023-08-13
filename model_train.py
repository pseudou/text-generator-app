'''
training model
'''

from Model import model
from Settings import settings as default
from Settings import dataset

if __name__=="__main__":
    models = model.TextGenerator(default.EMBEDDING_DIM,default.HIDDEN_LAYER_DIM,default.VOCAB_SIZE)
    models.to(default.DEVICE)
    models = model.train(models,25)