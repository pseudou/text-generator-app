from Model import model
from Settings import settings as default
import os

if __name__=="__main__":

    models = model.load_model(path=os.path.join(default.WEIGHTS_PATH,'shakespeare_epoch(25)_loss(4.9277167320251465).pth'))
    models.to(default.DEVICE)
    print(model.generate_text(models,initial_text="in sooth",length=400))