
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        lookup=[10,100,20,200,2000,50,500]
        idg = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        img = Image.open(imagename).convert("RGB").resize((150,150))
        img = np.array(img)
        img = idg.flow(np.array([img]))
        predict = model.predict(img)
        v=np.argmax(predict)
        return {"image":lookup[v]}


