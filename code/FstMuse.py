import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import random
import dataPipeline as DP

from tensorflow.keras import layers, models, regularizers
import librosa as lb








class fstMuse:
    def __init__(self,dim,classes):
        self.shapes=dim
        self.loaded=None
        reg=0
        #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (9, 9), activation='relu', input_shape=(dim[0], dim[1], 1)))#,kernel_regularizer=regularizers.l2(reg)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))#,kernel_regularizer=regularizers.l2(reg)))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(64, (9, 9), activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((4, 4)))

        self.model.add(layers.Conv2D(64, (9, 9), activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        #self.model.add(layers.Dropout(0.2))
        #self.model.add(layers.Conv2D(128, (2, 2), activation='relu'))
        

        #self.model.add(layers.MaxPooling2D((2, 2)))
        
        
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(32, activation='relu'))#,kernel_regularizer=regularizers.l2(reg)))    
        #self.model.add(layers.Dense(16, activation='relu'))    
        self.model.add(layers.Dense(classes, activation='softmax')) 

        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return
    def train(self,data, epochs=10):
        ret = self.model.fit(tf.stack(data[0]), tf.stack(data[1]), epochs=epochs, validation_data=(tf.stack(data[2]), tf.stack(data[3])))
        return ret
        ''' p=int(len(data)*0.2)
        random.shuffle(data)
        labels=[]
        images=[]
        #print(type(data))
        for ind, elem in enumerate(data):
            im=elem.toMelSpectroGram(mels=128)
            #np.resize(im,inShape)
            im=DP.normalizeDim(self.shapes,im)
            #if im.shape!=(1291,128):
             #   c=np.abs(im.shape[0]-self.shapes[0])
              #  im= im[:-c, :]
            if elem.label=="folk":
                labels.append(np.array([1,0,0]))
            elif elem.label=="hiphop":
                labels.append(np.array([0,1,0]))
            else:
                labels.append(np.array([0,0,1]))
            #print(im.shape)
                #print(elem.title)
                #name=elem.save(path="assets\\buffer",mode="mel")
            images.append(im)
        validationDataX=tf.stack(images[:p])
        validationDataY=tf.stack(labels[:p])
        trainingDataX=tf.stack(images[p:])
        trainingDataY=tf.stack(labels[p:])'''
        
    
    def predict(self,data):
        if self.loaded!=None:
            labels=[]
            datain=[]
            inference=self.loaded.signatures['serving_default']
            for elem in data:
                im=elem.toMelSpectroGram(mels=128)
                im=DP.normalizeDim(self.shapes,im)
                datain.append(np.expand_dims(im, axis=-1))
            predictions = inference(tf.stack(datain))

            denseOutputTensor = predictions['dense_1']

            denseOutputArray = denseOutputTensor.numpy()

            predictedIndices = np.argmax(denseOutputArray, axis=1)
            classLabels = ["pop","jazz","rock","folk","hiphop","punk","electronic"] 
            predictedLabels = [classLabels[i] for i in predictedIndices]
            return predictedLabels
        else: 
            print("model not initialized")
            return []

    def save(self,filename):
        self.model.save(filename)
        return filename
    def load(self,name):
        self.loaded = tf.saved_model.load(name)
        return
    
    


def main():
    classes=7
    dims=(1291, 128)
    genres=["pop","jazz","rock","folk","hiphop","punk","electronic"]
    return "done"

main()