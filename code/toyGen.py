import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import random
import dataPipeline as DP
import os
import librosa as lb
import soundfile as sf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

class toyGan(models.Model):
    def __init__(self,dimensions,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dimensions
        self.genOpt = Adam(learning_rate=0.0001) 
        self.discOpt = Adam(learning_rate=0.00001) 
        self.genLoss = BinaryCrossentropy()
        self.discLoss = BinaryCrossentropy()
        self.channels=3
        self.generator()
        self.discriminator()
        

      
        return
    def generator(self):
        self.gen= models.Sequential()
        
        self.gen.add(layers.Dense(323*32*128, input_dim=128))
        self.gen.add(layers.LeakyReLU(0.2))
        self.gen.add(layers.Reshape((323,32,128)))
        
       
        self.gen.add(layers.UpSampling2D())
        self.gen.add(layers.Conv2D(128, 5, padding='same'))
        self.gen.add(layers.LeakyReLU(0.2))

        
         
        self.gen.add(layers.UpSampling2D())
        self.gen.add(layers.Conv2D(128, 5, padding='same'))
        self.gen.add(layers.LeakyReLU(0.2))
        
        
        self.gen.add(layers.Conv2D(128, 4, padding='same'))
        self.gen.add(layers.LeakyReLU(0.2))
        
        
        self.gen.add(layers.Conv2D(128, 4, padding='same'))
        self.gen.add(layers.LeakyReLU(0.2))
        
        
        self.gen.add(layers.Conv2D(1, 4, padding='same', activation='sigmoid'))


        return "gen built"
    
    def discriminator(self):
        self.disc = models.Sequential()
        
        self.disc.add(layers.Conv2D(32, 5, input_shape = (1291,128,1)))
        self.disc.add(layers.LeakyReLU(0.2))
        self.disc.add(layers.Dropout(0.4))
  
        self.disc.add(layers.Conv2D(64, 5))
        self.disc.add(layers.LeakyReLU(0.2))
        self.disc.add(layers.Dropout(0.4))
        
        self.disc.add(layers.Conv2D(128, 5))
        self.disc.add(layers.LeakyReLU(0.2))
        self.disc.add(layers.Dropout(0.4))
      
        self.disc.add(layers.Conv2D(256, 5))
        self.disc.add(layers.LeakyReLU(0.2))
        self.disc.add(layers.Dropout(0.4))
       
        self.disc.add(layers.Flatten())
        self.disc.add(layers.Dropout(0.4))
        self.disc.add(layers.Dense(1, activation='sigmoid'))

        
        return "disc built"
    def train_step(self,batch):
        real =batch
        fake= self.gen(tf.random.normal((1,128)),training=False)
        #print(fake)
        #print(fake.shape)
        fake=DP.normalizeDim2(self.dim,fake)
        print(fake.shape)
        #print(type(real)[0][0])
        #print(real.shape)
        #new_shape = (-1, 1292, 128, 1)  # Use -1 to automatically infer the batch size
        #reshapedFake = tf.reshape(fake, new_shape)

        with tf.GradientTape() as Gtape:
            yHreal=self.disc(real[0][0][0],training=True)
            yHfalse=self.disc(fake,training=True)
            yHrf=tf.concat([yHreal,yHfalse],axis=0)

            yRF=tf.concat([tf.zeros_like(yHreal), tf.ones_like(yHfalse)],axis=0)

            realNoise=0.15*tf.random.uniform(tf.shape(yHreal))
            falseNoise= -0.15*tf.random.uniform(tf.shape(yHfalse))
            yRF+= tf.concat([realNoise,falseNoise],axis=0)

            dLoss=self.discLoss(yRF,yHrf)
        Gdisc=Gtape.gradient(dLoss,self.disc.trainable_variables)
        self.discOpt.apply_gradients(zip(Gdisc,self.disc.trainable_variables))

        with tf.GradientTape() as dTape:
            newIm=self.gen(tf.random.normal((1,128)), training=True)
            newIm=DP.normalizeDim2(self.dim,newIm)
            preds= self.disc(newIm,training=False)
            gLoss=self.genLoss(tf.zeros_like(preds),preds)
        Ggen=dTape.gradient(gLoss,self.gen.trainable_variables)
        self.genOpt.apply_gradients(zip(Ggen,self.gen.trainable_variables))

        return {"discLoss": dLoss, "genLoss":gLoss}
    






    def compile(self,*args, **kwargs):
        super().compile(*args, **kwargs)
        return
    def loadModel(self,modelPath):
        self.gen = tf.saved_model.load(os.path.join(modelPath,"genModel"))
        self.disc = tf.saved_model.load(os.path.join(modelPath,"discModel"))
        return "loaded"
    def generate(self,frequency=1000,filename="generated.wav"):
        noise=(tf.random.normal((1,128)))
        generated=self.gen(noise)
        generated=np.squeeze(generated, axis=3)
        recSignal = lb.feature.inverse.mel_to_audio(generated[0])
        sf.write(filename, recSignal, frequency, 'PCM_24')
        return recSignal,filename
    def save(self,folderPath):
    
        self.gen.save(os.path.join(folderPath,"genModel"))
        self.disc.save(os.path.join(folderPath,"discModel"))
 
        return "saved"



def main():
    inShape=(1,1291, 128,1)
    return "done"

main()