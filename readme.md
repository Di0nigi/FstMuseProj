Link to the drive to download the pretrained models: 

https://drive.google.com/drive/folders/1xeryoNxjx_beycPs8HLHYR_Hj3Z8n3gZ?usp=sharing

Libraries needed:

import pandas as pd
import numpy as np
import librosa as lb
import os
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import random 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import tensorflow_io as tfio
from scipy.ndimage import zoom
import tensorflow as tf
import soundfile as sf
import tkinter as tk 
from tkinter import filedialog