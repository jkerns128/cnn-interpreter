import tkinter as tk
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
import random
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import pathlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize


CNN = keras.models.load_model("model")
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

replace2linear = ReplaceToLinear()

saliency = Saliency(CNN,
                    model_modifier=replace2linear,
                    clone=True)

root = tk.Tk()
root.title('Image Segmenter')
windowSize = (1280,720)
root.geometry('{}x{}'.format(windowSize[0],windowSize[1]))

currentImage = None
predLabel = tk.StringVar()
clicked = StringVar()
sigma = 0

def inputImage():
    global currentImage
    
    imgCanvas.delete('all')
    modelImageCanvas.delete('all')
    predLabel.set(' ')
    
    ## Input of image and conversion to multiple formats
    file = tk.filedialog.askopenfilename(initialdir=pathlib.Path(__file__).parent.resolve() , title="Select Image", filetypes=(("JPG","*.jpg"),("PNG","*.png")))

    image = rescaleImage(Image.open(file), int(windowSize[1]*0.9))
    photoImage = ImageTk.PhotoImage(image)
    root.geometry('{}x{}'.format(700+image.width, windowSize[1]))


    imageSize32 = image.resize((32,32)).convert("RGB")

    currentImage = [image, photoImage, imageSize32, None, None]

    imgCanvas.create_image(0, 0, image=currentImage[1], anchor='nw')
    updateModelImg(None)
    return

## Preserves w:h ratio
def rescaleImage(image, desiredheight):
    scaleFactor = desiredheight/image.height
    image = image.resize((int(scaleFactor*image.width),desiredheight))
    return image

## Gaussian noise for one pixel
def noisePixel(pixel):
    return min(max(0, pixel+random.normalvariate(0,sigma)), 255)

def preprocess(image):
    return np.array([image.getdata()]).reshape(1,32,32,3)/255

def predict(image):
    prediction = np.argmax(CNN.predict(preprocess(image)))
    predLabel.set("Predicted Label: {}".format(classes[prediction]))
    return

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def updateModelImg(event):
    global sigma
    sigma = noiseScale.get()

    newModelImage = Image.eval(currentImage[2], noisePixel)
    photoImageSize32 = ImageTk.PhotoImage(rescaleImage(newModelImage, int(windowSize[1]*0.25)))

    modelImageCanvas.delete('all')
    interpretationCanvas.delete('all')
    modelImageCanvas.create_image(0, 0, image=photoImageSize32, anchor='nw')
    
    currentImage[3] = photoImageSize32
    
    label = clicked.get()
    if(label != '' ):
        index = classes.index(label)
        saliencyMap = saliency(CategoricalScore(index),
                            preprocess(newModelImage).reshape(32,32,3),
                            smooth_samples=20, # The number of calculating gradients iterations.
                            smooth_noise=0.20) # noise spread level.

        f, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(saliencyMap.reshape(32,32), cmap='jet')
        ax.axis('off')
        plt.tight_layout()

        img = fig2img(f)

        saliencyImg = ImageTk.PhotoImage(rescaleImage(img, int(windowSize[1]*0.25)))
        interpretationCanvas.create_image(0,0, image=saliencyImg, anchor='nw')
        currentImage[4] = saliencyImg
    
    

    predict(newModelImage)

    return

def loadMainFrame():
    mainFrame.place(relwidth=1, relheight=1)
    imgCanvas.place(relwidth=0.6, relheight=0.9, relx=0.05,rely=0.05)
    mainButtonFrame.place(relwidth=.25, relheight=.25, relx=.7, rely=.65)
    modelImageCanvas.place(relwidth=.25, relheight=.25, relx=.7, rely=.05)
    interpretationCanvas.place(relwidth=.25, relheight=.25, relx=.7, rely=.35)

    
    modelLabel.place(relwidth=1, relheight=.25)
    drop.place(relwidth=1, relheight=.25,rely=.25)
    noiseScale.place(relwidth=1, relheight=.25,rely=.5)
    inputImageButton.place(relwidth= 1, relheight=.25, rely=.75)
    return


mainFrame = tk.Frame(root, bg="#353542")
imgCanvas = tk.Canvas(mainFrame, bg='#353542', highlightthickness=0)
mainButtonFrame = tk.Frame(mainFrame, bg='#353542')

modelImageCanvas=tk.Canvas(mainFrame, bg='#353542', highlightthickness=0)
interpretationCanvas=tk.Canvas(mainFrame, bg='#353542', highlightthickness=0)

modelLabel = tk.Label(mainButtonFrame, textvariable=predLabel, fg="white", bg='#353542', font=('Arial 20'))

inputImageButton=tk.Button(mainButtonFrame, text="Open Image", fg="white", bg="#353542", command=inputImage)
noiseScale = tk.Scale(mainButtonFrame, from_=0, to=200, orient=HORIZONTAL)
noiseScale.bind("<ButtonRelease-1>", updateModelImg)
drop = OptionMenu(mainButtonFrame, clicked, *classes)
drop.bind("<Configure>", updateModelImg)


loadMainFrame()
root.resizable(False,False)
root.mainloop()