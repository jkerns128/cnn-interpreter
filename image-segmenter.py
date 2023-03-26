import tkinter as tk
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
import random
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import pathlib


# make box for 32x32 image in top right corner with predicted label

CNN = keras.models.load_model("model")
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

root = tk.Tk()
root.title('Image Segmenter')
windowSize = (1280,720)
root.geometry('{}x{}'.format(windowSize[0],windowSize[1]))

images = []
distortedImages = []
segmentations = []
pointList = []
lineList = []

currentImage = None

imagesCompleted = 0
imageNo = 0
labelsCorrect = 0

drawingSegment=False
showRLines = True

plabel = tk.StringVar()
mlabel = tk.StringVar()
rlabel = tk.StringVar()
predLabel = tk.StringVar()
fractionLabel = tk.StringVar()
checkboxVar = tk.IntVar()

    
#=========== Page Loads/Transitions ===========

#Page Loads
def loadInputFrame():
    inputFrame.place(relwidth=1, relheight=1)
    previewCanvas.place(relwidth=0.6, relheight=0.9, relx=0.05,rely=0.05)
    inputButtonFrame.place(relwidth=.25, relheight=.2, relx=.7, rely=.4)

    inputImageButton.place(relwidth= .33, relheight=.5, rely=.5)
    addImageButton.place(relwidth= .34, relheight=.5, rely=.5, relx=.33)
    finishInputButton.place(relwidth= .33, relheight=.5, rely=.5, relx=.67)
    previewLabel.place(relwidth= 1, relheight=.5)
    return

def loadMainFrame():
    mainFrame.place(relwidth=1, relheight=1)
    imgCanvas.place(relwidth=0.6, relheight=0.9, relx=0.05,rely=0.05)
    mainButtonFrame.place(relwidth=.25, relheight=.6, relx=.7, rely=.35)
    modelImageCanvas.place(relwidth=.25, relheight=.25, relx=.7, rely=.05)

    modelLabel.place(relwidth=1, relheight=.15)
    labelCheckbox.place(relwidth=1, relheight=.3,rely=.15)
    mainLabel.place(relwidth= 1, relheight=.15, rely=.45)
    drawSeg.place(relwidth= .5, relheight=.2, rely=.6)
    endSeg.place_forget()
    clear.place(relwidth= .5, relheight=.2, relx=.5, rely=.6)
    submit.place(relwidth= 1, relheight=.2, rely=.8)
    resultsButton.place_forget()
    return

def loadResultsFrame():
    resultsFrame.place(relwidth=1, relheight=1)
    resultsCanvas.place(relwidth=0.6, relheight=0.9, relx=0.05,rely=0.05)
    resultsButtonFrame.place(relwidth=.25, relheight=.3, relx=.7, rely=.4)

    previousImageButton.place(relwidth= .5, relheight=.25, rely=.25)
    nextImageButton.place(relwidth= .5, relheight=.25, rely=.25, relx=.5)
    beginningButton.place(relwidth= .5, relheight=.25, rely=.5, relx=.5)
    hideLinesButton.place(relwidth= .5, relheight=.25, rely=.5)
    resultsLabel.place(relwidth= 1, relheight=.25)
    fractionCorrectLabel.place(relwidth=1, relheight=.25, rely=.75)
    fractionLabel.set('Model accuracy: {}/{}'.format(labelsCorrect,int(len(images)/2)))

    return

#Unloads

def unloadInputFrame():
    inputFrame.place_forget()
    previewCanvas.place_forget()
    inputButtonFrame.place_forget()

    inputImageButton.place_forget()
    addImageButton.place_forget()
    finishInputButton.place_forget()
    previewLabel.place_forget()
    previewCanvas.delete('all')
    plabel.set(' ')
    return

def unloadMainFrame():
    mainFrame.place_forget()
    imgCanvas.place_forget()
    mainButtonFrame.place_forget()
    clearLines()
    imgCanvas.delete('all')
    unloadMainButtons()
    
    return

def unloadMainButtons():
    modelLabel.place_forget()
    labelCheckbox.place_forget()
    mainLabel.place_forget()
    drawSeg.place_forget()
    submit.place_forget()
    clear.place_forget()
    endSeg.place_forget()
    mlabel.set(' ')
    predLabel.set(' ')
    return

def unloadResultsFrame():
    resultsFrame.place_forget()
    resultsCanvas.place_forget()
    resultsButtonFrame.place_forget()

    previousImageButton.place_forget()
    nextImageButton.place_forget()
    beginningButton.place_forget()
    hideLinesButton.place_forget()
    resultsLabel.place_forget()
    rlabel.set(' ')
    fractionLabel.set(' ')
    resultsCanvas.delete('all')
    return

#Transitions

def finishInput():
    global images
    if len(images) == 0:
        return
    images += distortedImages
    unloadInputFrame()
    loadMainFrame()
    loadImage(imgCanvas, imagesCompleted, mlabel, modelCanvas=modelImageCanvas)
    mlabel.set("Your Label: {}".format(mlabel.get()))
    return

def resultsTransition():
    global imageNo

    unloadMainFrame()
    loadResultsFrame()
    
    imageNo = 0
    loadImage(resultsCanvas, imageNo, rlabel, showRLines)
    return

def restartProgram():
    global images, distortedImages, segmentations, pointList, lineList
    global currentImage, imagesCompleted, imageNo, drawingSegment, showRLines, labelsCorrect
    global plabel, mlabel, rlabel, predLabel
    
    root.geometry('{}x{}'.format(windowSize[0],windowSize[1]))

    images = []
    distortedImages = []
    segmentations = []
    pointList = []
    lineList = []

    currentImage = None
    imagesCompleted = 0
    imageNo = 0
    labelsCorrect = 0
    drawingSegment=False
    showRLines = True

    plabel.set(' ')
    mlabel.set(' ')
    rlabel.set(' ')
    predLabel.set(' ')

    unloadResultsFrame()
    loadInputFrame()
    return

#=========== Image handlers ===========

def inputImage():
    global currentImage, plabel

    previewCanvas.delete('all')
    plabel.set(' ')
    file = tk.filedialog.askopenfilename(initialdir=pathlib.Path(__file__).parent.resolve() , title="Select Image", filetypes=(("JPG","*.jpg"),("PNG","*.png")))
    image = rescaleImage(Image.open(file), int(windowSize[1]*0.9))
    photoImage = ImageTk.PhotoImage(image)
    previewCanvas.create_image(0, 0, image=photoImage, anchor='nw')
    root.geometry('{}x{}'.format(700+image.width, windowSize[1]))
    label = tkinter.simpledialog.askstring(" ", "Label Image")
    if type(label) != str:
        label = ''

    imageSize32 = image.resize((32,32)).convert("RGB")
    photoImageSize32 = ImageTk.PhotoImage(rescaleImage(imageSize32, int(windowSize[1]*0.25))) #resized for canvas

    currentImage = (label, image, photoImage, imageSize32, photoImageSize32)
    plabel.set(label)
    return

def addImage():
    global currentImage, plabel
    
    if currentImage == None:
        return
    distortedImage = Image.eval(currentImage[1].effect_spread(25).convert("RGBA"), noisePixel)
    images.append(currentImage)
    distortedImages.append((currentImage[0], distortedImage, ImageTk.PhotoImage(distortedImage)))
    previewCanvas.delete('all')
    currentImage = None
    plabel.set(' ')
    return

def noisePixel(pixel):
    sigma = 50

    return min(max(0, pixel+random.normalvariate(0,sigma)), 255)

def loadImage(canvas, number, varlabel, loadLines=False, modelCanvas=None):
    global currentImage, mlabel, imagesCompleted

    canvas.delete('all')
    modelImageCanvas.delete('all')
    varlabel.set(' ')
    currentImage = images[number]
    canvas.create_image(0, 0, image=currentImage[2], anchor='nw')
    root.geometry('{}x{}'.format(700+currentImage[1].width, windowSize[1]))
    varlabel.set(currentImage[0])

    if modelCanvas != None:
        modelCanvas.create_image(0,0, image=currentImage[4], anchor='nw')
        prediction = np.argmax(CNN.predict(np.array([currentImage[3].getdata()]).reshape(1,32,32,3)/255))
        predLabel.set(classes[prediction])
        predLabel.set("Predicted Label: {}".format(predLabel.get()))

    if loadLines:
        points = segmentations[number][2]
        for i in range(len(points)-1):
            canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

    return

def nextImage():
    global imageNo
    imageNo += 1
    imageNo %= len(images)
    loadImage(resultsCanvas, imageNo, rlabel, showRLines)
    return

def prevImage():
    global imageNo
    imageNo -= 1
    imageNo %= len(images)
    loadImage(resultsCanvas, imageNo, rlabel, showRLines)
    return

def rescaleImage(image, desiredheight):
    scaleFactor = desiredheight/image.height
    image = image.resize((int(scaleFactor*image.width),desiredheight))
    return image

#Drawing Handlers

def drawSegment():
    global drawingSegment

    drawSeg.place_forget()
    drawingSegment = True
    endSeg.place(relwidth= .5, relheight=.2, rely=.6)
    return

def endSegment():
    global drawingSegment
    
    endSeg.place_forget()
    drawingSegment=False
    drawSeg.place(relwidth= .5, relheight=.2, rely=.6)
    return

def mouseClick(event):
    global pointList, lineList
    
    if drawingSegment:
        if pointList == []:
            pointList.append((event.x, event.y))
        else:
            pointList.append((event.x, event.y))
            newLine = imgCanvas.create_line(pointList[-2][0],pointList[-2][1],pointList[-1][0],pointList[-1][1])
            lineList.append(newLine)
    return

def clearLines():
    global lineList, pointList
    
    for line in lineList:
        imgCanvas.delete(line)
    lineList = []
    pointList = []
    return

def hideLines():
    global showRLines

    showRLines = not showRLines
    if showRLines:
        showLinesButton.place_forget()
        hideLinesButton.place(relwidth= .5, relheight=.34, rely=.66)
    else:
        hideLinesButton.place_forget()
        showLinesButton.place(relwidth= .5, relheight=.34, rely=.66)
    
    loadImage(resultsCanvas, imageNo, rlabel, showRLines)
    return

def submitSegment():
    global imagesCompleted, labelsCorrect

    if imagesCompleted < int(len(images)/2) and checkboxVar.get() == 1:
        labelsCorrect += 1
    segmentations.append((currentImage[0],currentImage[1],pointList.copy()))
    predLabel.set(' ')
    clearLines()
    imagesCompleted += 1

    if imagesCompleted >= int(len(images)/2):
        labelCheckbox.place_forget()

    if imagesCompleted < len(images):
        mCanvas = None
        if imagesCompleted < int(len(images)/2):
            mCanvas = modelImageCanvas
            loadImage(imgCanvas, imagesCompleted, mlabel, modelCanvas=mCanvas)
        else:
            loadImage(imgCanvas, imagesCompleted, mlabel, modelCanvas=mCanvas)
        mlabel.set("Your Label: {}".format(mlabel.get()))
    else:
        unloadMainButtons()
        resultsButton.place(relwidth=1, relheight=.66, rely=.34)
    return


#Frame Initialization
mainFrame = tk.Frame(root, bg="#244366")
imgCanvas = tk.Canvas(mainFrame, bg='#244366', highlightthickness=0)
imgCanvas.bind("<Button-1>", mouseClick)
mainButtonFrame = tk.Frame(mainFrame, bg='#244366')
modelImageCanvas=tk.Canvas(mainFrame, bg='#244366', highlightthickness=0)

inputFrame = tk.Frame(root, bg="#244366")
inputButtonFrame = tk.Frame(inputFrame)
previewCanvas = tk.Canvas(inputFrame, bg='#244366', highlightthickness=0)

resultsFrame = tk.Frame(root, bg="#244366")
resultsButtonFrame = tk.Frame(resultsFrame)
resultsCanvas = tk.Canvas(resultsFrame, bg='#244366', highlightthickness=0)

#Button/Label Declaration

inputImageButton=tk.Button(inputButtonFrame, text="Open Image", fg="white", bg="#244366", command=inputImage)
addImageButton=tk.Button(inputButtonFrame, text="Save Image", fg="white", bg="#244366", command=addImage)
finishInputButton=tk.Button(inputButtonFrame, text="Finish input", fg="white", bg="#244366", command=finishInput)
previewLabel = tk.Label(inputButtonFrame, textvariable=plabel, fg="white", bg='#244366', font=('Arial 20'))

drawSeg=tk.Button(mainButtonFrame, text="Draw segments", fg="white", bg="#244366", command=drawSegment)
endSeg=tk.Button(mainButtonFrame, text="End segmentation", fg="white", bg="#112030", command=endSegment)
clear=tk.Button(mainButtonFrame, text="Clear lines", fg="white", bg="#244366", command=clearLines)
submit=tk.Button(mainButtonFrame, text="Submit", fg="white", bg="#244366", command=submitSegment)
resultsButton=tk.Button(mainButtonFrame, text="Results", fg="white", bg="#112030", command=resultsTransition)
labelCheckbox = tk.Checkbutton(mainButtonFrame, text='Model label Correct?',variable=checkboxVar, onvalue=1, offvalue=0, bg="#244366", fg="white", font=('Arial 20'))
mainLabel = tk.Label(mainButtonFrame, textvariable=mlabel, fg="white", bg='#244366', font=('Arial 20'))
modelLabel = tk.Label(mainButtonFrame, textvariable=predLabel, fg="white", bg='#244366', font=('Arial 20'))


nextImageButton=tk.Button(resultsButtonFrame, text=">>", fg="white", bg="#244366", command=nextImage)
previousImageButton=tk.Button(resultsButtonFrame, text="<<", fg="white", bg="#244366", command=prevImage)
beginningButton=tk.Button(resultsButtonFrame, text="Back to beginning", fg="white", bg="#244366", command=restartProgram)
hideLinesButton=tk.Button(resultsButtonFrame, text="Hide lines", fg="white", bg="#244366", command=hideLines)
showLinesButton=tk.Button(resultsButtonFrame, text="Show lines", fg="white", bg="#112030", command=hideLines)
resultsLabel = tk.Label(resultsButtonFrame, textvariable=rlabel, fg="white", bg='#244366', font=('Arial 20'))
fractionCorrectLabel = tk.Label(resultsButtonFrame, textvariable=fractionLabel, fg="white", bg='#244366', font=('Arial 20'))

loadInputFrame()

root.resizable(False,False)
root.mainloop()