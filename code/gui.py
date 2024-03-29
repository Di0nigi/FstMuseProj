import tkinter as tk 
from tkinter import filedialog
import dataPipeline as dP
import toyModel as tM
import toyGen as tG
import FstMuse as fM
from tkinter import ttk

class App:
    def __init__(self,r):
        self.mode="toy"
        self.currentText=""
        self.height=220
        self.width=600
        self.root=r
        self.panels=[]
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.title("FstMuse")
        self.gridLayer()
        self.panels[1].config(bg="gray", font=("Courier New", 15),anchor="nw",justify="left")
        #self.panels[3].config(bg="gray", font=("Courier New", 15),anchor="nw",justify="left")

        selectButton = tk.Button(self.panels[0], text="Select File", command=self.selectFile)
        selectButton.place(x=5, y=5, width=150, height=100)
        startButton=tk.Button(self.panels[0],text= "Start Model", comma=self.runModel)
        startButton.place(x=5, y=110, width=150, height=100)

        selectToy= tk.Button(self.panels[0], text="ToyModel", command=self.selectToy)
        selectToy.place(x=160, y=5, width=80, height=60)
        selectFst= tk.Button(self.panels[0], text="FstMuse", command=self.selectFst)
        selectFst.place(x=160, y=71, width=80, height=60)
        selectGan= tk.Button(self.panels[0], text="ToyGan", command=self.selectGan)
        selectGan.place(x=160, y=137, width=80, height=60)
        return

    def gridLayer(self):
        # Create widgets for the grid layer
        label = tk.Label(self.root,bg="gray", fg="black")
        label.place(x=0, y=0, width=(self.width//2), height=self.height)
        self.panels.append(label)
        
        label2 = tk.Label(self.root,bg="gray", fg="black")
        label2.place(x=(self.width//2), y=0, width=(self.width//2), height=self.height)
        self.panels.append(label2)
                
            
    def selectFile(self):
        # Open a file dialog for file selection
        self.filePath = filedialog.askopenfilename(title="Select a File", filetypes=[("Mp3 files", "*.mp3"), ("All files", "*.*")])

        # Display the selected file path (you can customize this part)
        if self.filePath:
            print(f"Selected File: {self.filePath}")
            self.openSong()


    def selectToy(self):
        self.mode="toy"
        return
    def selectFst(self):
        self.mode="fst"
        return
    def selectGan (self):
        self.mode="gan"
        return
            

    def runModel(self):
        if self.mode == "toy":
            inShape=(1291, 128)
            m=tM.toyClassifier(inShape,3)
            m.load("models\Toymodel85")
            pred=m.predict([self.track])[0]
            self.panels[1].config(text=f"{self.track.title} is {pred}")
            print("run")
        elif self.mode == "fst":
            classes=7
            dims=(1291, 128)
            genres=["pop","jazz","rock","folk","hiphop","punk","electronic"]
            m=fM.fstMuse(dim=dims,classes=7)
            m.load("models\\fstMuse35")
            pred=m.predict([self.track])[0]
            self.panels[1].config(text=f"{self.currentText}\n \n \nprediction: \n{self.track.title} is {pred}")

        elif self.mode=="gan":
            inShape=(1,1291, 128,1)
            m=tG.toyGan(dimensions=inShape)
            m.loadModel("models\\toyGan")
            gen=m.generate(filename="newSample.wav")
            self.audio_file = gen[1]
            self.panels[1].config(text=f"done generating file: \n{gen[1]}")

        return
    def openSong(self):
        self.track=dP.loadAndParse([self.filePath])[0]
        disp=f"title: {self.track.title}\ngenre: {self.track.label}"
        self.currentText=disp
        self.panels[1].config(text=disp)

        return
   

            






def main():
    root=tk.Tk()
    mainApp=App(root)
    root.mainloop()
    return

main()