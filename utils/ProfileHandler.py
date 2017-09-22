from Tkinter import *
import csv
import tkMessageBox
import numpy as np
class Profile:

    name=""
    age=1
    nationality=""
    id=0
    photo=""
    legal=""

    def __init__(self,name, age, nationality,id,photo, legal="yes"):
        self.name=name
        self.age=age
        self.nationality=nationality
        self.id=id
        self.legal=legal
        self.photo=photo

    def display_portrait(self):
        color="red"
        if(self.legal.upper()=="LEGAL"):
            color="green"
        root = Toplevel()
        root.geometry("450x350")
        #dire = "../att_faces/s1/1.pgm"
        dire = self.photo
        photo = PhotoImage(file=dire)
        photo2 = photo.zoom(3, 3)
        pic = Label(root, image=photo2)
        pic.pack()
        pic.place(x=6, y=6)
        labels = list()
        labels.append(Label(root, text="NAME", font=("Helvetica", 10)))
        labels.append(Label(root, text="{}".format(self.name ), font=("Helvetica", 11)))
        labels.append(Label(root, text="AGE", font=("Helvetica", 10)))
        labels.append(Label(root, text="{}".format(self.age), font=("Helvetica", 12)))
        labels.append(Label(root, text="ID", font=("Helvetica", 10)))
        labels.append(Label(root, text="{}".format(self.id), font=("Helvetica", 12)))
        labels.append(Label(root, text="NATIONALITY", font=("Helvetica", 10)))
        labels.append(Label(root, text="{}".format(self.nationality), font=("Helvetica", 12)))
        labels.append(Label(root, text="STATUS", font=("Helvetica", 10)))
        labels.append(Label(root, text="{}".format(self.legal), fg=color, font=("Helvetica bold", 17)))
        dy = 8
        for x in labels:
            x.pack()
            x.place(x=290, y=dy)
            dy += 30
        root.mainloop()
        root.wait_window(root)

    def toString(self):
        return "name = {} age = {} nat = {} id = {} legal = {}".format(self.name,self.age,self.nationality,self.id
                                                                       ,self.legal)


