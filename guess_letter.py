# Draw a letter on tkinter canvas and press space for network to guess letter. Output will be on the command line window
# Requires ghostscript
# Requires tensorflow 1.5, but might work on newer versions without too much change in code
from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageEnhance
from PIL import Image
from PIL.ImageOps import invert
from tensorflow import keras
import numpy as np
import random
import pickle
import os
import sys

class Guess(object):

    DEFAULT_PEN_SIZE = 1.0
    DEFAULT_COLOR = 'black'
    
    def __init__(self):

        self.class_names = 'abcdefghijklmnopqrstuvwxyz'
        self.models = []
        print('\nLoading models...')
        self.path = 'trained_models/'
        self.models = []
        self.n = 1
        # Tries to load all models with the name "cnn(N).h5" where N is the model number
        # If you have one model in the path, it's name must be "cnn1.h5", and if you have a second model, it's name must be "cnn2.h5" and so on
        while True:
            try:
                self.models.append(keras.models.load_model('{}cnn{}.h5'.format(self.path,self.n)))
                self.n += 1
            except:
                break
        if len(self.models) == 0:
            print('\nNo models found in \'{}\''.format(self.path))
            input('Press anything to exit.')
            sys.exit()
        
        self.root = Tk()
        
        self.root.bind('<space>',self.space_pressed)
            
        os.system('cls')
        
        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)
        
        self.clear_button = Button(self.root, text='clear', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=4)

        self.choose_size_button = Scale(self.root, from_=60, to=100, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=5)

        self.c = Canvas(self.root, bg='white', width=700, height=600)
        self.c.grid(row=1, columnspan=5)
        
        self.setup()
        self.root.mainloop()
    
    def format(self):
        self.path = 'tmp/image.ps'
        self.c.postscript(file=self.path, colormode='color')
        self.im = Image.open(self.path).resize((28,28)).convert('L')
        return invert(self.im)
    
    def guess(self):
        self.image = np.asarray(self.format()).reshape(1,28,28,1) / 255.0 
        self.prediction = 0
        # Adds all models predictions
        for model in self.models:
            self.prediction += model.predict(self.image)
        print('\nThat looks like the letter {}!'.format(self.class_names[np.argmax(self.prediction[0])]))

    def space_pressed(self,key):
        self.guess()
    
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
    
    def clear_canvas(self):
        self.c.delete('all')
        
    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line((self.old_x, self.old_y, event.x, event.y),
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            
            
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Guess()
