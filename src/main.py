import os
import sys
import tkinter as tk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from detectors.latex_hole import LatexHoleDetector
from detectors.latex_stain import LatexStainDetector
from detectors.latex_tear import LatexTearDetector
from detectors.oven_burn import OvenBurnDetector
from detectors.oven_flour import OvenFlourDetector
from detectors.oven_frosting import OvenFrostingDetector


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.maxsize(1138, 535)
        self.resizable(False, False)
        self.geometry("1138x535")

        self.image_list = []
        for file in os.listdir('img'):
            self.image_list.append(file)

        self.title('Glove Defect Detection System')

        self.setup_widgets()

    def setup_widgets(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.listbox_frame = tk.Frame(self)
        self.listbox_frame.grid(
            row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)

        self.image_list_var = tk.Variable(value=self.image_list)
        self.images_listbox = tk.Listbox(
            self.listbox_frame, width=30, selectmode=tk.SINGLE, listvariable=self.image_list_var)
        self.images_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        self.images_listbox.pack(expand=True, fill=tk.BOTH)

        self.ori_image_frame = tk.Frame(self)
        self.ori_image_frame.grid(
            row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        self.prc_image_frame = tk.Frame(self)
        self.prc_image_frame.grid(
            row=0, column=2, sticky=tk.NSEW, padx=5, pady=5)

        self.ori_label = tk.Label(self.ori_image_frame, text='Original Image')
        self.ori_label.pack()
        self.ori_image_label = tk.Label(
            self.ori_image_frame, image=None)
        self.ori_image_label.pack()

        self.prc_label = tk.Label(self.prc_image_frame, text='Processed Image')
        self.prc_label.pack()
        self.prc_image_label = tk.Label(
            self.prc_image_frame, image=None)
        self.prc_image_label.pack()

    def on_image_select(self, event):
        img_index = self.images_listbox.curselection()[0]
        # Increased res due to detection issues w/ oven mitts flour
        pil_img = Image.open('img/' + self.image_list[img_index]).resize((500, 500))
        self.ori_image = ImageTk.PhotoImage(pil_img)
        self.ori_image_label.configure(image=self.ori_image)

        np_img = np.array(pil_img)
        np_img = cv.cvtColor(np_img, cv.COLOR_BGR2RGB)

        # add your detection code here
        # hole_result = LatexHoleDetector(np_img).detect()
        # tear_result = LatexTearDetector(np_img).detect()
        # stain_result = LatexStainDetector(np_img).detect()

        # Oven Mitts Detectors
        frosting_result = OvenFrostingDetector(np_img).detect()
        burn_results = OvenBurnDetector(np_img).detect()
        flour_results = OvenFlourDetector(np_img).detect()

        combined_result = np.zeros(
            (np_img.shape[0], np_img.shape[1], 4), dtype='uint8')

        # then add the result into this array
        # for result in [hole_result, tear_result, stain_result]:
        for result in [frosting_result, burn_results, flour_results]:
            combined_result += result

        alpha_foreground = combined_result[:, :, 3] / 255.0
        for color in range(0, 3):
            np_img[:, :, color] = (1.0 - alpha_foreground) * np_img[:, :, color] + \
                alpha_foreground * combined_result[:, :, color]

        np_img = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
        pil_img = Image.fromarray(np.uint8(np_img))
        self.prc_image = ImageTk.PhotoImage(pil_img)
        self.prc_image_label.configure(image=self.prc_image)


if __name__ == '__main__':
    app = App()

    app.mainloop()
