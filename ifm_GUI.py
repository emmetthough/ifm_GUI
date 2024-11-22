# IFM GUI script
# Emmett Hough, September 2024

import sys
import os
import pickle, gzip
import traceback
import time
from time import sleep
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

import importlib
import ifm_fitting_tools
importlib.reload(ifm_fitting_tools)
from ifm_fitting_tools import initialize_bounds, atomPicture, load_image, Bounds, on_select, on_enter, gaussian

import threading
import glob
import traceback

from datetime import datetime


LARGE_FONT = ("Verdana 24 bold")
MEDIUM_FONT = ("Verdana 18")
ROI_size = 16

global path 
path = os.getcwd()

global imageDir

def get_pic_num(fname):
    return int(fname.split('/')[-1].split('.tif')[0].split('_')[-1])

class analysis_GUI(tk.Tk):

    def __init__(self):

        # Initialize
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, "IFM Image Streaming")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initialize different pages
        self.frames = {}
        for page in (StartPage, AcquisitionPage, AnalysisPage):

            frame = page(container, self)
            self.frames[page] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        button3 = tk.Button(self.frames[StartPage], text="Load Existing Dataset", command=self.load_dataset)
        button3.pack()

        # Show Start Page
        self.show_frame(StartPage)
            
        
        # self.show_frame(AcquisitionPage)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def show_frame(self, container):
        frame = self.frames[container]
        frame.tkraise()
    
    def on_close(self):
        if tk.messagebox.askokcancel("Quit", "Save Data?"):
            try:
                print('Closing and Saving!')
                self.frames[AcquisitionPage].save_dataset()
                self.destroy()
            except:
                print('Error Saving!')
                self.destroy()
        else:
            print('Closing without Saving')
            try:
                os.rmdir(self.frames[AcquisitionPage].fits_path)
                print(f"Directory '{self.frames[AcquisitionPage].fits_path}' has been deleted successfully.")
                os.rmdir(self.frames[AcquisitionPage].saved_data_path)
                print(f"Directory '{self.frames[AcquisitionPage].saved_data_path}' has been deleted successfully.")
            except OSError as e:
                print(f"Error: {e.strerror}. Directory '{self.frames[AcquisitionPage].fits_path}' could not be deleted.")
                print(f"Error: {e.strerror}. Directory '{self.frames[AcquisitionPage].saved_data_path}' could not be deleted.")
            self.destroy()

    def load_dataset(self):

        try:
            dir_to_load = tk.filedialog.askdirectory(initialdir='saved_data')

            dset_type = glob.glob(os.path.join(dir_to_load, 'dataset*'))[0].split('.')[-1]
            if dset_type == 'npy':
                dset_to_load = os.path.join(dir_to_load, 'dataset.npy')
                dset = np.load(dset_to_load, allow_pickle=True)
            elif dset_type == 'p':
                try:
                    dset_to_load = os.path.join(dir_to_load, 'dataset.p')
                    with gzip.open(dset_to_load, 'rb') as f:
                        dset = pickle.load(f)
                except Exception:
                    dset_to_load = os.path.join(dir_to_load, 'dataset.gz.p')
                    with gzip.open(dset_to_load, 'rb') as f:
                        dset = pickle.load(f)

            bnds_to_load = os.path.join(dir_to_load, 'metadata.npy')

            self.frames[AcquisitionPage].fname0 = dset_to_load
            self.frames[AcquisitionPage].dataset = [pic for pic in dset]
            self.frames[AcquisitionPage].all_bounds = np.load(bnds_to_load, allow_pickle=True)
            self.frames[AcquisitionPage].options = [get_pic_num(pic.image_path) for pic in self.frames[AcquisitionPage].dataset]

            self.frames[AcquisitionPage].initialized = True
            self.frames[AcquisitionPage].update_options()
            self.frames[AcquisitionPage].start.config(fg='green')

            self.show_frame(AcquisitionPage)

            self.frames[AcquisitionPage].listbox.insert(tk.END, "Loaded from "+dir_to_load)
            fig,ax = self.frames[AcquisitionPage].dataset[-1].show_full_image(show=False)
            self.frames[AcquisitionPage].update_graph(fig,ax)

        except Exception as e:
            print('Error loading dataset! ', e)
            traceback.print_exc()
    
class StartPage(tk.Frame):

    def __init__(self, parent, controller):

        # Initialize with navigation buttons
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Welcome to IFM Fitting", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="Acquisition", command=lambda: controller.show_frame(AcquisitionPage))
        button.pack()

        button2 = tk.Button(self, text="Analysis", command=lambda: controller.show_frame(AnalysisPage))
        button2.pack()



class AcquisitionPage(tk.Frame):

    def __init__(self, parent, controller):

        # Initialize frames for GUI elements
        tk.Frame.__init__(self, parent)
        self.mframe = tk.Frame(self)
        self.mframe.pack()

        # title frame
        frame0 = tk.Frame(self.mframe)
        # buttons
        buttons_frame = tk.Frame(self.mframe)
        # controls frame
        self.controls_frame = tk.Frame(self.mframe)
        # graphs frame
        self.graph_frame = tk.Frame(self.mframe)
        graph_title = tk.Label(self.graph_frame, text="Image", font=MEDIUM_FONT)
        graph_title.grid(row=0, column=0, sticky='nsew')
        self.init_image_frame()
        # phase and flag frame
        self.phase_flag_frame = tk.Frame(self.mframe)
        # console output
        console_frame = tk.Frame(self.mframe)

        # geometry manager for all frames
        frame0.grid(row=0, column=0, columnspan=2, sticky='nsew')
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.controls_frame.grid(row=2, column=0, columnspan=1, sticky='new')
        self.graph_frame.grid(row=2, column=1, columnspan=1, sticky='nsew')
        self.phase_flag_frame.grid(row=3, column=1, columnspan=1, sticky='nsw')
        console_frame.grid(row=4, column=0, columnspan=2, sticky='nsew')

        self.mframe.grid_columnconfigure(0, weight=1)
        self.mframe.grid_columnconfigure(1, weight=1)
        self.mframe.grid_rowconfigure(0, weight=1)
        self.mframe.grid_rowconfigure(1, weight=1)
        self.mframe.grid_rowconfigure(2, weight=1)
        self.mframe.grid_rowconfigure(3, weight=1)
        self.mframe.grid_rowconfigure(4, weight=1)

        # phase and flag widgets
        self.init_phase_flag_frame()


        # Buttons and title widgets
        label = tk.Label(frame0, text="Acquisition", font=LARGE_FONT)
        label.pack(pady=5,padx=10)

        home = tk.Button(buttons_frame, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        change = tk.Button(buttons_frame, text="Change working directory", command=self.change_path)
        sel_first_im = tk.Button(buttons_frame, text="Select first image", command=self.select_first_image)
        self.start = tk.Button(buttons_frame, text="Start!", command=self.start_waiting, fg='red')

        home.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        change.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        sel_first_im.grid(row=0, column=2, sticky='ew', padx=5, pady=5)
        self.start.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        buttons_frame.rowconfigure(0, weight=1)
        buttons_frame.rowconfigure(1, weight=1)


        # Controls frame
        self.init_controls_frame()

        # Console frame
        scrollbar = tk.Scrollbar(console_frame) 
        self.listbox = tk.Listbox(console_frame, width=80)
        scrollbar.pack(side=tk.RIGHT, pady=5)
        self.listbox.pack(fill=tk.X, expand=True, padx=5, pady=5)

        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.insert(tk.END, path)

        # initialize other things

        self.initialized = False
        self.dataset = []
        self.kwargs = {'angle': 7, 'blur': 3} # add buttons for this!
        self.init_dirs()

    def init_image_frame(self):

        self.fig,self.ax = plt.subplots(2,2, figsize=(3,3))

        for a in self.ax:
            for sub_a in a:
                sub_a.axis('off')

        # self.ax1.set_ymargin(m=0.5)
        # self.fig.subplotpars.top = 5
        # self.fig.subplotpars.bottom = 0.1
        # self.fig.subplotpars.hspace = 0.5

        # self.vbar = tk.Scrollbar(self.frame4, orient=tk.VERTICAL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        # self.canvas.get_tk_widget().config(width=800, height=1000, scrollregion=(0,0,800,800))
        # self.canvas.get_tk_widget().config(yscrollcommand=self.vbar.set)

        # self.vbar.pack(side=tk.RIGHT, fill=tk.Y, expand=1)
        # self.vbar.config(command=self.canvas.get_tk_widget().yview)

        # self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        # self.toolbar.update()

        self.canvas.get_tk_widget().grid(row=1, column=0)
        # self.toolbar.pack()
        # self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def init_controls_frame(self):

        controls_title = tk.Label(self.controls_frame, text="Controls", font=MEDIUM_FONT)

        img_params_lbl = tk.Label(self.controls_frame, text='Image Params:')

        rotate_lbl = tk.Label(self.controls_frame, text='Rotate:')
        self.rotate_entry = tk.Entry(self.controls_frame)
        self.rotate_entry.insert(0, '7')

        blur_lbl = tk.Label(self.controls_frame, text='Blur:')
        self.blur_entry = tk.Entry(self.controls_frame)
        self.blur_entry.insert(0, '3')

        mag_lbl = tk.Label(self.controls_frame, text='Mag:')
        self.mag_entry = tk.Entry(self.controls_frame)
        self.mag_entry.insert(0, '1.8')

        px_size_lbl = tk.Label(self.controls_frame, text='Px Size:')
        self.px_size_entry = tk.Entry(self.controls_frame)
        self.px_size_entry.insert(0, '7.4')

        fits_lbl = tk.Label(self.controls_frame, text="Fits:")

        self.x_bool = tk.BooleanVar(value=True)
        x_chkbtn = tk.Checkbutton(self.controls_frame, text="x", variable=self.x_bool, onvalue=True, offvalue=False)

        self.y_bool = tk.BooleanVar(value=False)
        y_chkbtn = tk.Checkbutton(self.controls_frame, text="y", variable=self.y_bool, onvalue=True, offvalue=False)

        self.full_bool = tk.BooleanVar(value=True)
        full_chkbtn = tk.Checkbutton(self.controls_frame, text="full", variable=self.full_bool, onvalue=True, offvalue=False)

        self.ports_bool = tk.BooleanVar(value=True)
        ports_chkbtn = tk.Checkbutton(self.controls_frame, text="ports", variable=self.ports_bool, onvalue=True, offvalue=False)

        save_fits_lbl = tk.Label(self.controls_frame, text="Save fits:")
        self.save_fits_bool = tk.BooleanVar(value=False)
        save_fits_chkbtn = tk.Checkbutton(self.controls_frame, variable=self.save_fits_bool, onvalue=True, offvalue=False)

        selected_img_lbl = tk.Label(self.controls_frame, text="Image:")
        self.selected_img = tk.StringVar(self.controls_frame)
        self.options = ['None Available!']
        self.selected_dropdown = tk.OptionMenu(self.controls_frame, self.selected_img, *self.options)
        self.show_fits_btn = tk.Button(self.controls_frame, text='Show Fits', command=self.show_fits)
        self.change_bounds_btn = tk.Button(self.controls_frame, text='Change bounds', command=self.change_bounds)

        self.selected_img.trace_add("write", self.update_graph_from_options)

        self.show_fringe_btn = tk.Button(self.controls_frame, text='Show Fringe', command=self.show_fringe_window)

        # rows = 6
        columns = 5

        # geometry manager for controls frame

        row_i = 0
        controls_title.grid(row=row_i, column=0, columnspan=5, sticky='nsew')

        row_i += 1
        img_params_lbl.grid(row=row_i, column=0, sticky='ne')
        mag_lbl.grid(row=row_i, column=1, stick='nw')
        self.mag_entry.grid(row=row_i, column=2, sticky='nw')
        px_size_lbl.grid(row=row_i, column=3, sticky='nw')
        self.px_size_entry.grid(row=row_i, column=4, sticky='nw')

        row_i += 1
        rotate_lbl.grid(row=row_i, column=1, sticky='nw')
        self.rotate_entry.grid(row=row_i, column=2, sticky='nw')
        blur_lbl.grid(row=row_i, column=3, sticky='nw')
        self.blur_entry.grid(row=row_i, column=4, sticky='nw')

        row_i += 1
        fits_lbl.grid(row=row_i, column=0, sticky='ne')
        x_chkbtn.grid(row=row_i, column=1, sticky='nw')
        y_chkbtn.grid(row=row_i, column=2, sticky='nw')

        row_i += 1
        full_chkbtn.grid(row=row_i, column=1, sticky='nw')
        ports_chkbtn.grid(row=row_i, column=2, sticky='nw')

        row_i += 1
        save_fits_lbl.grid(row=row_i, column=0, sticky='ne')
        save_fits_chkbtn.grid(row=row_i, column=1, sticky='nw')

        row_i += 1
        selected_img_lbl.grid(row=row_i, column=0, sticky='ne')
        self.dropdown_row = row_i
        self.dropdown_col = 1
        self.selected_dropdown.grid(row=self.dropdown_row, column=self.dropdown_col, sticky='nw')
        self.show_fits_btn.grid(row=row_i, column=2, sticky='nw')
        self.change_bounds_btn.grid(row=row_i, column=3, stick='nw')

        row_i += 2
        self.show_fringe_btn.grid(row=row_i, column=0, columnspan=columns, sticky='nw')

        for col in range(columns):
            self.controls_frame.columnconfigure(col, weight=1)
        for row in range(row_i):
            self.controls_frame.rowconfigure(row, weight=1)

    def init_phase_flag_frame(self):

        self.phase_lbl = tk.Label(self.phase_flag_frame, text='Phase:')
        self.phase_entry = tk.Entry(self.phase_flag_frame, width=10)
        self.flag_lbl = tk.Label(self.phase_flag_frame, text='Flag')
        self.flag_bool = tk.BooleanVar(value=False)
        self.flag_chkbtn = tk.Checkbutton(self.phase_flag_frame, variable=self.flag_bool)
        self.phase_flag_btn = tk.Button(self.phase_flag_frame, text='Update', command=self.update_phase_flag)
        

        self.phase_lbl.grid(row=0, column=0, sticky='ne')
        self.phase_entry.grid(row=0, column=1, sticky='nw')
        self.flag_lbl.grid(row=0, column=2, sticky='ne')
        self.flag_chkbtn.grid(row=0, column=3, sticky='nw')
        self.phase_flag_btn.grid(row=0, column=4, sticky='nsew')

        for row in range(5):
            self.phase_flag_frame.grid_rowconfigure(row, weight=1)
        for col in range(1):
            self.phase_flag_frame.grid_columnconfigure(col, weight=1)

    def select_first_image(self):

        if self.initialized:
            self.listbox.insert(tk.END, "Already Initialized! File: "+self.fname0)
            self.listbox.see(tk.END)
            return
        
        self.fname0 = tk.filedialog.askopenfilename(initialdir='.')

        if len(self.fname0)==0:
            self.listbox.insert(tk.END, "First image canceled, try again")
            self.listbox.see(tk.END)
            return

        try:
            self.listbox.insert(tk.END, "First image loading: "+self.fname0)
            self.listbox.see(tk.END)
            plt.close(self.fig)

            try:
                self.kwargs = {'angle': int(self.rotate_entry.get()), 'blur': int(self.blur_entry.get()), 
                               'px_size': float(self.px_size_entry.get()), 'mag': float(self.mag_entry.get())}
            except Exception as e:
                print('Error with setting kwargs!')
                print(e)
                self.listbox.insert(tk.END, "Error with setting kwargs!")
                self.listbox.insert(tk.END, e)
                self.listbox.see(tk.END)

            # self.all_bounds = initialize_bounds(self.fname0, **self.kwargs)
            self.all_bounds = self.initialize_bounds_tk(self.fname0, **self.kwargs)

            pic = atomPicture(self.fname0, self.all_bounds[0], self.all_bounds[1:], **self.kwargs)
            self.dataset.append(pic)

            save = self.save_fits_bool.get()

            if self.full_bool.get():
                    if self.x_bool.get():
                        pic.fit_gaus(type='full', axis='x', save=save, fits_path=self.fits_path)
                    if self.y_bool.get():
                        pic.fit_gaus(type='full', axis='y', save=save, fits_path=self.fits_path)

            if self.ports_bool.get():
                if self.x_bool.get():
                    pic.fit_gaus(type='ports', axis='x', save=save, fits_path=self.fits_path)
                if self.y_bool.get():
                    pic.fit_gaus(type='ports', axis='y', save=save, fits_path=self.fits_path)

            self.start.config(fg='green')
            self.initialized = True
            
            self.options = [get_pic_num(self.fname0)]
            self.update_options()

            fig0,ax0 = pic.show_full_image(show=False)
            self.update_graph(fig0,ax0)

        except Exception as e:
            print('Error loading image! Expection: ',e)
            traceback.print_exc()

    def rectange_selector_tk(self, fig, ax, bounds, popup, props=dict(edgecolor='black', alpha=0.3, fill=False)):
        
        enter_pressed = threading.Event()
        
        def on_select_tk(eclick, erelease, bounds=bounds):
            bounds.xmin = int(min(eclick.xdata, erelease.xdata))
            bounds.xmax = int(max(eclick.xdata, erelease.xdata))
            bounds.ymin = int(min(eclick.ydata, erelease.ydata))
            bounds.ymax = int(max(eclick.ydata, erelease.ydata))

        def on_enter_tk(event):
            enter_pressed.set()
            popup.destroy()
            plt.close(fig)

        rect_sel = RectangleSelector(
            ax, lambda eclick, erelease: on_select_tk(eclick, erelease, bounds=bounds), 
            drawtype='box',
            useblit=True,
            props=props,
            interactive=True)
        
        popup.bind('<Return>', on_enter_tk)
        popup.wait_window()

        enter_pressed.wait()
        rect_sel.set_active(False)

    def initialize_bounds_tk(self, fname, **kwargs):

        if 'angle' in kwargs.keys():
            angle = kwargs['angle']
        else:
            angle = 7

        if 'blur' in kwargs.keys():
            blur = kwargs['blur']
        else:
            blur = 3

        img_raw = load_image(fname, angle=angle, blur=blur) # load first image 

        full_bounds = Bounds()

        popup = tk.Toplevel(self.mframe)
        popup.title('Atom Region')

        fig,ax = plt.subplots(1, figsize=(8,6))
        ax.imshow(img_raw, cmap='Greys_r') #display
        fig.suptitle('Select Atom Region', fontsize=30)

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        self.rectange_selector_tk(fig, ax, full_bounds, popup, props=dict(edgecolor='black', alpha=0.3, fill=False))

        # define bounds for atom region
        xbnds0 = [round(full_bounds.xmin), round(full_bounds.xmax)]
        ybnds0 = [round(full_bounds.ymin), round(full_bounds.ymax)]

        # atom region
        img = load_image(fname, xbnd=xbnds0, ybnd=ybnds0, blur=blur, angle=angle, show=False)

        p1_bounds = Bounds()

        popup = tk.Toplevel(self.mframe)
        popup.title('Port 1')

        fig,ax = plt.subplots(1, figsize=(8,6))
        ax.imshow(img, cmap='Greys_r')
        fig.suptitle('Select Port 1', fontsize=30)

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        self.rectange_selector_tk(fig, ax, p1_bounds, popup, props=dict(edgecolor='blue', alpha=0.3, fill=False))

        # define bounds for port 1, in full image coordinates
        xbnds1 = [xbnds0[0]+round(b) for b in [p1_bounds.xmin, p1_bounds.xmax]]
        ybnds1 = [ybnds0[0]+round(b) for b in [p1_bounds.ymin, p1_bounds.ymax]]

        p2_bounds = Bounds()

        popup = tk.Toplevel(self.mframe)
        popup.title('Port 2')

        fig,ax = plt.subplots(1, figsize=(8,6))
        ax.imshow(img, cmap='Greys_r')
        fig.suptitle('Select Port 2', fontsize=30)
        port1 = Rectangle(
            (xbnds1[0]-xbnds0[0], ybnds1[0]-ybnds0[0]),                 # Bottom-left corner (x, y)
            xbnds1[1]-xbnds1[0],                       # Width of the rectangle
            ybnds1[1]-ybnds1[0],                      # Height of the rectangle
            facecolor='none',        # Transparent fill
            edgecolor='blue',        # Color of the border
            linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
            linewidth=2,             # Border width
            alpha=0.5,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
            label='Port 1'
            )     
        
        ax.add_patch(port1)

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        self.rectange_selector_tk(fig, ax, p2_bounds, popup, props=dict(edgecolor='green', alpha=0.3, fill=False))

        xbnds2 = [xbnds0[0]+round(b) for b in [p2_bounds.xmin, p2_bounds.xmax]]
        ybnds2 = [ybnds0[0]+round(b) for b in [p2_bounds.ymin, p2_bounds.ymax]]

        return [(xbnds0,ybnds0), (xbnds1,ybnds1), (xbnds2,ybnds2)]

    def init_dirs(self):
        self.dataset_name = datetime.now().strftime("%Y_%m_%d-%H_%M")
        if "fits" not in os.listdir():
            os.mkdir("fits")
        self.fits_path = os.path.join("fits", self.dataset_name)
        print("Fits path: ", self.fits_path)
        os.mkdir(self.fits_path)
        if "saved_data" not in os.listdir():
            os.mkdir("saved_data")
        self.saved_data_path = os.path.join("saved_data", self.dataset_name)
        os.mkdir(self.saved_data_path)

    def save_dataset(self):
        try:
            # np.save(os.path.join(self.saved_data_path, 'dataset'), self.dataset, allow_pickle=True)
            with gzip.open(os.path.join(self.saved_data_path, 'dataset.gz.p'), 'wb') as f:
                pickle.dump(self.dataset, f)
            np.save(os.path.join(self.saved_data_path, 'metadata'), self.all_bounds, allow_pickle=True)
            print('Successfully saved')
        except Exception as e:
            print('Error saving! Exception thrown: ', e)

    def update_options(self):
        self.selected_dropdown.destroy()
        self.selected_img.set(self.options[-1])
        self.selected_dropdown = tk.OptionMenu(self.controls_frame, self.selected_img, *self.options)
        self.selected_dropdown.grid(row=self.dropdown_row, column=self.dropdown_col, sticky='nw')

    def start_waiting(self):

        if not self.initialized:
            self.listbox.insert(tk.END, "Please initialize by selecting first image!")
            self.listbox.see(tk.END)
        else:
            self.listbox.insert(tk.END, "Starting acquisition!")
            self.listbox.see(tk.END)
            
            self.start.config(text='Stop', fg='orange', command=self.toggle_start_stop)

            self.running = True

            thread = threading.Thread(target=self.check_new_files, daemon=True)
            thread.start()
    
    def toggle_start_stop(self):
        if self.running:
            self.running = False
            self.start.config(text='Start!', fg='green', command=self.start_waiting)
            self.listbox.insert(tk.END, 'Stopping acquisition')
            self.listbox.see(tk.END)

    def check_new_files(self):
        while self.running:
            self.after(5000, self.update)

    def change_bounds(self):

        if self.initialized:

            pic = self.get_pic_from_selected()

            xbnds0 = pic.atom_xbnds
            ybnds0 = pic.atom_ybnds

            # atom region
            img = load_image(pic.image_path, xbnd=xbnds0, ybnd=ybnds0, blur=self.kwargs['blur'], angle=self.kwargs['angle'], show=False)

            p1_bounds = Bounds()

            popup = tk.Toplevel(self.mframe)
            popup.title('Port 1')

            fig,ax = plt.subplots(1, figsize=(8,6))
            ax.imshow(img, cmap='Greys_r')
            fig.suptitle('Select Port 1', fontsize=30)

            oldport1 = Rectangle(
            (pic.p1_xbnds[0]-pic.atom_xbnds[0], pic.p1_ybnds[0]-pic.atom_ybnds[0]),                 # Bottom-left corner (x, y)
            pic.p1_xbnds[1]-pic.p1_xbnds[0],                       # Width of the rectangle
            pic.p1_ybnds[1]-pic.p1_ybnds[0],                      # Height of the rectangle
            facecolor='none',        # Transparent fill
            edgecolor='blue',        # Color of the border
            linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
            linewidth=1,             # Border width
            alpha=0.3,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
            label='Port 1'
            )     

            oldport2 = Rectangle(
            (pic.p2_xbnds[0]-pic.atom_xbnds[0], pic.p2_ybnds[0]-pic.atom_ybnds[0]),                 # Bottom-left corner (x, y)
            pic.p2_xbnds[1]-pic.p2_xbnds[0],                       # Width of the rectangle
            pic.p2_ybnds[1]-pic.p2_ybnds[0],                      # Height of the rectangle
            facecolor='none',        # Transparent fill
            edgecolor='green',        # Color of the border
            linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
            linewidth=1,             # Border width
            alpha=0.3,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
            label='Port 2'
            )      

            ax.add_patch(oldport1)
            ax.add_patch(oldport2)

            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()

            self.rectange_selector_tk(fig, ax, p1_bounds, popup, props=dict(edgecolor='blue', alpha=0.3, fill=False))

            # define bounds for port 1, in full image coordinates
            xbnds1 = [xbnds0[0]+round(b) for b in [p1_bounds.xmin, p1_bounds.xmax]]
            ybnds1 = [ybnds0[0]+round(b) for b in [p1_bounds.ymin, p1_bounds.ymax]]

            p2_bounds = Bounds()

            popup = tk.Toplevel(self.mframe)
            popup.title('Port 2')

            fig,ax = plt.subplots(1, figsize=(8,6))
            ax.imshow(img, cmap='Greys_r')
            fig.suptitle('Select Port 2', fontsize=30)

            oldport1 = Rectangle(
            (pic.p1_xbnds[0]-pic.atom_xbnds[0], pic.p1_ybnds[0]-pic.atom_ybnds[0]),                 # Bottom-left corner (x, y)
            pic.p1_xbnds[1]-pic.p1_xbnds[0],                       # Width of the rectangle
            pic.p1_ybnds[1]-pic.p1_ybnds[0],                      # Height of the rectangle
            facecolor='none',        # Transparent fill
            edgecolor='blue',        # Color of the border
            linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
            linewidth=1,             # Border width
            alpha=0.3,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
            label='Port 1'
            )     

            oldport2 = Rectangle(
            (pic.p2_xbnds[0]-pic.atom_xbnds[0], pic.p2_ybnds[0]-pic.atom_ybnds[0]),                 # Bottom-left corner (x, y)
            pic.p2_xbnds[1]-pic.p2_xbnds[0],                       # Width of the rectangle
            pic.p2_ybnds[1]-pic.p2_ybnds[0],                      # Height of the rectangle
            facecolor='none',        # Transparent fill
            edgecolor='green',        # Color of the border
            linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
            linewidth=1,             # Border width
            alpha=0.3,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
            label='Port 2'
            )      

            port1 = Rectangle(
                (xbnds1[0]-xbnds0[0], ybnds1[0]-ybnds0[0]),                 # Bottom-left corner (x, y)
                xbnds1[1]-xbnds1[0],                       # Width of the rectangle
                ybnds1[1]-ybnds1[0],                      # Height of the rectangle
                facecolor='none',        # Transparent fill
                edgecolor='blue',        # Color of the border
                linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
                linewidth=2,             # Border width
                alpha=0.5,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
                label='Port 1'
                )     
            
            ax.add_patch(port1)
            ax.add_patch(oldport1)
            ax.add_patch(oldport2)

            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()

            self.rectange_selector_tk(fig, ax, p2_bounds, popup, props=dict(edgecolor='green', alpha=0.3, fill=False))

            xbnds2 = [xbnds0[0]+round(b) for b in [p2_bounds.xmin, p2_bounds.xmax]]
            ybnds2 = [ybnds0[0]+round(b) for b in [p2_bounds.ymin, p2_bounds.ymax]]

            new_pic = atomPicture(pic.image_path, (pic.atom_xbnds, pic.atom_ybnds), [[xbnds1,ybnds1],[xbnds2, ybnds2]], **self.kwargs)

            try:
                save = self.save_fits_bool.get()
                if self.full_bool.get():
                    if self.x_bool.get():
                        pic.fit_gaus(type='full', axis='x', save=save, fits_path=self.fits_path)
                    if self.y_bool.get():
                        pic.fit_gaus(type='full', axis='y', save=save, fits_path=self.fits_path)

                if self.ports_bool.get():
                    if self.x_bool.get():
                        pic.fit_gaus(type='ports', axis='x', save=save, fits_path=self.fits_path)
                    if self.y_bool.get():
                        pic.fit_gaus(type='ports', axis='y', save=save, fits_path=self.fits_path)
                
            except Exception as e:
                self.listbox.insert(tk.END, f"Failed to fit pic {self.last_in}.")
                self.listbox.see(tk.END)
                print("Failed to fit pic {self.last_in}. FIX THIS HANDLING! AND FITTING IN GENERAL!")

            pic_num = int(self.selected_img.get())
            dataset_nums = np.array([get_pic_num(pic.image_path) for pic in self.dataset])
            self.dataset[np.where(dataset_nums==pic_num)[0][0]] = new_pic
            self.update_graph_from_options()

        else:
            self.listbox.insert(tk.END, "Not initialized!")
            self.listbox.see(tk.END)

    def update_phase_flag(self):

        if self.initialized:

            try:
                dset_nums = np.array([get_pic_num(pic.image_path) for pic in self.dataset])
                sel_num = int(self.selected_img.get())
                ind = np.where(dset_nums == sel_num)[0][0]

                phase = self.phase_entry.get()
                if len(phase) > 0:
                    self.dataset[ind].phase = float(self.phase_entry.get())
                self.dataset[ind].flag = self.flag_bool.get()

                self.listbox.insert(tk.END, f'Pic {sel_num}: Phase {self.dataset[ind].phase}, Flag {self.dataset[ind].flag}')
                self.listbox.see(tk.END)

            except Exception as e:
                self.listbox.insert(tk.END, f'Error saving phase and flag for pic {self.selected_img.get()}')
                self.listbox.insert(tk.END, e)
                self.listbox.see(tk.END)
        else:
            self.listbox.insert(tk.END, "Not initialized!")
            self.listbox.see(tk.END)

    def update(self):

        self.last_in = get_pic_num(self.dataset[-1].image_path)
        initial_pic_num = get_pic_num(self.dataset[0].image_path)

        all_fnames = sorted(glob.glob('*.tif'))
        to_add = [f for f in all_fnames if (get_pic_num(f) > initial_pic_num) and (get_pic_num(f) > self.last_in)]

        save = self.save_fits_bool.get()

        for new_fname in to_add:
            
            try:
                print('adding new pic: ', new_fname)
                self.listbox.insert(tk.END, 'Adding picture: '+new_fname)
                self.listbox.see(tk.END)

                self.update_phase_flag()

                pic = atomPicture(new_fname, self.all_bounds[0], self.all_bounds[1:], **self.kwargs)
                self.dataset.append(pic)

                self.last_in = get_pic_num(new_fname)

                try:
                
                    if self.full_bool.get():
                        if self.x_bool.get():
                            pic.fit_gaus(type='full', axis='x', save=save, fits_path=self.fits_path)
                        if self.y_bool.get():
                            pic.fit_gaus(type='full', axis='y', save=save, fits_path=self.fits_path)

                    if self.ports_bool.get():
                        if self.x_bool.get():
                            pic.fit_gaus(type='ports', axis='x', save=save, fits_path=self.fits_path)
                        if self.y_bool.get():
                            pic.fit_gaus(type='ports', axis='y', save=save, fits_path=self.fits_path)
                
                except Exception as e:
                    self.listbox.insert(tk.END, f"Failed to fit pic {self.last_in}.")
                    self.listbox.see(tk.END)
                    print("Failed to fit pic {self.last_in}. FIX THIS HANDLING! AND FITTING IN GENERAL!")

                self.options.append(get_pic_num(new_fname))
                self.update_options()

                if len(to_add) == 1:
                    fig,ax = pic.show_full_image(show=False)
                    self.update_graph(fig,ax)

            except Exception as e:
                print(f'Error with {new_fname} in update! Exception: {e}')

    def update_graph_from_options(self, *args):
        if self.initialized:
            try:
                if len(self.dataset) > 1:

                    pic_to_draw = self.get_pic_from_selected()
                    
                    fig,ax = pic_to_draw.show_full_image(show=False)
                    self.update_graph(fig,ax)
            except Exception as e:
                self.listbox.insert(tk.END, f"Error showing Pic {int(self.selected_img.get())}! Exception thrown:")
                self.listbox.insert(tk.END, e)
                self.listbox.see(tk.END)
        else:
            self.listbox.insert(tk.END, "Not initialized!")
            self.listbox.see(tk.END)

    def update_graph(self, fig, ax):

        plt.close(self.fig)

        fig.set_size_inches(3,3)

        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0)
        self.canvas.draw()

        pic = self.get_pic_from_selected()

        self.phase_entry.delete(0, tk.END)
        if pic.phase is not None:
            self.phase_entry.insert(0, pic.phase)
        self.flag_bool.set(pic.flag)

    def get_pic_from_selected(self):
        pic_num = int(self.selected_img.get())
        dataset_nums = np.array([get_pic_num(pic.image_path) for pic in self.dataset])
        return self.dataset[np.where(pic_num==dataset_nums)[0][0]]

    def show_fringe_window(self):
        self.fringe_window = tk.Toplevel(self)
        self.fringe_window.title("Streaming fringe")
        lbl = tk.Label(self.fringe_window, text='FRINGE')
        lbl.pack()
        close_btn = tk.Button(self.fringe_window, text='close', command=self.fringe_window.destroy)
        close_btn.pack()
        show_avg_btn = tk.Button(self.fringe_window, text='show avg', command=self.show_asym_avg)
        show_avg_btn.pack()

        fig,ax = self.plot_fringe()
        canvas = FigureCanvasTkAgg(fig, master=self.fringe_window)
        canvas.get_tk_widget().pack()
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.fringe_window)
        toolbar.update()
        toolbar.pack()

    def plot_fringe(self):
        self.fringe_fig,ax = plt.subplots(2, 1, figsize=(5,4))
        # good, flagged

        data = {}
        data['pic_nums'] = ([],[])
        objects = ['phase', 'full_Nx', 'full_Ny', 'p1_Nx', 'p2_Nx', 'p1_Ny', 'p2_Ny']
        for obj in objects:
            data[obj] = ([],[])


        for pic in self.dataset:

            if pic.phase is not None:

                i = 0
                if pic.flag:
                    i = 1

                data['phase'][i].append(pic.phase)
                data['pic_nums'][i].append(get_pic_num(pic.image_path))

                if self.full_bool.get():
                    if self.x_bool.get():
                        # data['full_Nx'][i].append(pic.full_Nx)
                        data['full_Nx'][i].append(pic.fits['full_Nx'])
                    if self.y_bool.get():
                        # data['full_Ny'][i].append(pic.full_Ny)
                        data['full_Ny'][i].append(pic.fits['full_Ny'])

                if self.ports_bool.get():
                    if self.x_bool.get():
                        # data['p1_Nx'][i].append(pic.p1_Nx)
                        data['p1_Nx'][i].append(pic.fits['p1_Nx'])
                        # data['p2_Nx'][i].append(pic.p2_Nx)
                        data['p2_Nx'][i].append(pic.fits['p2_Nx'])
                    if self.y_bool.get():
                        # data['p1_Ny'][i].append(pic.p1_Ny)
                        data['p1_Ny'][i].append(pic.fits['p1_Ny'])
                        # data['p2_Ny'][i].append(pic.p2_Ny)
                        data['p2_Ny'][i].append(pic.fits['p2_Ny'])

        markers = ['o', '+', 'x', '*', '<', '>']
        i = -1
        self.good_artists = []
        self.bad_artists = []

        for quantity in objects[1:]:
            i += 1
            good,bad = data[quantity]
            if len(good)>0:
                gartist = ax[0].scatter(data['phase'][0], good, c='g', marker=markers[i], label=quantity, picker=True, pickradius=10)
                self.good_artists.append(gartist)
            if len(bad)>0:
                bartist = ax[0].scatter(data['phase'][1], bad, c='r', marker=markers[i], picker=True, pickradius=10)
                self.bad_artists.append(bartist)
        ax[0].legend()

        try:
            asym_good_x = (np.array(data['p1_Nx'][0]) - np.array(data['p2_Nx'][0]))/(np.array(data['p1_Nx'][0]) + np.array(data['p2_Nx'][0]))
            asym_bad_x =  (np.array(data['p1_Nx'][1]) - np.array(data['p2_Nx'][1]))/(np.array(data['p1_Nx'][1]) + np.array(data['p2_Nx'][1]))

            asym_good_artist = ax[1].scatter(data['phase'][0], asym_good_x, c='g', picker=True, pickradius=10)
            asym_bad_artist = ax[1].scatter(data['phase'][1], asym_bad_x, c='r', picker=True, pickradius=10)

            self.good_artists.append(asym_good_artist)
            self.bad_artists.append(asym_bad_artist)
            self.good_asym_data = [data['phase'][0], asym_good_x]

            ax[1].set_ylabel('Asymmetry')
        except:
            print('Bad asym!')

        self.data_pic_nums = data['pic_nums']

        self.fringe_fig.canvas.mpl_connect('pick_event', self.on_pick)
        cb_registry0 = ax[0].callbacks
        cb_registry1 = ax[1].callbacks
        cid0y = cb_registry0.connect('ylim_changed', lambda x: self.reattach_picker())
        cid0x = cb_registry0.connect('xlim_changed', lambda x: self.reattach_picker())
        cid1y = cb_registry1.connect('ylim_changed', lambda x: self.reattach_picker())
        cid1x = cb_registry1.connect('xlim_changed', lambda x: self.reattach_picker())

        return self.fringe_fig,ax
    
    def show_asym_avg(self):

        avg_popup = tk.Toplevel(self)
        avg_popup.title("Averaged Fringe")
        # close_btn = tk.Button(self.fringe_window, text='close', command=avg_popup.destroy)
        # close_btn.pack()

        phases = []
        sorted_data = []

        for phase in self.good_asym_data[0]:
            if phase not in phases:
                phases.append(phase)

        phases = np.array(sorted(phases))
        data_avg = np.zeros(phases.size)
        data_err = np.zeros(phases.size)

        for i,phase in enumerate(phases):
            inds = np.where(self.good_asym_data[0]==phase)[0]
            data_pts = []
            for ind in inds:
                data_pts.append(self.good_asym_data[1][ind])
            data_avg[i] = np.mean(data_pts)
            data_err[i] = np.std(data_pts)/np.sqrt(len(data_pts))

        fig,ax = plt.subplots(1)
        ax.errorbar(phases, data_avg, yerr=data_err, fmt='o', ecolor='k', elinewidth=1.5, capsize=4, capthick=1.5)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Asymmetry')


        canvas = FigureCanvasTkAgg(fig, master=avg_popup)
        canvas.get_tk_widget().pack()
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, avg_popup)
        toolbar.update()
        toolbar.pack()

    def show_fits(self):

        pic = self.get_pic_from_selected()

        popup = tk.Toplevel(self.mframe)
        popup.title('Fits '+pic.image_name)
        
        if self.full_bool.get():
            if self.x_bool.get() and self.y_bool.get():
                fig,ax = plt.subplots(2,1)
                ax[0].plot(np.arange(0,pic.fullxint.size), pic.fullxint, 'o', label='data')
                ax[0].plot(np.linspace(0,pic.fullxint.size, 100), gaussian(np.linspace(0,pic.fullxint.size, 100), *pic.fits['full_x_params'][0]), label='fit')
                ax[1].plot(np.arange(0,pic.fullyint.size), pic.fullyint, 'o', label='data')
                ax[1].plot(np.linspace(0,pic.fullyint.size, 100), gaussian(np.linspace(0,pic.fullyint.size, 100), *pic.fits['full_y_params'][0]), label='fit')
                ax[0].legend()
                ax[1].legend()
                ax.set_title('Full ints '+pic.image_name)

            elif self.x_bool.get():
                fig,ax = plt.subplots(1)
                ax.plot(np.arange(0,pic.fullxint.size), pic.fullxint, 'o', label='data')
                ax.plot(np.linspace(0,pic.fullxint.size, 100), gaussian(np.linspace(0,pic.fullxint.size, 100), *pic.fits['full_x_params'][0]), label='fit')
                ax.legend()
                ax.set_title('Full x int '+pic.image_name)

            elif self.y_bool.get():
                fig,ax = plt.subplots(1)
                ax.plot(np.arange(0,pic.fullyint.size), pic.fullyint, 'o', label='data')
                ax.plot(np.linspace(0,pic.fullyint.size, 100), gaussian(np.linspace(0,pic.fullyint.size, 100), *pic.fits['full_y_params'][0]), label='fit')
                ax.legend()
                ax.set_title('Full y int '+pic.image_name)

            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0)
            popup.columnconfigure(0, weight=1)

        if self.ports_bool.get():
            if self.x_bool.get() and self.y_bool.get():
                fig2,ax2 = plt.subplots(2,2)
                ax2[0,0].plot(np.arange(0,pic.p1xint.size), pic.p1xint, 'o', label='data')
                ax2[0,0].plot(np.linspace(0,pic.p1xint.size, 100), gaussian(np.linspace(0,pic.p1xint.size, 100), *pic.fits['p1_x_params'][0]), label='fit')
                ax2[0,1].plot(np.arange(0,pic.p1yint.size), pic.p1yint, 'o', label='data')
                ax2[0,1].plot(np.linspace(0,pic.p1yint.size, 100), gaussian(np.linspace(0,pic.p1yint.size, 100), *pic.fits['p1_y_params'][0]), label='fit')
                ax2[0,0].legend()

                ax2[1,0].plot(np.arange(0,pic.p2xint.size), pic.p2xint, 'o', label='data')
                ax2[1,0].plot(np.linspace(0,pic.p2xint.size, 100), gaussian(np.linspace(0,pic.p2xint.size, 100), *pic.fits['p2_x_params'][0]), label='fit')
                ax2[1,1].plot(np.arange(0,pic.p2yint.size), pic.p2yint, 'o', label='data')
                ax2[1,1].plot(np.linspace(0,pic.p2yint.size, 100), gaussian(np.linspace(0,pic.p2yint.size, 100), *pic.fits['p2_y_params'][0]), label='fit')

                fig2.suptitle('Ports ints '+pic.image_name)
                ax2[0,0].set_title('p1 x int')
                ax2[0,1].set_title('p1 y int')
                ax2[1,0].set_title('p2 x int')
                ax2[1,1].set_title('p2 y int')

            elif self.x_bool.get():
                fig2,ax2 = plt.subplots(2,1)
                ax2[0].plot(np.arange(0,pic.p1xint.size), pic.p1xint, 'o', label='data')
                ax2[0].plot(np.linspace(0,pic.p1xint.size, 100), gaussian(np.linspace(0,pic.p1xint.size, 100), *pic.fits['p1_x_params'][0]), label='fit')
                ax2[0].legend()
                ax2[1].plot(np.arange(0,pic.p2xint.size), pic.p2xint, 'o', label='data')
                ax2[1].plot(np.linspace(0,pic.p2xint.size, 100), gaussian(np.linspace(0,pic.p2xint.size, 100), *pic.fits['p2_x_params'][0]), label='fit')
                fig2.suptitle('x ints '+pic.image_name)
                ax2[0].set_title('port 1')
                ax2[1].set_title('port 2')

            elif self.y_bool.get():
                fig2,ax2 = plt.subplots(2,1)
                ax2[0].plot(np.arange(0,pic.p1yint.size), pic.p1yint, 'o', label='data')
                ax2[0].plot(np.linspace(0,pic.p1yint.size, 100), gaussian(np.linspace(0,pic.p1yint.size, 100), *pic.fits['p1_y_params'][0]), label='fit')
                ax2[0].legend()
                ax2[1].plot(np.arange(0,pic.p2yint.size), pic.p2yint, 'o', label='data')
                ax2[1].plot(np.linspace(0,pic.p2yint.size, 100), gaussian(np.linspace(0,pic.p2yint.size, 100), *pic.fits['p2_y_params'][0]), label='fit')
                fig2.suptitle('y ints '+pic.image_name)
                ax2[0].set_title('port 1')
                ax2[1].set_title('port 2')

            canvas2 = FigureCanvasTkAgg(fig2, master=popup)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.grid(row=0, column=1)
            popup.columnconfigure(1, weight=1)

        popup.rowconfigure(0, weight=1)

    def on_pick(self, event):

        ind = event.ind[0]
        artist = event.artist
        # print(f'Pick detected label: {artist.get_label()}')
        # print(f'Datapoint from pic {self.dataset[ind].image_name}')
        # print(f'Phase: {self.dataset[ind].phase}')

        if artist in self.good_artists:
            pic_num = self.data_pic_nums[0][ind]
        if artist in self.bad_artists:
            pic_num = self.data_pic_nums[1][ind]

        dset_pic_nums = np.array([get_pic_num(pic.image_path) for pic in self.dataset])
        print(f'detected pic: {pic_num}, phase: {self.dataset[np.where(dset_pic_nums==pic_num)[0][0]].phase}')

        # pic_num = int(self.dataset[ind].image_name.split('_')[-1])
        try:
            self.selected_img.set(pic_num)
            self.update_graph_from_options()
            self.fringe_window.destroy
            self.listbox.insert(tk.END, f'Detected pic: {pic_num}')
            self.listbox.see(tk.END)
        except:
            print('Error showing image!')

    def reattach_picker(self):
        self.fringe_fig.canvas.mpl_connect('pick_event', self.on_pick)

    def change_path(self):
        # Called upon change path button, handles directory selection and change
        global path
        path = filedialog.askdirectory(initialdir=os.getcwd())
        os.chdir(path)
        self.listbox.delete(0, tk.END)
        self.listbox.insert(tk.END, path)
        self.listbox.see(tk.END)
        print('Acquisition path:', path)
        if path != os.getcwd():
            print("ERROR: directory failure. See change_path()")


class AnalysisPage(tk.Frame):

    def __init__(self, parent, controller):

        # Initialize with buttons
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Analysis", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        frame1 = tk.Frame(self)
        frame1.pack()

        button1 = tk.Button(frame1, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        button2 = tk.Button(frame1, text="Load from directory", command=self.change_trialpath)
        button2.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
    
        self.imageDir = os.getcwd()

        self.imageDir_lbl = tk.Label(self, text='Current trial dir: ' + self.imageDir)
        self.imageDir_lbl.pack()


    def change_trialpath(self):
        # Called on load from directory button, handles selection and sets attribute for access across methods
        try:
            self.imageDir = filedialog.askdirectory(initialdir=os.getcwd())
            os.chdir(self.imageDir)
            self.imageDir_lbl.config(text='Current trial dir: ' + self.imageDir)
            self.imageDir_lbl.update_idletasks()

            self.populate_images()
        except Exception as e:
            print('change_trialpath() ERROR: ', e)

    def load_from_acquisition(self, path=os.getcwd()):
        # Called from Acquisition frame to pop up and populate analysis frame
        try:
            self.imageDir = path
            self.imageDir_lbl.config(text='Current trial dir: ' + self.imageDir)
            self.imageDir_lbl.update_idletasks()

            os.chdir(path)

            self.populate_images()
        except:
            print('No stored acquisition data... Manually select trial directory.')
    
    
    def analysis_graphs(self, npfile_lst, trialnum):
        # Called on trial button push, handles canvas and graphing

        return None

        self.frame1 = tk.Frame(self)
        self.frame1.pack()

        try:
            self.ax1.clear()
            self.ax2.clear()
        except: # On first call
            self.fig = Figure(figsize=(6,10), dpi=100)

            # self.ax1.set_ymargin(m=0.5)
            # self.fig.subplotpars.top = 5
            # self.fig.subplotpars.bottom = 0.1
            self.fig.subplotpars.hspace = 0.5

            self.ax1 = self.fig.add_subplot(211)
            self.ax2 = self.fig.add_subplot(212)

            self.vbar = tk.Scrollbar(self.frame1, orient=tk.VERTICAL)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
            self.canvas.get_tk_widget().config(width=800, height=1000, scrollregion=(0,0,800,800))
            self.canvas.get_tk_widget().config(yscrollcommand=self.vbar.set)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

            self.vbar.pack(side=tk.RIGHT, fill=tk.Y, expand=1)
            self.vbar.config(command=self.canvas.get_tk_widget().yview)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()
            self.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Get data
        means, ffts = cca.data_analysis(npfile_lst)
        freqs, xfft, yfft = ffts

        # Plot!
        self.fig.suptitle(trialnum, fontweight='bold')
        self.ax1.plot(np.arange(len(means[0])), means[0], label='x')
        self.ax1.plot(np.arange(len(means[0])), means[1], label='y')
        self.ax2.plot(freqs, xfft, label='x')
        self.ax2.plot(freqs, yfft, label='y')
        self.ax1.set_title("Bead positions")
        self.ax2.set_title("Position FFTs")
        self.ax1.legend()
        self.ax2.loglog()
        self.ax2.legend()
        self.ax1.set(xlabel='test')
        self.canvas.draw()


    def on_key_press(self, event):
        if event.key == 's':
            print("Saving plots...")
        key_press_handler(event, self.canvas, self.toolbar)

if __name__ == '__main__':  
    window = analysis_GUI()
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    window.mainloop()