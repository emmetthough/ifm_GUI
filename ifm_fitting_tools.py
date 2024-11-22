import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

import os, glob
import cv2
from PIL import Image
from scipy.optimize import curve_fit

import tkinter as tk
from tkinter.filedialog import askopenfilename
# from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 

from time import sleep
from datetime import datetime

from pynput import keyboard

def rotate_image(image, angle):
    # given image (numpy array) and rotation angle (degrees), return rotated image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def load_image(path, xbnd=None, ybnd=None, show=False, blur=3, angle=7):
    # given image path and bounding box, return cropped and blurred image
    
    img = np.array(Image.open(path)) # load image
    img_rotate = rotate_image(img, angle) # rotate image
    
    if xbnd is not None and ybnd is not None:
        # crop image if bounds are given
        img_rotate = img_rotate[ybnd[0]:ybnd[1], xbnd[0]:xbnd[1]]
    
    # blur image with 3x3 kernal
    img_blur = cv2.GaussianBlur(img_rotate, (blur,blur), 0)
    
    if show:
        plt.imshow(img_blur, cmap='Greys_r')
        plt.grid(False)
        plt.show()
        
    return img_blur

def density(image, mag=1.8, l0=399e-9, px_size=7.4):
    # returns optical density from image
    
    s0 = 3*l0**2/(2*np.pi) # resonant cross section
    OD = -np.log(image/np.max(image))/s0 # normalized image to optical density
    
    return OD*(px_size*1e-6/mag)**2 # convert from pixels to um

def kill_fig_on_zoom(axes, fig):
    # print('zoom detected, killing window!')
    plt.close(fig)

def initialize_bounds0(fname, **kwargs):

    if 'angle' in kwargs.keys():
        angle = kwargs['angle']
    else:
        angle = 7

    if 'blur' in kwargs.keys():
        blur = kwargs['blur']
    else:
        blur = 3

    img_raw = load_image(fname, angle=angle, blur=blur) # load first image 

    fig,ax = plt.subplots(1, figsize=(8,6))
    ax.imshow(img_raw, cmap='Greys_r') #display
    fig.suptitle('Select Atom Region', fontsize=30)

    # func = lambda axes: print("New axis y-limits are", axes.get_ylim())

    cb_registry = ax.callbacks
    cid = cb_registry.connect('ylim_changed', lambda axes: kill_fig_on_zoom(axes, fig))

    plt.show()

    # define bounds for atom region
    xbnds0 = [round(b) for b in ax.get_xbound()]
    ybnds0 = [round(b) for b in ax.get_ybound()]

    # atom region
    img = load_image(fname, xbnd=xbnds0, ybnd=ybnds0, blur=blur, angle=angle, show=False)

    fig,ax = plt.subplots(1, figsize=(8,6))
    ax.imshow(img, cmap='Greys_r')
    fig.suptitle('Select Port 1', fontsize=30)

    cb_registry = ax.callbacks
    cid = cb_registry.connect('ylim_changed', lambda axes: kill_fig_on_zoom(axes, fig))

    plt.show()
    plt.close(fig)

    # define bounds for port 1, in full image coordinates
    xbnds1 = [xbnds0[0]+round(b) for b in ax.get_xbound()]
    ybnds1 = [ybnds0[0]+round(b) for b in ax.get_ybound()]

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

    cb_registry = ax.callbacks
    cid = cb_registry.connect('ylim_changed', lambda axes: kill_fig_on_zoom(axes, fig))

    plt.show()
    plt.close(fig)

    xbnds2 = [xbnds0[0]+round(b) for b in ax.get_xbound()]
    ybnds2 = [ybnds0[0]+round(b) for b in ax.get_ybound()]

    return [(xbnds0,ybnds0), (xbnds1,ybnds1), (xbnds2,ybnds2)]


class Bounds():

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

def on_enter(event, fig=None):
    if event.key == 'enter':
        plt.close(fig)

def on_select(eclick, erelease, bounds=None):
        """
        Store rectangle bounds when selection is made.
        """
        bounds.xmin = int(min(eclick.xdata, erelease.xdata))
        bounds.xmax = int(max(eclick.xdata, erelease.xdata))
        bounds.ymin = int(min(eclick.ydata, erelease.ydata))
        bounds.ymax = int(max(eclick.ydata, erelease.ydata))

def initialize_bounds(fname, **kwargs):

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

    fig,ax = plt.subplots(1, figsize=(8,6))
    ax.imshow(img_raw, cmap='Greys_r') #display
    fig.suptitle('Select Atom Region', fontsize=30)

    rect_selector = RectangleSelector(
        ax, lambda eclick, erelease: on_select(eclick, erelease, bounds=full_bounds),
        drawtype='box',
        useblit=True,
        props=dict(edgecolor='black', alpha=0.3, fill=False),
        interactive=True
    )

    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', lambda event: on_enter(event, fig=fig))

    plt.show()

    # define bounds for atom region
    xbnds0 = [round(full_bounds.xmin), round(full_bounds.xmax)]
    ybnds0 = [round(full_bounds.ymin), round(full_bounds.ymax)]

    # atom region
    img = load_image(fname, xbnd=xbnds0, ybnd=ybnds0, blur=blur, angle=angle, show=False)

    p1_bounds = Bounds()

    fig,ax = plt.subplots(1, figsize=(8,6))
    ax.imshow(img, cmap='Greys_r')
    fig.suptitle('Select Port 1', fontsize=30)

    rect_selector = RectangleSelector(
        ax, lambda eclick, erelease: on_select(eclick, erelease, bounds=p1_bounds),
        drawtype='box',
        useblit=True,
        props=dict(edgecolor='blue', alpha=0.3, fill=False),
        interactive=True
    )

    fig.canvas.mpl_connect('key_press_event', lambda event: on_enter(event, fig=fig))

    plt.show()
    plt.close(fig)

    # define bounds for port 1, in full image coordinates
    xbnds1 = [xbnds0[0]+round(b) for b in [p1_bounds.xmin, p1_bounds.xmax]]
    ybnds1 = [ybnds0[0]+round(b) for b in [p1_bounds.ymin, p1_bounds.ymax]]

    p2_bounds = Bounds()

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

    rect_selector = RectangleSelector(
        ax, lambda eclick, erelease: on_select(eclick, erelease, bounds=p2_bounds),
        drawtype='box',
        useblit=True,
        props=dict(edgecolor='green', alpha=0.3, fill=False),
        interactive=True
    )

    fig.canvas.mpl_connect('key_press_event', lambda event: on_enter(event, fig=fig))

    plt.show()
    plt.close(fig)

    xbnds2 = [xbnds0[0]+round(b) for b in [p2_bounds.xmin, p2_bounds.xmax]]
    ybnds2 = [ybnds0[0]+round(b) for b in [p2_bounds.ymin, p2_bounds.ymax]]

    return [(xbnds0,ybnds0), (xbnds1,ybnds1), (xbnds2,ybnds2)]


def OD_ints(img, xbnds, ybnds):

    dens = density(img[ybnds[0]:ybnds[1], xbnds[0]:xbnds[1]])
    xint = np.sum(dens, axis=0)
    yint = np.sum(dens, axis=1)

    return xint,yint

def gaussian(x, *params):
    # simple gaussian with background
    A, B, mu, sig = params
    return B+A*np.exp(-0.5*((x-mu)/sig)**2)

def atom_number(int_trace, show=False):

    pos = np.arange(int_trace.size)

    p0 = [np.max(int_trace)-int_trace[0], int_trace[0], pos[np.where(int_trace==np.max(int_trace))[0][0]], 10]

    popt,pcov = curve_fit(gaussian, pos, int_trace, p0=p0)

    if show:
        pos_range = np.linspace(pos[0], pos[-1], 250)
        fig,ax = plt.subplots(1)
        ax.plot(pos, int_trace, 'o', label='data')
        ax.plot(pos_range, gaussian(pos_range, *popt), label='fit')
        ax.legend()
        plt.show()

    return popt[0]*popt[-1]

def fit_gaus_func(int_trace, save=False, fname='blank.png', title=None, show=False):

    pos = np.arange(int_trace.size)

    A0 = np.max(int_trace)-int_trace[0]
    B0 = int_trace[0]
    mu0 = pos[np.where(int_trace==np.max(int_trace))[0][0]]
    sig0 = np.abs(mu0-pos[np.where(int_trace-B0>=0.5*A0)[0][0]])

    p0 = [A0, B0, mu0, sig0]

    popt,pcov = curve_fit(gaussian, pos, int_trace, p0=p0, maxfev=2500)

    res = int_trace-gaussian(np.arange(int_trace.size),*popt)

    if save:
        pos_range = np.linspace(pos[0], pos[-1], 250)
        fig,ax = plt.subplots(1)
        ax.plot(pos, int_trace, 'o', label='data')
        ax.plot(pos_range, gaussian(pos_range, *popt), label='fit')
        ax.legend()
        if title != None:
            ax.set_title(title)
        plt.savefig(fname, dpi=150)

        if not show:
            plt.close(fig)

    return popt,pcov

class fileSelectorGUI(tk.Tk):



    def __init__(self):

        # Initialize
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, "File Selector")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.geometry('200x100')

        button = tk.Button(self, text="Select Initial Shot", command=self.get_file)
        button.pack()

        self.fname = None

    def get_file(self):
        self.fname = askopenfilename(initialdir='.')
        print('Got file!: ', self.fname)
        tk.Tk.quit(self)
        tk.Tk.destroy(self)

class atomPicture():

    def __init__(self, image_path, atom_bnds, port_bnds, **kwargs):

        if 'angle' in kwargs.keys():
            self.angle = kwargs['angle']
        else:
            self.angle = 7
        
        if 'blur' in kwargs.keys():
            self.blur = kwargs['blur']
        else:
            self.blur = 3

        self.image_path = image_path
        self.image_name = self.image_path.split('/')[-1].split('.tif')[0]

        self.atom_xbnds = atom_bnds[0]
        self.atom_ybnds = atom_bnds[1]

        self.p1_xbnds = port_bnds[0][0]
        self.p1_ybnds = port_bnds[0][1]

        self.p2_xbnds = port_bnds[1][0]
        self.p2_ybnds = port_bnds[1][1]

        self.full_img = load_image(self.image_path, xbnd=None, ybnd=None, angle=self.angle, blur=self.blur)
        self.atom_arr = load_image(self.image_path, xbnd=self.atom_xbnds, ybnd=self.atom_ybnds, angle=self.angle, blur=self.blur)

        self.fullxint, self.fullyint = OD_ints(self.full_img, self.atom_xbnds, self.atom_ybnds)
        self.p1xint, self.p1yint = OD_ints(self.full_img, self.p1_xbnds, self.p1_ybnds)
        self.p2xint, self.p2yint = OD_ints(self.full_img, self.p2_xbnds, self.p2_ybnds)

        self.phase = None
        self.flag = False

        self.fits = {}


    def fit_gaus(self, axis='x', type='full', save=False, show=False, fits_path=None):

        img_name = self.image_path.split('/')[-1].split('.tif')[0]

        if type == 'full':
            if axis == 'x':
                data1 = self.fullxint
                data2 = None
                title1 = img_name + '\n' + 'full xint'
                if fits_path is None:
                    fname1 = './fits/'+ img_name + '_full_xint.png'
                else:
                    fname1 = os.path.join(fits_path, img_name+'_full_xint.png')
            else:
                data1 = self.fullyint
                data2 = None
                title1 = img_name + '\n' + 'full yint'

                if fits_path is None:
                    fname1 = './fits/'+ img_name + '_full_yint.png'
                else:
                    fname1 = os.path.join(fits_path, img_name+'_full_yint.png')
        
        elif type == 'ports':
            if axis == 'x':
                data1 = self.p1xint
                data2 = self.p2xint
                title1 = img_name + '\n' + 'port 1 xint'
                title2 = img_name + '\n' + 'port 2 xint'

                if fits_path is None:
                    fname1 = './fits/'+ img_name + '_p1_xint.png'
                    fname2 = './fits/'+ img_name + '_p2_xint.png'
                else:
                    fname1 = os.path.join(fits_path, img_name+'_p1_xint.png')
                    fname2 = os.path.join(fits_path, img_name+'_p2_xint.png')

            else:
                data1 = self.p1yint
                data2 = self.p2yint
                title1 = img_name + '\n' + 'port 1 yint'
                title2 = img_name + '\n' + 'port 2 yint'

                if fits_path is None:
                    fname1 = './fits/'+ img_name + '_p1_yint.png'
                    fname2 = './fits/'+ img_name + '_p2_yint.png'
                else:
                    fname1 = os.path.join(fits_path, img_name+'_p1_yint.png')
                    fname2 = os.path.join(fits_path, img_name+'_p2_yint.png')

        popt1, pcov1 = fit_gaus_func(data1, save=save, fname=fname1, show=show, title=title1)

        if data2 is not None:
            popt2, pcov2 = fit_gaus_func(data2, save=save, fname=fname2, show=show, title=title2)

        if type == 'full':
            if axis == 'x':
                self.fits['full_x_params'] = popt1,pcov1
                self.fits['full_Nx'] = popt1[0]*popt1[-1]
                self.full_Nx = popt1[0]*popt1[-1]
            if axis == 'y':
                self.fits['full_y_params'] = popt1,pcov1
                self.fits['full_Ny'] = popt1[0]*popt1[-1]
                self.full_Ny = popt1[0]*popt1[-1]
        
        if type == 'ports':
            if axis == 'x':
                self.fits['p1_x_params'] = popt1,pcov1
                self.fits['p1_Nx'] = popt1[0]*popt1[-1]
                self.p1_Nx = popt1[0]*popt1[-1]

                self.fits['p2_x_params'] = popt2,pcov2
                self.fits['p2_Nx'] = popt2[0]*popt2[-1]
                self.p2_Nx = popt2[0]*popt2[-1]
            if axis == 'y':
                self.fits['p1_y_params'] = popt1,pcov1
                self.fits['p1_Ny'] = popt1[0]*popt1[-1]
                self.p1_Nx = popt1[0]*popt1[-1]

                self.fits['p2_y_params'] = popt2,pcov2
                self.fits['p2_Ny'] = popt2[0]*popt2[-1]
                self.p2_Nx = popt2[0]*popt2[-1]

    def show_full_image(self, show=False):

        trsfm_xbnds1 = (self.p1_xbnds[0]-self.atom_xbnds[0], self.p1_xbnds[1]-self.atom_xbnds[0])
        trsfm_ybnds1 = (self.p1_ybnds[0]-self.atom_ybnds[0], self.p1_ybnds[1]-self.atom_ybnds[0])
        trsfm_xbnds2 = (self.p2_xbnds[0]-self.atom_xbnds[0], self.p2_xbnds[1]-self.atom_xbnds[0])
        trsfm_ybnds2 = (self.p2_ybnds[0]-self.atom_ybnds[0], self.p2_ybnds[1]-self.atom_ybnds[0])

        fig,ax = plt.subplots(2,2, figsize=(8,8))
        ax[0,0].imshow(self.atom_arr, cmap='Greys_r')
        ax[0,0].set_xlim(0,self.atom_xbnds[1]-self.atom_xbnds[0])
        ax[0,0].set_ylim(self.atom_ybnds[1]-self.atom_ybnds[0], 0)

        port1 = Rectangle(
        (trsfm_xbnds1[0], trsfm_ybnds1[0]),                 # Bottom-left corner (x, y)
        self.p1_xbnds[1]-self.p1_xbnds[0],                       # Width of the rectangle
        self.p1_ybnds[1]-self.p1_ybnds[0],                      # Height of the rectangle
        facecolor='none',        # Transparent fill
        edgecolor='blue',        # Color of the border
        linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
        linewidth=2,             # Border width
        alpha=0.5,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
        label='Port 1'
        )     

        port2 = Rectangle(
        (trsfm_xbnds2[0], trsfm_ybnds2[0]),                 # Bottom-left corner (x, y)
        self.p2_xbnds[1]-self.p2_xbnds[0],                       # Width of the rectangle
        self.p2_ybnds[1]-self.p2_ybnds[0],                      # Height of the rectangle
        facecolor='none',        # Transparent fill
        edgecolor='green',        # Color of the border
        linestyle='--',          # Dotted border style ('--' for dashed, ':' for dotted)
        linewidth=2,             # Border width
        alpha=0.5,               # Transparency of the border (0 = fully transparent, 1 = fully opaque)
        label='Port 2'
        )      

        ax[0,0].add_patch(port1)
        ax[0,0].add_patch(port2)

        ax[0,1].plot(self.p1yint-np.mean(self.p1yint[:10]), np.arange(trsfm_ybnds1[0],trsfm_ybnds1[1]), c='b', lw=2.5)
        ax[0,1].plot(self.p2yint-np.mean(self.p1yint[:10]), np.arange(trsfm_ybnds2[0],trsfm_ybnds2[1]), c='g', lw=2.5)
        ax[0,1].plot(self.fullyint-np.mean(self.fullyint[:10]), np.arange(0, self.atom_ybnds[1]-self.atom_ybnds[0]), c='k', alpha=0.6, lw=1)

        ax[0,1].invert_yaxis()
        ax[0,1].sharey(ax[0,0])

        ax[1,0].plot(np.arange(trsfm_xbnds1[0],trsfm_xbnds1[1]), self.p1xint-np.mean(self.p1xint[:10]), c='b', lw=2.5)
        ax[1,0].plot(np.arange(trsfm_xbnds2[0],trsfm_xbnds2[1]), self.p2xint-np.mean(self.p2xint[:10]), c='g', lw=2.5)
        ax[1,0].plot(np.arange(0, self.atom_xbnds[1]-self.atom_xbnds[0]), self.fullxint-np.mean(self.fullxint[:10]), c='k', alpha=0.6, lw=1)
        ax[1,0].sharex(ax[0,0])

        ax[1,1].axis('off')

        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticklabels([])

        ax[1,0].set_xticklabels([])
        ax[1,0].set_yticklabels([])

        ax[0,1].set_xticklabels([])
        ax[0,1].set_yticklabels([])

        # Get the figure size in inches
        fig_width, fig_height = fig.get_size_inches()

        # Get the bounding box of the first subplot
        bbox = ax[0, 0].get_position()  # Get the position of the first subplot

        # Calculate subplot width and height in inches
        subplot_width = bbox.width * fig_width
        subplot_height = bbox.height * fig_height

        fig.set_size_inches(2*subplot_width*1.5, 2*subplot_height*1.2)
        fig.suptitle(self.image_name)

        fig.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
        else:
            return fig,ax
        
