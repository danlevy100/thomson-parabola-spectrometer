# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:45:32 2020

@author: Danle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.constants import c, e, m_p
import os
import re
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QInputDialog

width  = 3.487
height = width / 1.2

plt.rc('font', family='serif', serif='Times')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)

class Curve():
    def __init__(self, curve_x, curve_y, img_vals_mean, img_vals_stdev, energies, QoverM, thickness):
        self.x = curve_x
        self.y = curve_y
        self.img_vals_mean = img_vals_mean
        self.img_vals_stdev = img_vals_stdev
        self.energies = energies
        self.QoverM = QoverM
        self.thickness = thickness        
        self.signal_integral = sum(img_vals_mean)
        
    #def calcCutoff(self, bgr, thresh_factr=1.1):
    #    threshold = thresh_factr*bgr
    #    cutoff_index = max([index for index, value in enumerate(self.img_vals) \
    #                        if value>threshold*(self.thickness*2+1)])
    #    self.signal_integral = sum(self.img_vals_mean[0:cutoff_index])
    #    self.cutoff = self.energies[cutoff_index]        
        
class Image():
    def __init__(self, dir_, file):
        self.img = plt.imread(f'{dir_}/{filename}')
        self.px2mm = None
        self.zero_xy = None        
        self.bgr = 0
    
    def rotate(self, deg):
        self.img = ndimage.rotate(self.img, deg)

    def setBgr(self, x1, x2, y1, y2):
        self.bgr = int(np.mean(self.img[y1:y2,x1:x2]))        

class TPS():        
    def __init__(self, LB, LE, Di, Bfield, Efield):
        
        self.LB = LB
        self.LE = LE
        self.Di = Di
        self.Bfield = Bfield
        self.Efield = Efield      
        
        self.curves = []
        self.bgr_curve = None
        self.image = None
        
        # Relativistic velocity of particle with initial energy Epart (in MeV) 
        # and QoverM charge-mass ratio (in e/m_proton)
        self.v = lambda Epart, QoverM: (1 - 1/(Epart*QoverM/938.27 + 1)**2)**0.5*c 
    
        # relativistic gamma factor
        self.gamma = lambda Epart, QoverM: 1/((1 - (self.v(Epart, QoverM)/c)**2))**0.5
        
        # Proton cyclotron radius in m
        self.R = lambda Epart, Bfield, QoverM: \
            self.gamma(Epart, QoverM)*m_p*self.v(Epart, QoverM)/(QoverM*e*Bfield)
        
        # Exit angle after magnets, xz plane.
        # (not zx plane. z is propagation direction, x is up.)
        #self.theta = lambda Epart, Bfield, QoverM: np.arccos(LB/self.R(Epart, Bfield, QoverM))
        self.theta = lambda Epart, Bfield, QoverM: np.pi/2-self.LB/self.R(Epart,Bfield,QoverM)
        # Exit angle after electric field, xy plane
        self.phi = lambda Epart, Bfield, Efield, QoverM: \
              np.arctan((self.v(Epart, QoverM)**2*np.sin(self.theta(Epart, Bfield, QoverM))**2 \
              *m_p*self.gamma(Epart, QoverM)/(QoverM*e*Efield*LE)))
        
        # Position after magnets, meters 
        self.xB = lambda Epart, Bfield, QoverM: self.R(Epart, Bfield, QoverM) - \
             (self.R(Epart, Bfield, QoverM)**2 - LB**2)**0.5 * np.sign(Bfield)
        self.yB = 0
        self.zB = LB
        
        # Position after electrodes, meters
        self.xE = lambda Epart, Bfield, QoverM: self.xB(Epart, Bfield, QoverM) + \
             LE/np.tan(self.theta(Epart, Bfield, QoverM))
        self.yE = lambda Epart, Bfield, Efield, QoverM: \
             (QoverM*e*Efield*LE**2)/(2*np.sin(self.theta(Epart, Bfield, QoverM))**2* \
             self.v(Epart, QoverM)**2*m_p*self.gamma(Epart, QoverM))
        self.zE = LB + LE
        
        # Position at screen, meters. Di is screen (MCP) distance in meters from beginning of magnets
        self.xD = lambda Epart, Bfield, Di, QoverM: \
             self.xE(Epart, Bfield, QoverM) + (Di-LE-LB)/np.tan(self.theta(Epart, Bfield, QoverM))
        self.yD = lambda Epart, Bfield, Efield, Di, QoverM: \
             self.yE(Epart, Bfield, Efield, QoverM) + \
             (Di-LE-LB)/np.tan(self.phi(Epart, Bfield, Efield, QoverM))
        self.zD = lambda Di: Di                       
        
    def addCurve(self, E_min, E_max, thickness, QoverM=1, bgr_curve_yoffset = 0):
        Epart = np.linspace(E_min,E_max,500)
        mm2px = 1/self.image.px2mm      
        
        xDpix = lambda Epart,Bfield,Di,QoverM: \
            self.image.zero_xy[0] - self.xD(Epart,Bfield,Di,QoverM)*10**3 / mm2px
        yDpix = lambda Epart,Bfield,Efield,Di,QoverM: \
            self.image.zero_xy[1] - self.yD(Epart,Bfield,Efield,Di,QoverM)*10**8 / mm2px
        # The minus sign is because of the 0,0 position
                    
        curve_x = xDpix(Epart,self.Bfield,self.Di,QoverM)
        curve_y = yDpix(Epart,self.Bfield,self.Efield,self.Di,QoverM) + bgr_curve_yoffset
        
        img_vals = [self.image.img[curve_y.astype(int)+y_thick, \
                 curve_x.astype(int)] \
          for y_thick in range(-thickness,thickness+1)]
        
        img_vals = np.array(img_vals)
        
        summed_curve = np.sum(img_vals, axis=0)
        mean_curve = summed_curve/img_vals.shape[0]
        stdev_curve = np.std(img_vals, axis=0)
        curve = Curve(curve_x, curve_y, mean_curve, stdev_curve, Epart, QoverM, thickness)                
        
        #try:
        #    curve.calcCutoff(self.image.bgr)            
        #except:
        #    pass
        
        if not bgr_curve_yoffset:
            self.curves.append(curve)
        else:
            self.bgr_curve = curve
    
    def showImage(self, showCurves=True):              
        
        fig, ax = plt.subplots()
        plt.imshow(self.image.img)
        
        try:
            plt.clim(self.image.img_clim[0], self.image.img_clim[1])
        except:
            pass
        
        if showCurves: 
            
            # A couple of functions for displaying energy tooltips
            def update_annot(ind):
            
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos            
                            
                text = f'{Epart[ind["ind"][0]]:.2f} MeV'
                
                annot.set_text(text)            
                annot.get_bbox_patch().set_alpha(0.4)        
            
            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()       
                       
            annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->", color='white'))
            annot.set_visible(False)
            
            for curve in self.curves:            
                Epart = curve.energies
                sc = plt.scatter(curve.x, curve.y, color='red', s=5)                                      
                plt.plot(curve.x, curve.y+curve.thickness, 
                         color='lightblue', linestyle='--')
                plt.plot(curve.x, curve.y-curve.thickness,
                         color='lightblue', linestyle='--')                
                
                fig.canvas.mpl_connect("motion_notify_event", hover)
                
            Epart = curve.energies
            
            if self.bgr_curve:
                #sc = plt.scatter(self.bgr_curve.x, self.bgr_curve.y, color='green', s=5)
                plt.plot(self.bgr_curve.x, self.bgr_curve.y+self.bgr_curve.thickness, 
                         color='blue', linestyle=':')
                plt.plot(self.bgr_curve.x, self.bgr_curve.y-self.bgr_curve.thickness,
                         color='blue', linestyle=':')     
        
    def plotEspectra(self, subtract_bgr_curve, save=False, dir_=None, filename=None):
        for curve in self.curves:            
                      
            fig, ax = plt.subplots(1)          
            if subtract_bgr_curve:
                curve.img_vals_mean += -self.bgr_curve.img_vals_mean               
            
            dNdE = 250*curve.img_vals_mean*curve.energies**(-3/2)            
            
            # The Cutoff energy is: the minimum energy where mean_signal-bgr_std_signal < mean_noise.
            # bgr_std_signal is the standard deviation of the values over the thickness of the background curve.             
            # If that minimum energy is the minimum possible, set cutoff to 0.
            #cutoff = curve.energies[curve.img_vals_mean-curve.img_vals_stdev<=0].min()            
            curve.cutoff = curve.energies[curve.img_vals_mean-self.bgr_curve.img_vals_stdev<=0].min()
            curve.cutoff_ind = np.argwhere(curve.energies==curve.cutoff).min()
            curve.brightness = np.sum(dNdE[0:curve.cutoff_ind])
            #cutoff = curve.energies[curve.img_vals_mean-self.bgr_curve.img_vals_stdev-curve.img_vals_stdev<=0].min()
            if curve.cutoff == curve.energies[0]:
                curve.cutoff = 0            
            
            ax.plot(curve.energies, dNdE)            
            
            #plt.figure()
            #plt.plot(curve.energies, dNdE)
            
            #plt.axvline(x=curve.cutoff, ls='--', color='gray')
            ax.set_yscale('log')
            ax.set_xlabel('MeV')
            ax.set_ylabel('$d^2N/dEd\Omega$ [arb.u.]')
            #ax.set_ylim(1)
            #ax.set_xlim(2)
            if curve.QoverM == 1:
                ax.set_title('$H^+$ energy spectrum')
            #else:
            #    ax.set_title(f'$q/m$ = {curve.QoverM:.2f} [$e/m_e$]')    

            if save:
                if curve.cutoff>0:                                        
                    
                    plt.savefig(f'{dir_}/{filename[0:-5]}_protons_new.png')
                    a = np.array([curve.energies, dNdE]).T
                    np.savetxt(f'{dir_}/{filename[0:-5]}_spectrum_new.txt', a,
                        fmt='%1.4f', header='Energy [MeV] d(counts)/dE')                    
                    
                    #file_spectrum.close()               
                plt.close(fig)
                
            #fig.set_size_inches(width, height)
            #fig.subplots_adjust(left=0.1, bottom=.15, right=0.95, top=0.9)
            #plt.savefig('2.pdf', dpi=300)

        return curve.cutoff, curve.brightness
       
                
if __name__=='__main__':        
       
    app = QApplication(sys.argv)         

    WIS = False
    SN = True

    modes = ("Single image display and analysis", "Analyze and save spectra of all files in directory")
    mode, _ = QInputDialog.getItem(None, "Choose mode of operation","Mode:", modes, 0, False)    
    zero_x, _ = QInputDialog.getInt(None, "Zero point x coordinate","Enter zero point x coordinate (0-65535):", 411, 0, 2**16-1, 1)
    zero_y, _ = QInputDialog.getInt(None, "Zero point y coordinate","Enter zero point y coordinate (0-65535):", 480, 0, 2**16-1, 1)
    rot_angle, _ = QInputDialog.getDouble(None, "Image rotation angle","Enter rotation angle", -1.4, -360, 360, 1)

    if WIS:
    
        '''
        LB: Magnet length [m]
        LE: Electric field plate length [m]
        Di: Distance from beginning of magnet to MCP [m]
        Bfield: Strength of magnetic field in the middle [Tesla]
        Efield: Strength of electric field in the middle [kV/cm]
        '''        
        
        #WIS = TPS(LB=0.05, LE=0.05, Di=0.33, Bfield=0.38, Efield=6)      
        WIS = TPS(LB=0.05, LE=0.05, Di=0.36, Bfield=0.38, Efield=5.5)      
        
        # Display image 
        #dir_ = '../../../Measurements/2020/Weizmann TPS Jan 2020/20200119_2um_Ti/TPS_hamamatsu'
        #filename = '2um_Ti_00010.tif'
        
        dir_ = 'c:/users/danle/desktop/'
        filename = '1_00105.tif'
        #filename = '1_00079.tif'
        
        #dir_ = 'C:/Users/Danle/Desktop/20210309'
        #filename = '1_00020.tif'
        
        WIS.image = Image(dir_, filename)
        WIS.image.img_clim = [100, 1200] # Color range for contrast    
        
        #WIS.image.px2mm = (1850-116)/40
        #WIS.image.zero_xy = [1670, 910]
        #WIS.image.zero_xy = [1760, 1080]
                
        WIS.image.zero_xy = [3494, 975]
        
        #WIS.image.px2mm = (2048)/50
        WIS.image.px2mm = (1994/(55))
        #WIS.image.rotate(-0.4) 
        
        WIS.image.setBgr(950,1300,400,1200) # Set background level of image
            
        E_min, E_max = 2, 12 # Energy range in MeV
        thickness = 8 # Thickness of curve, used for integral    
        
        #WIS.addCurve(E_min, E_max, thickness, QoverM=1)
        
        # QoverM: charge over mass in [e/m_e]
        WIS.addCurve(E_min, E_max, thickness, QoverM=1)
        WIS.addCurve(E_min, E_max, thickness=15, QoverM=1, bgr_curve_yoffset = 50)
        
        WIS.showImage(showCurves=True)
        
        WIS.plotEspectra(subtract_bgr_curve = True)

        plt.show()
        
    if SN:
        #SN = TPS(LB=0.015, LE=0.05, Di=0.21, Bfield=-0.32, Efield=2.12)
        SN = TPS(LB=0.015, LE=0.05, Di=0.21, Bfield=-0.32, Efield=1.9)
        
        if mode == "Single image display and analysis":
            #dir_ = '../../../Measurements/2019/Salle Noire May2019/20190604/ions'
            dir_ = '../../../Measurements/2020/Salle Noire/ions'
            filename = 'tir186.tiff'            

            SN.image = Image(dir_, filename)
            #SN.image.rotate(-3) # 2019
            SN.image.rotate(rot_angle) # 2020
            SN.image.img_clim = [0, 5000] # Color range for contrast    
            SN.image.px2mm = (1004-90)/40            
            #SN.image.zero_xy = [326, 465] #20190604
            #SN.image.zero_xy = [346, 470] #20190531
            #SN.image.zero_xy = [418, 480] #2020
            SN.image.zero_xy = [zero_x, zero_y]

            #E_min, E_max = 0.075, 0.5 # Energy range in MeV, 2019
            E_min, E_max = 0.1, 0.6 # Energy range in MeV, 2020
            thickness = 5 # Thickness of curve, used for integral    
            SN.addCurve(E_min, E_max, thickness, QoverM=1/12)
            SN.addCurve(E_min, E_max, thickness=15, QoverM=1, bgr_curve_yoffset = 50)

            SN.showImage(showCurves=True)
            
            SN.plotEspectra(subtract_bgr_curve = True)

            plt.show()

        elif mode == "Analyze and save spectra of all files in directory":            
    
            # Get present working directory
            pwd = os.path.dirname(os.path.realpath(__file__))
            # initialize all parameters
            dir_ = QFileDialog.getExistingDirectory(None, \
                'Select a folder containing TP images:', pwd)           
            
            #dir_ = '../../../Measurements/2019/Salle Noire May2019/20190604/ions'
            #dir_ = '../../../Measurements/2019/Salle Noire May2019/20190528/ions'

            # Get all .tiff files in directory
            img_files = [name for name in os.listdir(dir_) \
                if os.path.isfile(os.path.join(dir_, name)) and name[-4:]=='tiff']
        
            # Sort the list according to the numbers in the file names
            img_files = sorted(img_files, key = lambda a: int(re.compile('(-?\d*\.?\d+)').split(a)[1]))
            # Extract the image numbers
            img_numbers=[int(re.compile('(-?\d*\.?\d+)').split(x)[1]) for x in img_files]                        
           
            cutoffs, brightnesses = [], []
            for filename in img_files:
                SN.image = Image(dir_, filename)
                #SN.image.zero_xy = [326, 465] #20190604
                #SN.image.zero_xy = [346, 470] #20190531
                SN.image.zero_xy = [411, 480] #2020
                
                #E_min, E_max = 0.075, 0.5 # Energy range in MeV
                E_min, E_max = 0.1, 0.5 # Energy range in MeV, 2020
                thickness = 5 # Thickness of curve, used for integral                            
                #SN.image.rotate(-3) #2019
                SN.image.rotate(-1.4) #2020
                #SN.image.img_clim = [0, 5000] # Color range for contrast    
                SN.image.px2mm = (1004-90)/40

                SN.addCurve(E_min, E_max, thickness, QoverM=1)
                SN.addCurve(E_min, E_max, thickness=15, QoverM=1, bgr_curve_yoffset = 50)

                #SN.showImage(showCurves=True)                
                cutoff, brightness = \
                    SN.plotEspectra(subtract_bgr_curve = True, save=True, dir_=dir_, filename=filename)

                cutoffs.append(cutoff)
                brightnesses.append(brightness)
            
                SN.curves = []            
            
            
            img_numbers = np.array(img_numbers).astype(str)
            cutoffs = np.array(cutoffs)
            cutoffs = np.around(cutoffs, decimals=4)
            brightnesses = np.array(brightnesses).astype(int)            

            a = np.array([img_numbers, cutoffs, brightnesses]).T
            np.savetxt(f'{dir_}/ions_new.txt', a,
                fmt = '%s', header='Shot number|Cutoff energy [MeV]|Signal brightness [arb. u.]')      

