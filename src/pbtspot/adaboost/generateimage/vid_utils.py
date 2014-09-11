"""
@author Sourav Chatterjee
@Commonly used utilities, mostly to handle video files
"""

#import from standard library
import os
import shutil
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
#Todo: Change foldername from A to Images. Does rest of the code depend on this?


def createimages(fname,formatname="AVI"):
    """
    creates images from a video file
    @param fname: filename string
    @param formatname: video format string
    """
    #requires ffmpeg installed on the system
    filename=fname+"."+formatname;
    if os.path.exists('A')==False and os.path.isfile(filename)==True:
        os.system("mkdir A");
        combstr="ffmpeg -i " +filename+ " -r 7 A/A%d.jpg"
        try:
            os.system(combstr);
        except:
            print "Could not make images. Check if ffmpeg is installed.";
    else:
        print "Did not make images."
        if(os.path.exists('A')==True):
            print "Images already exist."
        if(os.path.isfile(filename)==False):
            print "File does not exist in this folder."
        
    
    return;

def createVideo(fname):
    """
    creates .avi Video file from a series of images
    @param fname:filename string
    """
    #requires ffmpeg installed on the system
    
    filename=fname+"_mod.avi";
    if os.path.exists('A')==True and os.path.isfile(filename)==False:
        combstr="ffmpeg -f image2 -r 7 -i A/A%d.jpg -vcodec libx264 -b:v 10M "+filename;
        try:
            os.system(combstr);
        except:
            print "Could not make video. Check if ffmpeg is installed.";
    else:
        print "Did not make images."
        if(os.path.exists('A')==False):
            print "Images do not exist in Folder called A."
        if(os.path.isfile(filename)==True):
            print " Video already exists"
    return; 


def cleanupimgdir(foldername='A'):
    """
    Cleans up the image directory by deleting images from the directory
    @param foldername: String giving name of folder
    """
    if os.path.exists(foldername)==True:
        shutil.rmtree(foldername);
    else:
        print "Folder does not exist"
    return;

        
def nooffiles(foldername='A'):
    """
    Calculates the number of files in the folder
    @param foldername: name of Folder String
    @retval files: int,number of files
    """
    if os.path.exists(foldername)==True:
        path,dirs,files=os.walk(foldername).next();
        return len(files);
    else:
        print "Folder does not exist."
        return 0;
                  



def savefile(xfile,yfile,fnamestr):
    #Any way to use *args here?
    """
    saves a file given lists for x and y values
    @param xfile: list containing x values
    @param yfile: list containing y values
    @param fnamestr: file name String
    """
    DAT=np.column_stack((xfile,yfile));
    np.savetxt(fnamestr,DAT,delimiter='\t');
    return;

def plotimage(filename):
    """
    plots an image with pixel values on mouse cursors shown
    @param filename: image file name String
    """
    if os.path.isfile(filename)==False:
        print "Image File not found.";
        return;
    img=ndimage.imread(filename,flatten=True);# flatten=true takes care of making image greyscale
    fig=plt.figure();
    ax=fig.add_subplot(111);
    ax.imshow(img,cmap=cm.Greys_r,interpolation='nearest');
    numrows,numcols=img.shape;
    
    def format_coord(x,y):
        col=int(x);
        row=int(y);
        
        if col>0 and col<numcols and row>0 and row<numrows:
            z=img[row,col];
            return 'x=%1.1f, y=%1.1f, z=%1.1f'%(x, y, z);
        else:
            return 'x=%1.1f, y=%1.1f'%(x, y);
            
    ax.format_coord=format_coord;
    plt.show()
    return;

def saveimage(img,filename,formatname='tiff'):
    """
    saves an numpy matrix as an image file
    @param: img : 2D numpy matrix
    @param: filename: string 
    @param: formatname: string
    """
    #check if 2D image
    shape=img.shape;

    if len(shape)!=2 and len(shape)!=3:
        raise ValueError("numpy array must be 2 or 3 dimensional")


    savename=filename+'.'+formatname;
    misc.imsave(savename,img);
    return



    
