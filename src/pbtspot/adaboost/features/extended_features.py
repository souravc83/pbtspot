#import from std modules
from __future__ import division

#import from local modules
from basefeature import Rectangle,BaseFeature


class Ext_Edge_1(BaseFeature):
    """
    Figure 1(c) in LeinHardt paper
    """
    
    def __init__(self):
        self.width_incr=2;
        self.height_incr=1;
        self.weights=[1,-1];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended EdgeFeature1"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width/2,height,self.weights[0]);
        rect2=Rectangle(x_val+width/2,y_val+width/2,width/2,height,self.weights[1]);
        self.rectlist=[rect1,rect2];
        return;

class Ext_Edge_2(BaseFeature):
    """
    Figure 1:1(d) in LeinHardt paper
    """
    
    def __init__(self):
        self.width_incr=1;
        self.height_incr=2;
        self.weights=[-1,1];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended EdgeFeature2"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width,height/2,self.weights[0]);
        rect2=Rectangle(x_val-width/2,y_val-width/2,width,height/2,self.weights[1]);
        self.rectlist=[rect1,rect2];
        return;

class Ext_Line_1(BaseFeature):
    """
    Figure 1: 2(e) 
    """        
    
    def __init__(self):
        self.width_incr=1;
        self.height_incr=3;
        self.weights=[1,-2,1];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended LineFeature1"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width,height/3,self.weights[0]);
        rect2=Rectangle(x_val-height/3,y_val-height/3,width,height/3,self.weights[1]);
        rect3=Rectangle(x_val-2*height/3,y_val-2*height/3,width,height/3,self.weights[2]);
        
        self.rectlist=[rect1,rect2,rect3];
        return;
        
class Ext_Line_2(BaseFeature):
    """
    Figure 1: 2(f) 
    """        
    
    def __init__(self):
        self.width_incr=1;
        self.height_incr=4;
        self.weights=[1,-1,1];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended LineFeature2"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width,height/4,self.weights[0]);
        rect2=Rectangle(x_val-height/4,y_val-height/4,width,height/2,self.weights[1]);
        rect3=Rectangle(x_val-3*height/4,y_val-3*height/4,width,height/4,self.weights[2]);
        
        self.rectlist=[rect1,rect2,rect3];
        return;

class Ext_Line_3(BaseFeature):
    """
    Figure 1: 2(g) 
    """        
    
    def __init__(self):
        self.width_incr=3;
        self.height_incr=1;
        self.weights=[1,-2,1];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended LineFeature3"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width/3,height,self.weights[0]);
        rect2=Rectangle(x_val+width/3,y_val+width/3,width/3,height,self.weights[1]);
        rect3=Rectangle(x_val+2*width/3,y_val+2*width/3,width/3,height,self.weights[2]);
        
        self.rectlist=[rect1,rect2,rect3];
        return;

class Ext_Line_4(BaseFeature):
    """
    Figure 1: 2(h) 
    """        
    
    def __init__(self):
        self.width_incr=1;
        self.height_incr=4;
        self.weights=[1,-1,1];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended LineFeature2"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width/4,height,self.weights[0]);
        rect2=Rectangle(x_val+width/4,y_val+width/4,width/2,height,self.weights[1]);
        rect3=Rectangle(x_val+3*width/4,y_val+3*width/4,width/4,height,self.weights[2]);
        
        self.rectlist=[rect1,rect2,rect3];
        return;


class Ext_Center(BaseFeature):
    """
    Figure 1: 3(b)
    """
    
    def __init__(self):
        self.width_incr=1;
        self.height_incr=4;
        self.weights=[-1,-1,-1,-1,8];
        self.isRotated=True;
        
    def __repr__(self):
        return "Extended CenterFeature"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width/3,height,self.weights[0]);
        rect2=Rectangle(x_val+2*width/3,y_val+2*width/3,width/3,height,self.weights[1]);
        rect3=Rectangle(x_val-2*width/3,y_val+3*width/4,width/4,height,self.weights[2]);
        
        self.rectlist=[rect1,rect2,rect3];
        return;