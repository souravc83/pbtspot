#import from std modules
from __future__ import division

#import from local modules
import basefeature
reload(basefeature)
from basefeature import Rectangle,BaseFeature


#These feature types are defined from Figure 3 in the paper
#Smal et al,"Quantitative Comparison of Spot Detection Methods in Fluorescence Microscopy"

class EdgeFeature_1(BaseFeature):
    
    def __init__(self):
        BaseFeature.__init__(self)
        self.width_incr=2;
        self.height_incr=1;
        self.weights=[1,-1];
        self.isRotated=False;
        
    def __repr__(self):
        return "EdgeFeature1"
        
    def generate_feature(self,x_val,y_val,width,height):    
        rect1=Rectangle(x_val,y_val,width/2,height,self.weights[0]);
        rect2=Rectangle(x_val+width/2,y_val,width/2,height,self.weights[1]);
        self.rectlist=[rect1,rect2];
        return;
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class EdgeFeature_2(BaseFeature):
        
    def __init__(self):
        BaseFeature.__init__(self)
        self.width_incr=1;
        self.height_incr=2;
        self.weights=[1,-1];
        self.isRotated=False;
    
    def __repr__(self):
        return "EdgeFeature2"
    
    def generate_feature(self,x_val,y_val,width,height):
        rect1=Rectangle(x_val,y_val,width,height/2,self.weights[0]);
        rect2=Rectangle(x_val,y_val+height/2,width,height/2,self.weights[1]);
        self.rectlist= [rect1,rect2];
        return;
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
class LineFeature_1(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.width_incr=3;
        self.height_incr=1;
        self.weights=[-1,2,-1]
        self.isRotated=False;
    
    def __repr__(self):
        return "LineFeature1"
    
    def generate_feature(self,x_val,y_val,width,height):
        rect1=Rectangle(x_val,y_val,width/3,height,self.weights[0]);
        rect2=Rectangle(x_val+width/3,y_val,width/3,height,self.weights[1]);
        rect3=Rectangle(x_val+2*width/3,y_val,width/3,height,self.weights[2]);
        self.rectlist= [rect1,rect2,rect3];
        return;
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class LineFeature_2(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.width_incr=1;
        self.height_incr=3;
        self.weights=[-1,2,-1]
        self.isRotated=False;
    
    def __repr__(self):
        return "LineFeature2"
    
    def generate_feature(self,x_val,y_val,width,height):
        rect1=Rectangle(x_val,y_val,width,height/3,self.weights[0]);
        rect2=Rectangle(x_val,y_val+height/3,width,height/3,self.weights[1]);
        rect3=Rectangle(x_val,y_val+2*height/3,width,height/3,self.weights[2]);
        self.rectlist= [rect1,rect2,rect3];
        return;
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class LineFeature_3(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self);
        self.width_incr=4;
        self.height_incr=1;
        self.weights=[-1,1,-1];
        self.isRotated=False;
    
    def __repr__(self):
        return "LineFeature3"
    
    def generate_feature(self,x_val,y_val,width,height):
        rect1=Rectangle(x_val,y_val,width/4,height,self.weights[0]);
        rect2=Rectangle(x_val+width/4,y_val,width/2,height,self.weights[1]);
        rect3=Rectangle(x_val+3*width/4,y_val,width/4,height,self.weights[2]);
        self.rectlist= [rect1,rect2,rect3];
        return;
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
class LineFeature_4(BaseFeature):
    
    def __init__(self):
        BaseFeature.__init__(self)
        self.width_incr=1;
        self.height_incr=4;
        self.weights=[-1,1,-1];
        self.isRotated=False;
    
    def __repr__(self):
        return "LineFeature4"

    def generate_feature(self,x_val,y_val,width,height):
        rect1=Rectangle(x_val,y_val,width,height/4,self.weights[0]);
        rect2=Rectangle(x_val,y_val+height/4,width,height/2,self.weights[1]);
        rect3=Rectangle(x_val,y_val+3*height/4,width,height/4,self.weights[2]);
        self.rectlist=[rect1,rect2,rect3];
        return;
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class CenterFeature(BaseFeature):
    #This is more complicated than the others;
    def __init__(self):
        BaseFeature.__init__(self)
        self.width_incr=3;#always odd no.
        self.height_incr=3;#always odd no.
        self.weights=[-1,-1,-1,-1,8];
        self.isRotated=False;
        
    def __repr__(self):
        return "CenterFeature"
        
    def generate_feature(self,x_val,y_val,width,height):
        #width should be equal to height
        #if width!= height:
        #    raise ValueError("Width must be equal to height.Width:%d, height:%d"%(width,height));
        if (width%3)!=0:
            raise ValueError("Width must be a multiple of 3. Width:%d"%width); 
        
        if (height%3)!=0:
            raise ValueError("Height must be a multiple of 3. Height:%d"%height)
        
        #set weight of the small spot so as to equal the areas
        
            
        #divide black rect into 4 rectanges to make life easier
        
        rect1=Rectangle(x_val,y_val,width,height/3,self.weights[0]);
        rect2=Rectangle(x_val,y_val+height/3,width/3,height/3,self.weights[1]);
        rect3=Rectangle(x_val,y_val+2*height/3,width,height/3,self.weights[2]);
        rect4=Rectangle(x_val+2*width/3,y_val+height/3,width/3,height/3,self.weights[3]);
       
        
        #finally small spot rectangle,this is the one with a different weight
        rect5=Rectangle(x_val+width/3, y_val+height/3,width/3,height/3,self.weights[4]);
        
        self.rectlist=[rect1,rect2,rect3,rect4,rect5];
        return;
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
