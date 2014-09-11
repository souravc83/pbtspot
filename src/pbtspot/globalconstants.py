#import from standard modules
import os
import json

#local module imports


class GlobalConstants(object):
    
    def __init__(self):
        self._valuedict=None;
        self._jsonfilename='data/constants.json';
        self._read_from_file();
        
        
    
    def savefile(self):
        """
        This is the only public function.
        Saves json file to disc
        """
        self._write_to_file();
        return;
    
    def _read_from_file(self):
        if os.path.isfile(self._jsonfilename) == False:
            self._valuedict={};
            return;
        
        with open(self._jsonfilename,'r') as infile:
            self._valuedict=json.load(infile);
        return;
    
    def _write_to_file(self):
        """
        private function which saves the file with updated dict values
        """
        with open(self._jsonfilename,'w') as outfile:
            json.dump(self._valuedict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
        return;
    
    
    #all property functions
    @property
    def particle_shape(self):
        try:
            return self._valuedict['particle_shape'];
        except KeyError:
            raise KeyError("particle_shape not defined")
            
    @particle_shape.setter
    def particle_shape(self,value):
        if value != 'round' and\
           value != 'elongated':
            raise ValueError("Particle shape must be round or elongated");
        
        self._valuedict['particle_shape']=value;
        return;
    
    @property
    def global_window_width(self):
        try:
            return self._valuedict['global_window_width'];
        except KeyError:
            raise KeyError("window width not defined")
        
    @global_window_width.setter
    def global_window_width(self,value):
        if value<=0:
            raise ValueError("Width cannot be negative");
        
        self._valuedict['global_window_width']=value;
        return;
    
    @property
    def global_window_height(self):
        try:
            return self._valuedict['global_window_height'];
        except KeyError:
            raise KeyError("window height not defined")
        
    @global_window_height.setter
    def global_window_height(self,value):
        if value<=0:
            raise ValueError("Height cannot be negative");
        
        self._valuedict['global_window_height']=value;
        return;
    
    @property
    def image_type(self):
        try:
            return self._valuedict['image_type'];
        except KeyError:
            raise KeyError("image type not defined")
        
    @image_type.setter
    def image_type(self,value):
        if value!='A' and value!='B':
            raise ValueError("Image type must be A or B");
        
        self._valuedict['image_type']=value;
        return;
    
    @property 
    def maxdepth_tree(self):
        try:
            return self._valuedict['maxdepth_tree'];
        except KeyError:
            raise KeyError("maxdepth not defined")
            
    @maxdepth_tree.setter
    def maxdepth_tree(self,value):    
        self._valuedict['maxdepth_tree']=value;
        return;
    
    
    @property
    def no_pos_training_eg(self):
            try:
                return self._valuedict['no_pos_training_eg'];
            except KeyError:
                raise KeyError("no_pos_training_eg not defined")
            
    @no_pos_training_eg.setter
    def no_pos_training_eg(self,value):    
        self._valuedict['no_pos_training_eg']=value;
        return;
    
    @property
    def no_neg_training_eg(self):
            try:
                return self._valuedict['no_neg_training_eg'];
            except KeyError:
                raise KeyError("no_neg_training_eg not defined")
            
    @no_neg_training_eg.setter
    def no_neg_training_eg(self,value):    
        self._valuedict['no_neg_training_eg']=value;
        return;
    
    @property
    def no_pos_testing_eg(self):
            try:
                return self._valuedict['no_pos_testing_eg'];
            except KeyError:
                raise KeyError("no_pos_testing_eg not defined")
            
    @no_pos_testing_eg.setter
    def no_pos_testing_eg(self,value):    
        self._valuedict['no_pos_testing_eg']=value;
        return;
    
    @property
    def no_neg_testing_eg(self):
            try:
                return self._valuedict['no_neg_testing_eg'];
            except KeyError:
                raise KeyError("no_neg_testing_eg not defined")
            
    @no_neg_testing_eg.setter
    def no_neg_testing_eg(self,value):    
        self._valuedict['no_neg_testing_eg']=value;
        return;
    
    @property
    def prob_threshold(self):
            try:
                return self._valuedict['prob_threshold'];
            except KeyError:
                raise KeyError("prob_threshold not defined")
            
    @prob_threshold.setter
    def prob_threshold(self,value):    
        self._valuedict['prob_threshold']=value;
        return;
    
    
    @property
    def overfitting_e(self):
            try:
                return self._valuedict['overfitting_e'];
            except KeyError:
                raise KeyError("overfitting_e not defined")
            
    @overfitting_e.setter
    def overfitting_e(self,value):    
        self._valuedict['overfitting_e']=value;
        return;
    
        
    @property
    def training_rounds(self):
            try:
                return self._valuedict['training_rounds'];
            except KeyError:
                raise KeyError("training rounds not defined")
            
    @training_rounds.setter
    def training_rounds(self,value):    
        self._valuedict['training_rounds']=value;
        return;

    @property
    def is_lowres(self):
            try:
                return self._valuedict['is_lowres'];
            except KeyError:
                raise KeyError("is_lowres not defined")
            
    @is_lowres.setter
    def is_lowres(self,value):    
        self._valuedict['is_lowres']=value;
        return;
        
    @property
    def lowres_width(self):
            try:
                return self._valuedict['lowres_width'];
            except KeyError:
                raise KeyError("lowres_width not defined")
            
    @lowres_width.setter
    def lowres_width(self,value):    
        self._valuedict['lowres_width']=value;
        return;
    
    @property
    def lowres_height(self):
            try:
                return self._valuedict['lowres_height'];
            except KeyError:
                raise KeyError("lowres_height not defined")
            
    @lowres_height.setter
    def lowres_height(self,value):    
        self._valuedict['lowres_height']=value;
        return;    

    @property
    def gamma(self):
            try:
                return self._valuedict['gamma'];
            except KeyError:
                raise KeyError("gamma not defined")
            
    @gamma.setter
    def gamma(self,value):    
        self._valuedict['gamma']=value;
        return;    

    
