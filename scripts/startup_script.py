#Always run this script before other scripts to give default values to global variables
#ignore at your own peril
#do not change default values here
#change values in specific scripts
#these are default values and should NOT be touched

import pbtspot
from pbtspot import globalconstants

default_const=globalconstants.GlobalConstants();
#write all default values
default_const.particle_shape='round'
default_const.global_window_width=10;
default_const.global_window_height=10;
default_const.image_type='A'
#default values end here
default_const.savefile();