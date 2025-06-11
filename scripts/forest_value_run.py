# Description: This file contains the code to run the analysis for the harvest schedule in a deterministic setting
# Author: Bruno Silva
# Date: 2024-01-14

import os
import sys


#Set the working directory
DirA = "C://Users//bk82288//OneDrive - University of Georgia//02 Papers//33 - Stochastic Fertilization//Mateus//03 Script"    
DirInput = "C://Users//bk82288//OneDrive - University of Georgia//02 Papers//33 - Stochastic Fertilization//Mateus//01 Input"

os.chdir(DirA)
sys.path.append(DirA)

#from harvest_schedule_v2 import terminal_value 
from scripts.forest_value_stand import *

#load the yield curves
yc = pd.read_csv(DirInput + "//yc_example.csv")
#cc <- volume of clearcut
#th <- volume of thinning
#load timber prices
tp = pd.read_csv(DirInput + "//timber_prices_example.csv")
costA = pd.read_csv(DirInput + "//cost_area_example.csv")
discount = 0.04
#annual costs
ac = 8
forest_value = forest_value(yc,tp,costA,discount,ac)
forest_value.optimal_rotation()[0] #optimal rotation --> age at clearcut according to the optimal LEV (check i)
forest_value.optimal_rotation()[1] # the entire dataset
forest_value.forest_value0() # forest value