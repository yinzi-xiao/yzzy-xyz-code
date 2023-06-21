import sys
sys.path.append('..') 
import function

NUM_FILES = 100

data = function.reshape_data('xyz_x300bias',NUM_FILES)
function.save_simdata('xyz_x300bias.json',data)