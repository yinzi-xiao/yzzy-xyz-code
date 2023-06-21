import sys
sys.path.append('..') 
import function

NUM_FILES = 10

data = function.reshape_data('xyz_purez',NUM_FILES)
function.save_simdata('xyz_purez.json',data)