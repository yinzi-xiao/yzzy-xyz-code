import sys
sys.path.append('..') 
import function

NUM_FILES = 100

data = function.reshape_data('yzzy_x1000bias',NUM_FILES)
function.save_simdata('yzzy_x1000bias.json',data)