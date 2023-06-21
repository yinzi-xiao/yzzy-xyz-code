import sys 
sys.path.append("..") 
import function
from rotatedplanaryz import *
import logging
import os
import time
import argparse
import numpy as np
from qecsim import app
from qecsim import paulitools as pt
from qecsim.models.generic import BiasedDepolarizingErrorModel

start_time = time.time()

#----------------- set parameters -----------------------------
# set physical error probabilities
# error_probability_min, error_probability_max = 1001/2001
# error_probabilities = np.linspace(error_probability_min, error_probability_max, 1)
# set run times for each simulation
max_runs = 100
# save paths
save_path = os.getenv('SLURM_SUBMIT_DIR')
# get the task ID from jobscript for data storage
parser = argparse.ArgumentParser(description='task id')
parser.add_argument('--taskid', type=str, default = None)
args = parser.parse_args()
task_id = args.taskid

# create a list to store error probability and number of Pauli operators for failure chain
data = []

code = RotatedPlanarYZCode(77)
decoder = RotatedPlanarYZRNIIDMPSDecoder(chi=12)
error_model = BiasedDepolarizingErrorModel(bias=1000,axis='X')

logger = logging.getLogger(__name__)

# run for each error probability
error_probability = 1001/2001
prob_dist_list = []
prob_dist = error_model.probability_distribution(error_probability)
for i in range(code.n_k_d[0]):
    prob_dist_list.append(prob_dist)
    
# initialize rng
rng = np.random.default_rng()
error_paulis = []
for _ in range(max_runs):
    # generate a random error
    error_pauli = ''
    for i_qubit in range(code.n_k_d[0]):
        error_pauli += ''.join(rng.choice(('I', 'X', 'Y', 'Z'),
        size=1, p=prob_dist_list[i_qubit]))
    error_paulis.append(error_pauli)
    
for run in range(max_runs):
    error_pauli = error_paulis[run]
    error = pt.pauli_to_bsf(error_pauli)
    # transform error to syndrome
    syndrome = pt.bsp(error, code.stabilizers.T)
    # decode to find recovery
    recovery = decoder.decode(code, syndrome, prob_dist_list)
    # check if recovery is success or not
    # check if recovery communicate with stabilizers
    commutes_with_stabilizers = np.all(pt.bsp(recovery^error, code.stabilizers.T) == 0)
    if not commutes_with_stabilizers:
        log_data = {  # enough data to recreate issue
            # models
            'code': repr(code), 'decoder': repr(decoder),
            # variables
            'error': pt.pack(error), 'recovery': pt.pack(recovery),
        }
        logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
    # check if recovery communicate with logical operations
    commutes_with_logicals = np.all(pt.bsp(recovery^error, code.logicals.T) == 0)
    commutes_with_logicalx = np.all(pt.bsp(recovery^error, code.logical_xs.T) == 0)
    commutes_with_logicalz = np.all(pt.bsp(recovery^error, code.logical_zs.T) == 0)
    # success if recovery communicate with both stabilizers and logical operations
    success = commutes_with_stabilizers and commutes_with_logicals
    # record the logical x and z failures seperately
    failure_x = commutes_with_logicalz and not commutes_with_logicalx
    failure_z = commutes_with_logicalx and not commutes_with_logicalz
    # count the number of each Pauli operators for failure case
    if failure_x:
        # create a counter for I, X, Y, Z
        counter = [0,0,0,0]
        failure_type = 'Logical X'
        for p in error_pauli:
            if p=='I':
                counter[0] += 1
            elif p=='X':
                counter[1] += 1
            elif p=='Y':
                counter[2] += 1
            elif p=='Z':
                counter[3] += 1
        data.append([failure_type,counter])
    elif failure_z:
        # create a counter for I, X, Y, Z
        counter = [0,0,0,0]
        failure_type = 'Logical Z'
        for p in error_pauli:
            if p=='I':
                counter[0] += 1
            elif p=='X':
                counter[1] += 1
            elif p=='Y':
                counter[2] += 1
            elif p=='Z':
                counter[3] += 1
        data.append([failure_type,counter])
            
# store data into file
filename = os.path.join(save_path,'error_chain{}.txt'.format(task_id))
with open(filename,"w", encoding='utf-8') as f:
    f.write('failure type, #I, #X, #Y, #Z\n')
    for l in data:
        f.write('{}, {}, {}, {}, {}'.format(l[0],l[1][0],l[1][1],l[1][2],l[1][3]))
        f.write("\n")

end_time = time.time()
print("Code runs for {:.2f} minutes.".format((end_time-start_time)/60))