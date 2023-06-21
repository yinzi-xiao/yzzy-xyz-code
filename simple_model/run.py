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
from qecsim.models.generic import BitFlipErrorModel, BitPhaseFlipErrorModel, PhaseFlipErrorModel, DepolarizingErrorModel, BiasedDepolarizingErrorModel

start_time = time.time()

#----------------- set parameters -----------------------------
# set physical error probabilities
error_probability_min, error_probability_max = 0, 0.5
error_probabilities = np.linspace(error_probability_min, error_probability_max, 100)
# set run times for each simulation
max_runs = 1000
# save paths
save_path = os.getenv('SLURM_SUBMIT_DIR')
# get the task ID from jobscript for data storage
parser = argparse.ArgumentParser(description='task id')
parser.add_argument('--taskid', type=str, default = None)
args = parser.parse_args()
task_id = args.taskid

#----------------- test of different # qubits having error in non-IID error model -------------
if __name__ == '__main__':
    # initialize model
    codes = [RotatedPlanarYZCode(d) for d in {3,5,7,9,11}]
    decoder = RotatedPlanarYZRNIIDMPSDecoder(chi=12)
    error_model = PhaseFlipErrorModel()
    # run simulations and save data
    logger = logging.getLogger(__name__)
    # initialize simulation data
    data = []
    # run for each error model
    for code in codes:
        # run for each error probability
        for error_probability in error_probabilities:
            runs_data = run_xyz_multicore(code,decoder,error_model,error_probability,max_runs)
            # prob_dist_list = []
            # prob_dist = error_model.probability_distribution(error_probability)
            # for i in range(code.n_k_d[0]):
            #     prob_dist_list.append(prob_dist)
            # runs_data = run_niid_multicore(code,decoder,prob_dist_list,error_probability,max_runs)
            # add simulation data into data list
            data.append(runs_data)
    
    function.save_simdata(os.path.join(save_path,'xyz_purez{}.json'.format(task_id)),data)

end_time = time.time()
print("Code runs for {:.2f} minutes.".format((end_time-start_time)/60))