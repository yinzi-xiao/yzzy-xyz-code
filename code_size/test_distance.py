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
# set eta for biased error model
ETA = 150
# set run times for each simulation
max_runs = 100
# save paths
save_path = os.getenv('SLURM_SUBMIT_DIR')
# get the task ID from jobscript for data storage
parser = argparse.ArgumentParser(description='task id')
parser.add_argument('--taskid', type=str, default = None)
args = parser.parse_args()
task_id = args.taskid

#----------------- run simulation with different code distances -------------
if __name__ == '__main__':
    # initialize model
    codes = [RotatedPlanarYZCode(d) for d in range(5,85,8)]
    decoder = RotatedPlanarYZRNIIDMPSDecoder(chi=12)
    error_model = BiasedDepolarizingErrorModel(bias=ETA,axis='Z')
    # calculate the error probability p when px=pi
    # error_probability = (ETA+1)/(2*ETA+1)
    # use different error probabilities
    error_probability = 0.45
    # run simulations and save data
    logger = logging.getLogger(__name__)
    # initialize simulation data
    data = []
    # run for each code
    for code in codes:
        prob_dist_list = []
        prob_dist = error_model.probability_distribution(error_probability)
        for i in range(code.n_k_d[0]):
            prob_dist_list.append(prob_dist)
        runs_data = run_niid_multicore(code,decoder,prob_dist_list,error_probability,max_runs)
        # add simulation data into data list
        data.append(runs_data)
    
    function.save_simdata(os.path.join(save_path,'yzzy_z{}bias{}.json'.format(ETA,task_id)),data)

end_time = time.time()
print("Code runs for {:.2f} minutes.".format((end_time-start_time)/60))

