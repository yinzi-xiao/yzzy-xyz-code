#----------------- import packages ---------------------------
import json5
import os
import multiprocessing
import json
import function
import collections
import itertools
import statistics
import logging
import time
import argparse
import numpy as np
from qecsim import app
from qecsim import paulitools as pt
from qecsim.models.generic import BitPhaseFlipErrorModel, DepolarizingErrorModel, BitFlipErrorModel, PhaseFlipErrorModel
from rotatedplanaryz import *
# from qsdxzzx.rotatedplanarxz import RotatedPlanarXZPauli, RotatedPlanarXZCode, RotatedPlanarXZRMPSDecoder

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
            # add simulation data into data list
            data.append(runs_data)
    
    function.save_simdata(os.path.join(save_path,'xyz_purez{}.json'.format(task_id)),data)

end_time = time.time()
print("Code runs for {:.2f} seconds.".format(end_time-start_time))