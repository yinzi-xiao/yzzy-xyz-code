import json5

def save_simdata(filename,data):
    """
    save the simulation result(s) of qecsim to local pc
    filename: the name or relative path of saved file (format:"xxx.json")
    data: the data needed saving. it is a list of dictionaries
    return: none
    """
    with open(filename,"w", encoding='utf-8') as f:
        for dict in data:
            f.write(json5.dumps(dict))
            f.write("\n")

def load_simdata(filename):
    """
    load the file containing simulation result(s) of qecsim
    filename: the name or relative path of saved file (format:"xxx.json")
    return: data of simulation result(s). it is a list of dictionaries
    """
    data = []
    with open(filename,"r", encoding='utf-8') as f:
        for line in f:
            data.append(json5.loads(line.rstrip(';\n')))
    return data

def reshape_data(data_name,data_num):
    """
    compose a series of datas into one data. each data file should look like : data_name + data_num + '.json'
    data_name: the name of data file
    data_num: number of the data series
    return: one data file which contains all information of the data series
    """
    # make a list store all data together
    data_list = []
    data_eg = load_simdata(data_name+"0.json")
    for i in range(data_num):
        data_list.append(load_simdata(data_name+'{}'.format(i)+'.json'))
    # collect all data in one list
    data = []
    for i_run in range(len(data_eg)):
        dict = {
            'code': data_eg[i_run]['code'],
            'n_k_d': data_eg[i_run]['n_k_d'],
            'time_steps': 1, # 1 for ideal simulation
            'decoder': data_eg[i_run]['decoder'],
            'error_probability': data_eg[i_run]['error_probability'],
            'measurement_error_probability': 0.0, # 0 for ideal simulation
            'n_run': 0,
            'n_success': 0,
            'n_fail': 0,
            'n_xfail' : 0,
            'n_zfail' : 0,
            'n_logical_commutations': None,
            'custom_totals': None,
            'error_weight_total': 0,
            'error_weight_pvar': 0.0,
            'logical_failure_rate': 0.0,
            'physical_error_rate': 0.0,
            'wall_time': 0.0,
        }
        # if there's error model key in data, add it into dict.
        if 'error_model' in data_eg[i_run].keys():
            dict['error_model'] = data_eg[i_run]['error_model']
        for i in range(data_num):
            dict['n_run'] += (data_list[i][i_run])['n_run']
            dict['n_success'] += (data_list[i][i_run])['n_success']
            dict['n_fail'] += (data_list[i][i_run])['n_fail']
            dict['n_xfail'] += (data_list[i][i_run])['n_xfail']
            dict['n_zfail'] += (data_list[i][i_run])['n_zfail']
            dict['error_weight_total'] += (data_list[i][i_run])['error_weight_total']
            dict['error_weight_pvar'] += (data_list[i][i_run])['error_weight_pvar']
            dict['wall_time'] += (data_list[i][i_run])['wall_time']

        # add rate statistics
        dict['error_weight_pvar'] = dict['error_weight_pvar']/data_num
        time_steps = dict['time_steps']
        n_run = dict['n_run']
        n_fail = dict['n_fail']
        n_xfail = dict['n_xfail']
        n_zfail = dict['n_zfail']
        error_weight_total = dict['error_weight_total']
        code_n_qubits = dict['n_k_d'][0]

        dict['logical_failure_rate'] = n_fail / n_run
        dict['logicalx_failure_rate'] = n_xfail / n_run
        dict['logicalz_failure_rate'] = n_zfail / n_run
        dict['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run

        # add dict to list
        data.append(dict)
    return data

def reshape_data_list(data_name,data_num,list_num):
    """
    compose a series of data lists into one data list. each data file should look like : data_name + data_num + '.json'
    data_name: the name of data file
    data_num: number of the data list series
    list_num: number of data in one data list
    return: one data list file which contains all information of the data list series
    """
    # make a list store all data together
    data_list = []
    data_eg = load_simdata(data_name+"0.json")
    for i in range(data_num):
        data_list.append(load_simdata(data_name+"{}".format(i)+".json"))
    # collect all data in one list
    data = []
    for i in range(list_num):
        data.append([])
    for i_model in range(list_num):
        for i_run in range(len(data_eg[i_model])):
            dict = {
                'code': (data_eg[i_model][i_run])['code'],
                'n_k_d': (data_eg[i_model][i_run])['n_k_d'],
                'time_steps': 1, # 1 for ideal simulation
                'decoder': (data_eg[i_model][i_run])['decoder'],
                'error_probability': (data_eg[i_model][i_run])['error_probability'],
                'measurement_error_probability': 0.0, # 0 for ideal simulation
                'n_run': 0,
                'n_success': 0,
                'n_fail': 0,
                'n_logical_commutations': None,
                'custom_totals': None,
                'error_weight_total': 0,
                'error_weight_pvar': 0.0,
                'logical_failure_rate': 0.0,
                'physical_error_rate': 0.0,
                'wall_time': 0.0,
            }
            # if there's error model key in data, add it into dict.
            if 'error_model' in data_eg[i_model][i_run].keys():
                dict['error_model'] = data_eg[i_model][i_run]['error_model']
            for i in range(data_num):
                dict['n_run'] += (data_list[i][i_model][i_run])['n_run']
                dict['n_success'] += (data_list[i][i_model][i_run])['n_success']
                dict['n_fail'] += (data_list[i][i_model][i_run])['n_fail']
                dict['error_weight_total'] += (data_list[i][i_model][i_run])['error_weight_total']
                dict['error_weight_pvar'] += (data_list[i][i_model][i_run])['error_weight_pvar']
                dict['wall_time'] += (data_list[i][i_model][i_run])['wall_time']
                
            # add rate statistics
            dict['error_weight_pvar'] = dict['error_weight_pvar']/data_num
            time_steps = dict['time_steps']
            n_run = dict['n_run']
            n_fail = dict['n_fail']
            error_weight_total = dict['error_weight_total']
            code_n_qubits = dict['n_k_d'][0]

            dict['logical_failure_rate'] = n_fail / n_run
            dict['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run
            
            # add dict to list
            data[i_model].append(dict)
    return data