#!/usr/bin/env python
# coding: utf-8

from scipy.io import loadmat


def load_data(file_name, sid_list):
    """
    Load the data
    :param file_name: the name of the MATLAB file
    :param sid_list: list of group IDs
    :return: dictionary with parsed and filtered data
    """
    data_mat = loadmat(file_name)
    sids = data_mat['sid'].tolist()[0]
    knowns = data_mat['known'].transpose()
    coords = data_mat['coordinates'].transpose()
    
    res = {}
    for sid, known, coord in zip(sids, knowns, coords):
        vals = [known[0], coord[0]]
        if sid[0] in sid_list:
            res[sid[0]] = vals
            
    return res




data = load_data('train.mat', ['s4', 's14', 's24', 's34', 's10', 's20'])
