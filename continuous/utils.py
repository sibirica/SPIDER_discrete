import numpy as np
from library import *
from ipynb.fs.full.identify_models import Equation

def save(filename, *args):
    with open(filename, 'wb') as f:
        for arr in args:
            np.save(f, arr)
            
def load(filename, nload):
    to_load = []
    with open(filename, 'rb') as f:
        for i in range(nload):
            to_load.append(np.load(f))
    return tuple(to_load)

# Notes: this requires observables to already be constructed;
# no power abbreviations
def construct_from_string(input_str, type_str, obs_dict):
    # obs_dict: name -> Observable
    #if type_str == "Observable": 
        # makes more sense to construct from the usual constructor
    if type_str == "LibraryPrimitive" or type_str == "LP":
        token_list = input_str.split()
        torder = token_list.count("dt")
        xorder = token_list.count("dx")
        obs = obs_dict[token_list[-1]]
        return LibraryPrimitive(DerivativeOrder(torder, xorder), obs)
    #elif type_str == "IndexedPrimitive":
        # not implemented until it seems useful
    elif type_str == "LibraryTensor": # most likely not going to be used
        if input_str == 1:
            return ConstantTerm()
        token_list = input_str.split(" * ")
        product = construct_from_string(token_list[0], "LibraryPrimitive", obs_dict)
        for token in token_list[1:]:
            product *= construct_from_string(token, "LibraryPrimitive", obs_dict)
    elif type_str == "LibraryTerm" or type_str == "LT":
        if input_str == 1:
            return ConstantTerm()
        token_list = input_str.split(" * ")
        obs_list = []
        index_list = []
        for token in token_list: 
            obs, inds1, inds2 = term_plus_inds(token, obs_dict)
            obs_list.append(obs)
            index_list = index_list + [inds1, inds2]
        lt = LibraryTensor(obs_list)
        return LibraryTerm(lt, index_list=index_list)
    #elif type_str == "IndexedTerm":
        # not implemented until it seems useful
    elif type_str == "Equation" or type_str == "EQ":
        term_list = []
        coeffs = []
        token_list = input_str.split(" + ")
        print(token_list)
        for token in token_list:
            if token[0] == '-':
                coeff = -1
                term = token[1:]
            elif token[0].isdigit():
                coeff, term = token.split(" * ", 1)
            else:
                term = token
                coeff = 1
            coeffs.append(float(coeff))
            term_list.append(construct_from_string(term, "LibraryTerm", obs_dict).canonicalize())
        return Equation(term_list, coeffs)
    else:
        raise ValueError(type_str + " is not a valid option.")
        
def term_plus_inds(string, obs_dict):
    inds1 = []
    inds2 = []
    token_list = string.split()
    obs_token = token_list[-1]
    obs_tk_list = obs_token.split("_")
    if len(obs_tk_list)==1:
        obs_nm = obs_dict[obs_token]
    else:
        obs_nm = obs_dict[obs_tk_list[0]]
        for char in obs_tk_list[1]:
            inds2.append(let_to_num_dict[char])
    torder = 0
    xorder = 0
    for token in token_list[:-1]:
        if token == "dt":
            torder += 1
        else:
            xorder += 1
            inds1.append(let_to_num_dict[token[-1]])
    obs = LibraryPrimitive(DerivativeOrder(torder, xorder), obs_nm)
    return obs, inds1, inds2
