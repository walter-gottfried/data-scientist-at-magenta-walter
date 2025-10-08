import os
import pickle

def load_artifact(targ_file:str):
    dir = "../tmp8f2c47gv/storage/"
    targ_path = os.path.join(dir,targ_file)

    with open(targ_path,'rb') as fp:
        test_artifact = pickle.load(fp)

    return test_artifact