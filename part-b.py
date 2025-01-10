import numpy as np
import matplotlib.pyplot as plt

def read_WD_csv(filename):
    with open(filename, 'r') as f:  #Open safely using with statement
        lines = f.readlines()[1:]   #Read all lines except first into array
    data = []
    for line in lines:              #Iterate
        _splitted = line.strip().split(',') #Split lines from delimiter ,
        data.append([_splitted[0],_splitted[1],_splitted[2]])   #Append name of the white dwarf, logg and M to list
    data = np.array(data)           #Convert list of lists to nparray
    return data[:,0], data[:,1].astype(np.float64), data[:,2].astype(np.float64) #First column is name, second is logg, and third is M.