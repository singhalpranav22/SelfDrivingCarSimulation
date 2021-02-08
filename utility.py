import pandas as pd
import numpy as np
import os

def getEnd(name):
    return name.split('/')[-1]  #return last index of the file name after splitting
def importData(path):
    colums = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names=colums)
    data['Center']=data['Center'].apply(getEnd)
    print(data['Center'][0])
    print('Total Images imported:',data.shape[0])
    return data
