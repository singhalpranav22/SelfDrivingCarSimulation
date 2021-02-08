from utility import *
from sklearn.model_selection import train_test_split
path='data'
data = importData(path)

data=balanceData(data,display=False)
imagePath,steerings=loadData(path,data)
# print(imagePath[0],steerings[0])

xTrain,xVal,yTrain,yVal = train_test_split(imagePath,steerings,train_size=0.8,random_state=5)
# print('No of trains:' , len(xTrain))
# print('No of Validations:' , len(xVal))
