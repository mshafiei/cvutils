import os

def createIfNExist(dr):
    if(not os.path.exists(dr)):
        os.makedirs(dr)