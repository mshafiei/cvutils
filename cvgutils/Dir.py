import os

def getPathFilename(fn):
    bn = os.path.basename(fn)
    return fn[:-len(bn)]


def createIfNExist(dr):
    if(not os.path.exists(dr)):
        os.makedirs(dr)
