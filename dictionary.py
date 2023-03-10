import os

class Dictionary:
    def __init__(self,path):
        if not os.path.exists(path):
            raise Exception(f'Path {path} does not exist!')
        else:
            self.path = path
        
    def collectData(self):
        return os.listdir(self.path)
    
    def createDictionary(self):
        signs = self.collectData()
        dictionary = {}
        for number in range(len(signs)):
            dictionary[number] = signs[number]
        return dictionary
            
    
