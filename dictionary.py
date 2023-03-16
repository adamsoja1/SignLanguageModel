import os
import json
import ast
class Dictionary:
    def __init__(self,path):
        if not os.path.exists(path):
            raise Exception(f'Path {path} does not exist!')
        else:
            self.path = path
            

        self.createDictionary()

    def collectData(self):
        return os.listdir(self.path)
    
    def createDictionary(self):
        signs = self.collectData()
        self.dictionary = {}
        for number in range(len(signs)):
            self.dictionary[number] = signs[number]
        return self.dictionary
            
    def translate(self,element):
        return self.dictionary[element]
    def getDictionary(self):
        return self.dictionary


if __name__ == '__main__':
    dct = Dictionary('main_data/asl_alphabet_train')
    f = open("dictionary.txt","w")
    # write file
    f.write(str(dct.getDictionary()))
    # close file
    f.close()



