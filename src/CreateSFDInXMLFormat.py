import os
import constants as CONST
from docx_utils.flatten import opc_to_flat_opc

def ExtractSFDEnglishFromDirectory():
    path = CONST.DATA+r'caagis translations\SFD English'
    for root, dirs, files in os.walk(path):
        SFDEnglish = files
    print(SFDEnglish)
    return SFDEnglish,path

    
def ExtractSFDFrenchFromDirectory():
    path = CONST.DATA+r'caagis translations\SFD English'
    for root, dirs, files in os.walk(path):
        SFDFrench = files
    print(SFDFrench)
    return SFDFrench, path
    
def ConvertSFDEnglishToXMLFormat(SFDEnglish,path):
    for i in range(len(SFDEnglish)):
        opc_to_flat_opc((os.path.join(path,SFDEnglish[i])) ,(os.path.join(path,SFDEnglish[i].split('.')[0]+'.xml')))
        
def ConvertSFDFrenchToXMLFormat(SFDFrench,path):
    for i in range(len(SFDFrench)):
        opc_to_flat_opc((os.path.join(path,SFDFrench[i])) ,(os.path.join(path,SFDFrench[i].split('.')[0]+'.xml')))

def CreateEngishSFDInXMLFormat():
    SFDEnglish,path = ExtractSFDEnglishFromDirectory()
    ConvertSFDEnglishToXMLFormat(SFDEnglish,path)

def CreateFrenchSFDInXMLFormat():
    SFDFrench,path = ExtractSFDFrenchFromDirectory()
    ConvertSFDFrenchToXMLFormat(SFDFrench,path)
    
if __name__== '__main__':
    CreateEngishSFDInXMLFormat()
    CreateFrenchSFDInXMLFormat()
    
