from ReadXMLTry3 import extractTextFromXMLFile
from WriteCSVFile import writeListToCSVFile
from CreateSFDInXMLFormat import CreateEngishSFDInXMLFormat,CreateFrenchSFDInXMLFormat
import os

def ExtractTextFromXMLAndWriteInCSV(SFDEnglishXMLFiles,SFDFrenchXMLFiles):
    for i in range(len(SFDEnglishXMLFiles)):
#Extracting text of English SFD from XML Tags
        SFDListEnglishText = extractTextFromXMLFile(SFDEnglishXMLFiles[i])
        SFDListEnglishText = [elem for elem in SFDListEnglishText if elem.strip()]

#Extracting text of French SFD from XML Tags        
        SFDListFrenchText = extractTextFromXMLFile(SFDFrenchXMLFiles[i])
        SFDListFrenchText = [elem for elem in SFDListFrenchText if elem.strip()]
        
#Writing CSV File
        rows = zip(SFDListEnglishText,SFDListFrenchText)
        newfilePath = 'Dataset.csv'
        print('Writing CSV File')
        writeListToCSVFile(newfilePath,rows)


def ExtractEnglishSFDInXMLFromDir():
    path = r'C:\Users\Sanjot Kaur\Desktop\CAAS\French Translation Project\Source Code for building dataset\SFD English'
    SFDEnglishXMLFiles = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            SFDEnglishXMLFiles.append(os.path.join(path, file))
    print (SFDEnglishXMLFiles)
    return SFDEnglishXMLFiles
    
def ExtractFrenchSFDInXMLFromDir():
    path = r'C:\Users\Sanjot Kaur\Desktop\CAAS\French Translation Project\Source Code for building dataset\SFD French'
    SFDFrenchXMLFiles = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            SFDFrenchXMLFiles.append(os.path.join(path, file))
    print (SFDFrenchXMLFiles)
    return SFDFrenchXMLFiles 

if __name__== '__main__':
    CreateEngishSFDInXMLFormat()
    CreateFrenchSFDInXMLFormat()
    SFDEnglishXMLFiles = ExtractEnglishSFDInXMLFromDir()
    SFDFrenchXMLFiles = ExtractFrenchSFDInXMLFromDir()
    ExtractTextFromXMLAndWriteInCSV (SFDEnglishXMLFiles,SFDFrenchXMLFiles)