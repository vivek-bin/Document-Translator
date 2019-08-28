import xml.etree.ElementTree as ET 

def extractTextFromXMLFile(file):   
    ET.register_namespace('w',"http://schemas.openxmlformats.org/wordprocessingml/2006/main")
    ET.register_namespace('pkg',"http://schemas.microsoft.com/office/2006/xmlPackage")
    
#    tree = ET.parse(r'C:\Users\Sanjot Kaur\Desktop\CAAS\French Translation Project\SFD_VJ2RESAF - TRANSLATED - NEW_CORRECTED.xml')
    tree = ET.parse(file)
    root = tree.getroot() 
    
    ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    SFDText = []
    j = 0
       
    paragraphs = root.findall('.//w:p',ns)  
    numberOfParagrapghs = len(paragraphs)
    for i in range(numberOfParagrapghs):
        rows = paragraphs[i].findall('w:r',ns) 
        textInParagraph = ""        
        for row in rows:
            textInRow = row.find('w:t',ns)         
            if (textInRow != None):
                textInParagraph = textInParagraph + textInRow.text
                j = j+1
        print(textInParagraph)
        SFDText.append(textInParagraph)
    print(j)
    print(len(SFDText))
        
    print(SFDText)
    return SFDText
            
   
    
#extractTextFromXMLFile()    
