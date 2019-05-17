import csv

def csvFile(data, subject):
    
    nSubjects = 6
    nSessions = 5
    
    matWidth = nSubjects*nSessions
    csvData = [] 
    
    for i in matWidth: 
        
        aux = mfd(subject)
        MFDs = aux[0]
        
        MFD_true = MFDs[0]
        MFD_SD_true = MFDs[3]
        MFD_false = MFDs[1]
        MFD_SD_false = MFDs[4]    
    
        
    #    aux = msa(subject)
    #    MSAs = aux[0]
    #
    #    MSA_true = MSAs[0]
    #    MSA_SD_true = MSAs[3]
    #    MSA_false = MSAs[1]
    #    MSA_SD_false = MSAs[4]    
        
        MFD_overall = MFDs[2]
        MFD_overall_SD = MFDs[5]
        
    #    MSA_overall = MSAs[2]
    #    MSA_overall_SD = MSAs[5]
    
        values = [MFD_true, MFD_SD_true, MFD_false, MFD_SD_false, MSA_true, MSA_SD_true, MSA_false, MSA_SD_false, MFD_overall, MFD_overall_SD, MSA_overall, MSA_overall_SD ]
        csvData.extend(values)
    
    
    with open('output.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)