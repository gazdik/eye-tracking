import csv


class CSVGenerator:
    def __init__(self, file_name):
        self.fd = open(file_name, 'w')
        self.writer = csv.writer(self.fd, delimiter=' ')

    def append_row(self, sid, MFDs_t, MSAs_t):
        MFDs = MFDs_t[0]

        MFD_true = MFDs[0]
        MFD_SD_true = MFDs[3]
        MFD_false = MFDs[1]
        MFD_SD_false = MFDs[4]
    
        MSAs = MSAs_t[0]

        MSA_true = MSAs[0]
        MSA_SD_true = MSAs[3]
        MSA_false = MSAs[1]
        MSA_SD_false = MSAs[4]

        MFD_overall = MFDs[2]
        MFD_overall_SD = MFDs[5]
        
        MSA_overall = MSAs[2]
        MSA_overall_SD = MSAs[5]

        values = [sid, MFD_true, MFD_SD_true, MFD_false, MFD_SD_false, MSA_true, MSA_SD_true,
                  MSA_false, MSA_SD_false, MFD_overall, MFD_overall_SD, MSA_overall, MSA_overall_SD]

        self.writer.writerow(values)

    def close(self):
        self.fd.close()

