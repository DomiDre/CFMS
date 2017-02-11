import numpy as np
class CFMSData():
    '''
    Container class for storing and retrieving CFMS data.
    '''
    def __init__(self):
        self.data_path = ''

        #Datafiles: first line is header. 
        #For each entry in header is a list of numbers in data
        self.header = []
        self.data = {}
        self.N_pts = 0
        self.data_loaded = False

        self.averaged_data = {}
        self.N_pts_averaged = 0
        self.N_averages = 1 #averaged over every n-th point
        self.B_string = 'B_analog_(T)'
        self.T_string = 'sensor_B_(K)'
        self.M_string = 'moment_(emu)'
        self.data_averaged = False

        self.valid_point = None # Array which points are valid and which not (filled from clean_peaks)


    def set_data(self, header, data):
        self.header = header
        self.data = data
        self.N_pts = len(data[header[0]])
        self.valid_point = np.ones(self.N_pts, dtype=bool)
        self.data_loaded = True

    def set_averaged_data(self, N_averages, Bavg, Tavg, Mavg):
        self.N_averages = N_averages
        self.averaged_data[self.B_string] = Bavg
        self.averaged_data[self.T_string] = Tavg
        self.averaged_data[self.M_string] = Mavg
        self.N_pts_averaged = len(self.averaged_data[self.B_string][0])
        self.data_averaged = True

    def check_data_loaded(self, data_string):
        if not data_string in self.data:
            sys.exit("ERROR: Unable to find " + data_string + " in data.")
    
    def check_average_data_loaded(self, data_string):
        if not data_string in self.averaged_data:
            sys.exit("ERROR: Unable to find " + data_string + " in averaged data.")

    def get_B(self, get_all_points=False):
        self.check_data_loaded(self.B_string)
        if get_all_points:
            return self.data[self.B_string]
        else:
            return self.data[self.B_string][self.valid_point]

    def get_M(self, get_all_points=False):
        self.check_data_loaded(self.M_string)
        if get_all_points:
            return self.data[self.M_string]
        else:
            return self.data[self.M_string][self.valid_point]

    def get_T(self, get_all_points=False):
        self.check_data_loaded(self.T_string)
        if get_all_points:
            return self.data[self.T_string]
        else:
            return self.data[self.T_string][self.valid_point]

    def get_Bavg(self):
        self.check_average_data_loaded(self.B_string)
        return self.averaged_data[self.B_string]

    def get_Mavg(self):
        self.check_average_data_loaded(self.M_string)
        return self.averaged_data[self.M_string]

    def get_Tavg(self):
        self.check_average_data_loaded(self.T_string)
        return self.averaged_data[self.T_string]

