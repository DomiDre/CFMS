import numpy as np
import sys
class CFMSData():
    def __init__(self):
        #Container class for storing and retrieving CFMS data.
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

        self.diamagnetic_fit = None # lmfit MinimizerResult object
        self.do_diamagnetic_correction = False

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

    def set_data_invalid(self, invalid_points):
        if len(invalid_points) != self.N_pts:
            print('WARNING: Trying to set data (Pts: '+str(self.N_pts)+\
                  ') invalid with array that has length ' +\
                  str(len(invalid_points)))
        self.valid_point = np.logical_and(self.valid_point, ~invalid_points)

    def set_diamagnetic_fit(self, diamagnetic_fit):
        self.diamagnetic_fit = diamagnetic_fit
        self.do_diamagnetic_correction = True

    def check_data_loaded(self, data_string):
        if not data_string in self.data:
            sys.exit("ERROR: Unable to find " + data_string + " in data.")
    
    def check_average_data_loaded(self, data_string):
        if not data_string in self.averaged_data:
            sys.exit("ERROR: Unable to find " + data_string + " in averaged data.")

    def get(self, string, get_all_points=False):
        self.check_data_loaded(string)
        if get_all_points:
            return self.data[string]
        else:
            return self.data[string][self.valid_point]

    def get_B(self, get_all_points=False):
        return self.get(self.B_string)

    def get_M(self, get_all_points=False):
        self.check_data_loaded(self.M_string)
        if get_all_points:
            M_values = self.data[self.M_string]
        else:
            M_values = self.data[self.M_string][self.valid_point]
        if self.do_diamagnetic_correction:
            M_values -= self.diamagnetic_fit.params['m'].value*\
                        self.get_B(get_all_points)
        return M_values

    def get_T(self, get_all_points=False):
        return self.get(self.T_string)

    def get_Bavg(self):
        self.check_average_data_loaded(self.B_string)
        return self.averaged_data[self.B_string]

    def get_Mavg(self):
        self.check_average_data_loaded(self.M_string)
        return self.averaged_data[self.M_string]

    def get_Tavg(self):
        self.check_average_data_loaded(self.T_string)
        return self.averaged_data[self.T_string]

    def get_valid_points(self):
        return self.valid_point
