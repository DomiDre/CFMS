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


    def set_data(self, header, data):
        self.header = header
        self.data = data
        self.N_pts = len(data[header[0]])
        self.data_loaded = True

    def set_averaged_data(self, N_averages, Bavg, Tavg, Mavg):
        self.N_averages = N_averages
        self.averaged_data[self.B_string] = Bavg
        self.averaged_data[self.T_string] = Tavg
        self.averaged_data[self.M_string] = Mavg
        self.N_pts_averaged = len(self.averaged_data[self.B_string][0])
        self.data_averaged = True

    def check_data_string(self, data_string):
        if not data_string in self.data:
            sys.exit("ERROR: Unable to find " + data_string + " in data.")

    def get_B(self):
        self.check_data_string(self.B_string)
        return self.data[self.B_string]

    def get_M(self):
        self.check_data_string(self.M_string)
        return self.data[self.M_string]

    def get_T(self):
        self.check_data_string(self.T_string)
        return self.data[self.T_string]

    def get_Bavg(self):
        return self.averaged_data[self.B_string]

    def get_Mavg(self):
        return self.averaged_data[self.M_string]

    def get_Tavg(self):
        return self.averaged_data[self.T_string]

