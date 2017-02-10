from .cfms_data import CFMSData
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import warnings
class CFMSDataTreater():
    def __init__(self):
        self.data_container = CFMSData()
        self.log = ''

    def add_log(self, entry):
        self.log += '#' + entry + '\n'

    def print_log(self, entry):
        self.add_log(entry)
        print(entry)

    def load(self, data_path):
        self.data_container.data_path = data_path

        datafile = open(data_path, "r")
        self.print_log("Loading data from " + str(data_path))
        header = datafile.readline().rstrip().split()
        
        data = {}
        self.print_log("")
        self.print_log("Found columns:")
        for datahead in header:
            self.print_log(str(datahead))
            data[datahead] = []
            
        for line in datafile:
            linestrip = line.strip()
            if linestrip.startswith("#") or linestrip == "":
                continue
            split_line = linestrip.split()
            for i, val in enumerate(split_line):
                data[header[i]].append(float(val))
        datafile.close()
        
        self.print_log("")
        
        for datahead in data:
            data[datahead] = np.asarray(data[datahead])

        if "moment_(emu)" in data:
            data["moment_(emu)"] *= 1e3
            self.print_log("Transformed data from column 'moment_(emu)' "+\
                               "from emu to memu by multiplication with 1000.")

        self.data_container.set_data(header, data)
        self.print_log("Loaded "+ str(self.data_container.N_pts) +\
                       " datapoints.")


    def reduce_averages(self, N_averages):
        
        self.print_log("Averaging data, combining every " + str(N_averages) +\
                           " points.")
        B_avg = []
        sB_avg = []

        M_avg = []
        sM_avg = []

        T_avg = []
        sT_avg = []
        
        B = self.data_container.get_B()
        M = self.data_container.get_M()
        T = self.data_container.get_T()
        n_pts = len(B)
        for i in range(0, n_pts, N_averages):
            cur_b_avg = B[i:i+N_averages]
            cur_m_avg = M[i:i+N_averages]
            cur_t_avg = T[i:i+N_averages]
            if len(cur_b_avg) != N_averages or \
               len(cur_m_avg) != N_averages or \
               len(cur_t_avg) != N_averages:
                    self.print_log("Skipped line "+ str(i)+\
                            ": Does not commensurate.")
                    continue
            B_avg.append(np.mean(cur_b_avg))
            sB_avg.append(np.std(cur_b_avg, ddof=1))
            M_avg.append(np.mean(cur_m_avg))
            sM_avg.append(np.std(cur_m_avg, ddof=1))
            T_avg.append(np.mean(cur_t_avg))
            sT_avg.append(np.std(cur_t_avg, ddof=1))
            
        
        B_avg = np.asarray(B_avg)
        sB_avg = np.asarray(sB_avg)

        M_avg = np.asarray(M_avg)
        sM_avg = np.asarray(sM_avg)

        T_avg = np.asarray(T_avg)
        sT_avg = np.asarray(sT_avg)

        self.data_container.set_averaged_data(N_averages,\
                                              (B_avg, sB_avg),\
                                              (T_avg, sT_avg),\
                                              (M_avg, sM_avg))
        self.print_log("Average data contains " +\
                           str(self.data_container.N_pts_averaged) +\
                           " data points.")

    def clean_peaks(self,\
                      M_threshold = 20e-3,\
                      M_npts_mean = 11,\
                      B_threshold = 0.1,\
                      B_npts_mean = 5,\
                      show=True):
        '''
        Cleans data by forming for every point the median with npts_mean
        points surrounding it (median filter). Data points which are more distant
        than threshold are marked for cleaning.
        '''
        self.print_log("Cleaning data lists from spurious peaks.")
        B, M = self.get_BM(supress_log=True)
        M_mean = spsig.medfilt(M, M_npts_mean)
        B_mean = spsig.medfilt(B, B_npts_mean)
        M_diff = M-M_mean
        B_diff = B-B_mean
        M_peak_outlier = np.abs(M_diff) > M_threshold
        B_peak_outlier = np.abs(B_diff) > B_threshold
        invalid_points = np.logical_or(M_peak_outlier, B_peak_outlier)
        B_masked = B[invalid_points]
        M_masked = M[invalid_points]
        B_clean = B[-invalid_points]
        M_clean = M[-invalid_points]
        if show:
            fig, ax = plt.subplots()
            ax.axhline(0, c='gray')
            
            ax.plot(B, M, label='Raw Data', color='blue', marker='.', ls='None')
            ax.plot(B_mean, M_mean, label='Mean Data', marker='None',\
                    ls='-', color='black')

            with warnings.catch_warnings(): #he doesnt like circ and mu in my Ubuntu
                warnings.filterwarnings("ignore", module="matplotlib")
                ax.plot(B_masked, M_masked, label='Masked Points', color='red',\
                                ls='None', marker='$^\circ$', markersize=10)
                ax.set_xlabel("$ \mathit{\mu_0 H} \, / \, T$")
            ax.set_ylabel("$ \mathit{M} \, / \, memu$")
            
            ax.legend(loc='best')
            fig.tight_layout()         
            plt.show()
        self.data_container.valid_point = -invalid_points
        self.print_log('Set array for validated points')

    def get_BM(self, supress_log=False):
        if not supress_log:
            self.print_log('Loading ' + self.data_container.B_string + ' and ' +\
                                    self.data_container.M_string)
        B = self.data_container.get_B()
        M = self.data_container.get_M()
        return B, M

    def get_TM(self, supress_log=False):
        if not supress_log:
            self.print_log('Loading ' + self.data_container.T_string + ' and ' +\
                                    self.data_container.M_string)
        T = self.data_container.get_T()
        M = self.data_container.get_M()
        return T, M
        
    def get_BMavg(self):
        B, sB = self.data_container.get_Bavg()
        M, sM = self.data_container.get_Mavg()
        return B, sB, M, sM

    def get_TMavg(self):
        T, sT = self.data_container.get_Tavg()
        M, sM = self.data_container.get_Mavg()
        return T, sT, M, sM

    def plot_B_M(self):
        fig, ax = plt.subplots()
        B, M = self.get_BM()
        ax.plot(B, M)
        
        ax.set_xlabel("$ \mathit{B} \, / \, T$")
        ax.set_ylabel("$ \mathit{M} \, / \, memu$")
        fig.tight_layout()
        saveplot = self.data_container.data_path.rsplit(".",1)[0] + "_BM.png"
        fig.savefig(saveplot)
        print("Saved plot to " + saveplot)

    def plot_B_M_avg(self):
        fig, ax = plt.subplots()
        B, sB, M, sM = self.get_BMavg()

        valid_points = sM/M < 1e-1
        B = B[valid_points]
        sB = sB[valid_points]
        M = M[valid_points]
        sM = sM[valid_points]
        ax.errorbar(B, M, xerr=sB, yerr=sM)
        
        ax.set_xlabel("$ \mathit{B} \, / \, T$")
        ax.set_ylabel("$ \mathit{M} \, / \, memu$")
        fig.tight_layout()
        saveplot = self.data_container.data_path.rsplit(".",1)[0] + "_BMavg.png"
        fig.savefig(saveplot)
        print("Saved plot to " + saveplot)
        
    def plot_T_M(self):
        fig, ax = plt.subplots()
        T, M = self.get_TM()
        ax.plot(T, M)
        
        ax.set_xlabel("$ \mathit{T} \, / \, K$")
        ax.set_ylabel("$ \mathit{M} \, / \, memu$")
        fig.tight_layout()
        saveplot = self.data_container.data_path.rsplit(".",1)[0] + "_TM.png"
        fig.savefig(saveplot)
        print("Saved plot to " + saveplot)
    
    def export_avg(self, export_file=None):
        if export_file is None:
            export_file = self.data_container.data_path.rsplit(".",1)[0]+"_extracted.xye"
        
        B, sB = self.get_Bavg()
        M, sM = self.get_Mavg()
        T, sT = self.get_Tavg()

        self.print_log("Export averaged data to " + export_file)
        savefile = open(export_file, "w")
        
        savefile.write(self.log)
        savefile.write("#\n#B / T\tsB / T\tM / memu\tsM / memu\tT / K\tsT / K\n")
        for i in range(self.data_container.N_pts_avg):
            savefile.write(str(B[i]) + "\t" +\
                       str(sB[i]) + "\t" +\
                       str(M[i]) + "\t" +\
                       str(sM[i]) + "\t" +\
                       str(T[i]) + "\t" +\
                       str(sT[i]) + "\n")
        savefile.close()
    
    def show(self): 
        plt.show()