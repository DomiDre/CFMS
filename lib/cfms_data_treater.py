from .cfms_data import CFMSData
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import warnings
import lmfit

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
                      TM_mode = False,\
                      show=True):
        '''
        Cleans data by forming for every point the median with npts_mean
        points surrounding it (median filter). Data points which are more distant
        than threshold are marked for cleaning.
        '''
        self.print_log("Cleaning data lists from spurious peaks.")
        B, M = self.get_BM(get_all_points=True, supress_log=True)
        T = self.data_container.get_T(get_all_points=True)
        M_mean = spsig.medfilt(M, M_npts_mean)
        M_diff = M-M_mean
        M_peak_outlier = np.abs(M_diff) > M_threshold
        invalid_points = M_peak_outlier
        if not TM_mode:
            B_mean = spsig.medfilt(B, B_npts_mean)
            B_diff = B-B_mean
            B_peak_outlier = np.abs(B_diff) > B_threshold
            invalid_points = np.logical_or(invalid_points, B_peak_outlier)
        B_masked = B[invalid_points]
        M_masked = M[invalid_points]
        T_masked = T[invalid_points]
        B_clean = B[-invalid_points]
        M_clean = M[-invalid_points]
        T_clean = T[-invalid_points]
        if show:
            self.fig, self.ax = plt.subplots()
            self.ax.axhline(0, c='gray')
            if TM_mode:
                self.ax.plot(T, M, label='Raw Data', color='blue', marker='.', ls='None')
                self.ax.plot(T, M_mean, label='Mean Data', marker='None',\
                        ls='-', color='black')
                with warnings.catch_warnings(): #he doesnt like circ and mu in my Ubuntu
                    warnings.filterwarnings("ignore", module="matplotlib")
                    self.ax.plot(T_masked, M_masked, label='Masked Points', color='red',\
                                    ls='None', marker='$^\circ$', markersize=10)
                    self.ax.set_xlabel("$ \mathit{T} \, / \, K$")
            else:
                self.ax.plot(B, M, label='Raw Data', color='blue', marker='.', ls='None')
                self.ax.plot(B_mean, M_mean, label='Mean Data', marker='None',\
                        ls='-', color='black')
                with warnings.catch_warnings(): #he doesnt like circ and mu in my Ubuntu
                    warnings.filterwarnings("ignore", module="matplotlib")
                    self.ax.plot(B_masked, M_masked, label='Masked Points', color='red',\
                                    ls='None', marker='$^\circ$', markersize=10)
                    self.ax.set_xlabel("$ \mathit{\mu_0 H} \, / \, T$")
            self.ax.set_ylabel("$ \mathit{M} \, / \, memu$")
            
            self.ax.legend(loc='best')
            self.fig.tight_layout()         
        self.data_container.set_data_invalid(invalid_points)
        self.print_log('Set array for validated points')


    def remove_virgin_data(self, up_to=6.9):
        B = self.data_container.get_B(get_all_points=True)
        valid_point = np.ones(len(B), dtype=bool)
        if up_to > 0:
            for ib, b_val in enumerate(B):
                if b_val < up_to:
                    valid_point[ib] = False
                else:
                    break
        else:
            for ib, b_val in enumerate(B):
                if b_val > up_to:
                    valid_point[ib] = False
                else:
                    break
        self.data_container.set_data_invalid(-valid_point)

    def fit_diamagnetism(self, B0, B1,\
                            m_init=None, b_init=None,\
                            vary_m=True, vary_b=True,\
                            fit_both_sides=True,\
                            show=True):
        B, M = self.get_BM(supress_log=True)
        fit_range = np.logical_and(B0<B, B<B1)
        B_fit =  B[fit_range]
        M_fit = M[fit_range]

        if m_init is None:
            m_init = (M_fit[-1] - M_fit[0])/(B_fit[-1] - B_fit[0])
        if b_init is None:
            b_init = M_fit[0] - B_fit[0]*m_init 
        linear_model = lambda m, b, B_lin: m*B_lin + b
        
        def fit_between(B_fit, M_fit, m_init, b_init):
            linear_residuum = lambda p: linear_model(p['m'], p['b'], B_fit) - M_fit
            p = lmfit.Parameters()
            p.add("m", m_init, vary=vary_m)
            p.add("b", b_init, vary=vary_b)
            self.print_log("Running linear fit between B=("+\
                            str(min(B_fit)) + ' .. '+ str(max(B_fit))+") with LM algorithm.")
            fit_result = lmfit.minimize(linear_residuum, p)
            self.print_log(lmfit.fit_report(fit_result))
            return fit_result

        fit_result = fit_between(B_fit, M_fit, m_init, b_init)

        if fit_both_sides:
            fit_range2 = np.logical_and(-B1<B, B<-B0)
            B_fit2 =  B[fit_range2]
            M_fit2 = M[fit_range2]
            fit_result2 = fit_between(B_fit2, M_fit2,\
                                      fit_result.params['m'],\
                                      -fit_result.params['b'])
            m1 = fit_result.params['m']
            m2 = fit_result2.params['m']
            b1 = fit_result.params['b']
            b2 = fit_result2.params['b']

            pts1 = fit_result.ndata
            pts2 = fit_result2.ndata
            mean_slope = (m1.value*pts1 + m2.value*pts2) / (pts1 + pts2)
            mean_slope_std = np.sqrt((m1.stderr*pts1)**2 + (m2.stderr*pts2)**2)/\
                             (pts1 + pts2)
            mean_offset = (b1.value*pts1 - b2.value*pts2) / (pts1 + pts2)
            mean_offset_std = np.sqrt((b1.stderr*pts1)**2 + (b2.stderr*pts2)**2)/\
                             (pts1 + pts2)
            
            fit_result.params['m'].value = mean_slope
            fit_result.params['m'].stderr = mean_slope_std
            fit_result.params['b'].value = mean_offset
            fit_result.params['b'].stderr = mean_offset_std
        self.data_container.set_diamagnetic_fit(fit_result)
        if show:
            self.fig, self.ax = plt.subplots()
            self.ax.axhline(0, c='gray')
            
            self.ax.plot(B, M, label='Data', color='gray', marker='.', ls='None')
            self.ax.plot(B_fit, M_fit, label='Fit Region', marker='.',\
                    ls='None', color='red')
            self.ax.plot(B, linear_model(fit_result.params['m'],\
                            fit_result.params['b'], B),\
                    label='Linear Fit', marker='None',\
                    ls='-', color='black')
            if fit_both_sides:
                self.ax.plot(B_fit2, M_fit2, marker='.',\
                    ls='None', color='red')
            
                self.ax.plot(B, linear_model(fit_result.params['m'],\
                            -fit_result.params['b'], B),\
                    marker='None',\
                    ls='-', color='black')
            

            self.ax.set_xlabel("$ \mu_0 \mathit{ H} \, / \, T$")
            self.ax.set_ylabel("$ \mathit{M} \, / \, memu$")
            
            self.ax.legend(loc='best')
            self.fig.tight_layout()         
            

    def get_diamagnetic_slope(self):
        return self.data_container.diamagnetic_fit.params['m']
    
    def get_diamagnetic_offset(self):
        return self.data_container.diamagnetic_fit.params['b']

    def set_diamagnetic_slope(self, slope): 
        p = lmfit.Parameters()
        p.add('m', slope)
        p.add('b', 0.)
        diamagnetic_data = lmfit.minimize(lambda p: np.ones(10)*(p['m']+p['b']), p)
        diamagnetic_data.params = p
        self.data_container.set_diamagnetic_fit(diamagnetic_data)
    
    def get_offset(self):
        return self.data_container.diamagnetic_fit.params['b']

    def get(self, string, get_all_points=False, supress_log=False):
        if not supress_log:
            self.print_log('Loading ' + string)
        data = self.data_container.get(string, get_all_points)
        return data

    def get_BM(self, get_all_points=False, supress_log=False):
        if not supress_log:
            self.print_log('Loading ' + self.data_container.B_string + ' and ' +\
                                    self.data_container.M_string)
        B = self.data_container.get_B(get_all_points)
        M = self.data_container.get_M(get_all_points)
        return B, M

    def get_TM(self, get_all_points=False, supress_log=False):
        if not supress_log:
            self.print_log('Loading ' + self.data_container.T_string + ' and ' +\
                                    self.data_container.M_string)
        T = self.data_container.get_T(get_all_points)
        M = self.data_container.get_M(get_all_points)
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
        self.fig, self.ax = plt.subplots()
        B, M = self.get_BM()
        self.ax.plot(B, M)
        
        self.ax.set_xlabel("$ \mathit{B} \, / \, T$")
        self.ax.set_ylabel("$ \mathit{M} \, / \, memu$")
        self.fig.tight_layout()
        saveplot = self.data_container.data_path.rsplit(".",1)[0] + "_BM.png"
        self.fig.saveself.fig(saveplot)
        print("Saved plot to " + saveplot)

    def plot_B_M_avg(self):
        self.fig, self.ax = plt.subplots()
        B, sB, M, sM = self.get_BMavg()

        valid_points = sM/M < 1e-1
        B = B[valid_points]
        sB = sB[valid_points]
        M = M[valid_points]
        sM = sM[valid_points]
        self.ax.errorbar(B, M, xerr=sB, yerr=sM)
        
        self.ax.set_xlabel("$ \mathit{B} \, / \, T$")
        self.ax.set_ylabel("$ \mathit{M} \, / \, memu$")
        self.fig.tight_layout()
        saveplot = self.data_container.data_path.rsplit(".",1)[0] + "_BMavg.png"
        self.fig.saveself.fig(saveplot)
        print("Saved plot to " + saveplot)
        
    def plot_T_M(self):
        self.fig, self.ax = plt.subplots()
        T, M = self.get_TM()
        self.ax.plot(T, M)
        
        self.ax.set_xlabel("$ \mathit{T} \, / \, K$")
        self.ax.set_ylabel("$ \mathit{M} \, / \, memu$")
        self.fig.tight_layout()
        saveplot = self.data_container.data_path.rsplit(".",1)[0] + "_TM.png"
        self.fig.saveself.fig(saveplot)
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