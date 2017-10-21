import matplotlib.pyplot as plt    

class CFMSPlotter():
    def __init__(self):
        super(CFMSPlotter, self).__init__()
    
    def plot_B_M(self, B, M, label=None, marker='.', Bunit='T', Munit='memu'):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(B, M, label=label, marker=marker)
        self.ax.set_xlabel("$ \mathit{\mu_0 H} \, / \, "+Bunit+"$")
        self.ax.set_ylabel("$ \mathit{M} \, / \, "+Munit+"$")
        
    def plot_T_M(self, T, M, label=None, marker='.', Tunit='K', Munit='memu'):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(T, M, label=label, marker=marker)
        self.ax.set_xlabel("$ \mathit{T} \, / \, "+Tunit+"$")
        self.ax.set_ylabel("$ \mathit{M} \, / \, "+Munit+"$")
        
    def set_xlim(self, xmin=None, xmax=None):
        self.ax.set_xlim(xmin, xmax)

    def set_ylim(self, ymin=None, ymax=None):
        self.ax.set_ylim(ymin, ymax)
        
    def legend(self, loc='best'):
        self.fig.legend(loc='best')

    def tight_layout(self):
        self.fig.tight_layout()
    
    def savefig(self, savename):
        self.fig.savefig(savename)
        print("Saved plot to " + savename)

    def show(self):
        plt.show()