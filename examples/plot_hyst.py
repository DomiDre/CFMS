from CFMS.cfms import CFMS

# Load data
cfms_data = CFMS()
cfms_data.load("example_data.dat")
cfms_data.clean_peaks()

# Plot data
B, M = cfms_data.get_BM()
cfms_data.plot_B_M(B, M)
cfms_data.set_xlim(-7, 7)
cfms_data.set_ylim(-15, 15)
cfms_data.tight_layout()
cfms_data.savefig("example.png")
cfms_data.show()