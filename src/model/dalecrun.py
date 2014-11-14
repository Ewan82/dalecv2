"""Run this file from the command line to generate a list of evolved model 
values. Read called files for more info on functions. The first 6 elements of
each array at each time step are the 6 carbon pools (clab, cf, cr, cw, cl, cs)
which the dalecv2 model predicts, the next 17 elements are the model parameters
which stay constant at every time step. Uses model.py and data.py (data.py is a
 data class extracting data from the files in the data dir).
"""
import model as mod
import data as dC

d = dC.dalecData(365)
def dalecrun(initconditions = d.pvals, dataClass = d, start = 0, fin = 365):
    print mod.mod_list(initconditions, d, start, fin)
    
if __name__ == "__main__":
    dalecrun() 