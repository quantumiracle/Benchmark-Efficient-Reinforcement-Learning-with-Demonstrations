# https://matplotlib.org/users/customizing.html
# https://github.com/jbmouret/matplotlib_for_papers

%matplotlib inline
import matplotlib.pyplot as plt
# print(plt.style.available)  # show available template
plt.style.use(['seaborn-ticks','seaborn-paper'])  # use a templet

import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = 2
# mpl.rcParams['lines.color'] = 'r'
params = {
    'figure.figsize': [8, 6], # Note! figure unit is inch!  scale fontz size 2.54 to looks like unit cm
    'axes.labelsize': 7*2.54, # scale 2.54 to change to figure unit looks as cm
    'text.fontsize':  7*2.54,
    'legend.fontsize': 7*2.54,
    'xtick.labelsize': 6*2.54,
    'ytick.labelsize': 6*2.54,
    'text.usetex': False,  
    'xtick.direction': "in",
    'ytick.direction': "in", # ticket inside
    'legend.frameon' : True, 
    'legend.edgecolor': 'black',
    'legend.shadow': True,
    'legend.framealpha':1,
#     'patch.linewidth' : 0.5, 
}
mpl.rcParams.update(params)

# other package import
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar

import numpy as np

noise_amplitude = np.linspace(0.2,2.0,10)
x = noise_amplitude

decay_rate  = [ 82.550,  199.789,  391.643,  657.524, 1018.523,
                  1521.611, 1945.286, 2532.927, 3188.046, 3864.099]
decay_rate  = np.array(decay_rate)

from scipy.optimize import curve_fit

def line_fit(x,a,b):
    return a *x +b


y = np.sqrt(decay_rate)
popt,perr =  curve_fit(line_fit,x,y,[30.0,0.0])
print(popt)
# print(np.linalg(perr))

x_fit = np.linspace(0.1,2.1,100)
y_fit = line_fit(x_fit,*popt)

plt.plot(x, y, 'o', label='data',color ="xkcd:windows blue",markersize=15)
plt.plot(x_fit, y_fit,label = 'linear fit',color ="xkcd:amber",linewidth=3)

# set legend
leg = plt.legend(loc=4)
legfm = leg.get_frame()
legfm.set_edgecolor('black') # set legend fame color
legfm.set_linewidth(0.5)   # set legend fame linewidth

plt.xlabel('Amplitude (V)')
plt.ylabel(r'$\mathsf{\sqrt{\gamma} \;\; \left(\sqrt{Hz}\right)}$')

# plt.savefig('noise spetrum.png', dpi=600)  # 保存成png， 选择分辨率
plt.savefig('dephasing_with_amp.pdf')  # 保存成eps格式