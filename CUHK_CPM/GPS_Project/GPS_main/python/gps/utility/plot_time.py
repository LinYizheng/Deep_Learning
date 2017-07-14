import numpy as np
import cPickle as pickle

import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

gps_root = "/home/pi/gps_rr/"
exp_name = "ja_mode.experiment"

filename = gps_root + 'experiments/' + exp_name + '/average_time.pkl'

data = pickle.load(open(filename, 'rb'))

total, sample, train = data['total'], data['sample'], data['train']

rest = total - sample - train

x = np.arange(total.shape[0])

print 'mean total', np.mean(total)
print 'mean sample', np.mean(sample)
print 'mean train', np.mean(train)
print 'mean others', np.mean(rest)

plt.figure(1)
lines = plt.plot(x, total, x, sample, x, train, x, rest)
plt.legend(lines, ['total', 'sampling', 'training', 'others'], loc=0)
plt.xlabel('Iteration')
plt.ylabel('Time[seconds]')
plt.title('Elapsed time of each iteration')
plt.show()