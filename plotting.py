import csv
import collections
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

runs = os.listdir('runs')

data = dict()

for root, dirs, files in os.walk('runs', topdown = False):
    for name in files:
        if 'results' in name:
            with open(os.path.join(root, name), 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                results = collections.defaultdict(list)
                for row in reader:
                    for h, r in zip(header, row):
                        if r == 'lost_patience':
                            break
                        else:
                            results[h].append(float(r))
            
            run = '_'.join(root.split('/')[1:-1])

            print(run)
            
            data[run] = results 

fig = plt.figure(figsize=(20,20))
ax = plt.subplot(111)
for run, results in data.items():
    if 'model=transformer' not in run:
        continue
    if 'bpe_pct=0.5' not in run:
        continue
    y = results['valid_mrr']
    x = list(range(1, len(y)+1))
    ax.plot(x, y, '-' if 'lang=java-strip' in run else '--', label=run)
    ax.scatter(x[-1],y[-1], marker='*', s=100)
ax.grid()
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
fig.savefig('test', bbox_extra_artists=(lgd,), bbox_inches='tight')