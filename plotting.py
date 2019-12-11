import csv
import collections
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scale = 0.025

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

fig = plt.figure()
ax = plt.subplot(111)
for run, results in data.items():

    if '6L-java' in run:
        y = [i+scale for i in results['valid_mrr']]
    else:
        y = results['valid_mrr']

    if 'model=transformer' not in run:
        continue
    if 'bpe_pct=0.5' not in run:
        continue
    if 'seed=1' not in run:
        continue

    max_epochs = 25

    y = y[:max_epochs]

    x = list(range(1, len(y)+1))
    
    if 'lang=java-strip' in run:
        line_style = '--'
    elif 'lang=6L-java' in run:
        line_style = '-.'
    else:
        line_style = '-'

    label = run.split('_')[0].split('=')[-1]

    ax.plot(x, y, line_style, label = label)
    ax.scatter(x[-1], y[-1], marker = '*', s = 100)
ax.grid()
ax.set_title('MMR vs Epochs for 6L-Java, Java and Java-Strip Datasets')
ax.set_ylabel('MMR')
ax.set_xlabel('Epochs')
ax.set_yticks([0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6])
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
#lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
ax.legend(loc='lower right')
#fig.savefig('test', bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig('test', bbox_inches='tight')