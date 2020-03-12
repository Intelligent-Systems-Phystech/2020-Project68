import os
import glob
from pathlib import Path


from matplotlib import pylab as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 12
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 24
### Сохранение изображения ###

import pandas as pd

for file in glob.glob('reports/*.csv'):

    df = pd.read_csv(file)
    dataset_name = Path(file).stem

    df = df.sort_values('topic_num')
    plt.plot(df['topic_num'], df['f1_test_tm'])
    plt.title(dataset_name)
    plt.xlabel('num_topics')
    plt.ylabel('f1_test')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('img/{}_topic_num.svg'.format(dataset_name))
    plt.clf()

    print(dataset_name, df.iloc[df['f1_test_tm'].idxmax()]['topic_num'])