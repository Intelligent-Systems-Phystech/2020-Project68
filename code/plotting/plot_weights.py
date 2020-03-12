import os
import glob
from pathlib import Path

from matplotlib import pylab as plt
import seaborn as sns

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

for file in glob.glob('../reports/*class_ids.csv'):
    df = pd.read_csv(file)
    dataset_name = Path(file).stem
    # sns.pairplot(df[['label', 'ngrams', 'text', 'f1_test_tm']])
    plt.figure(figsize=(20, 10))
    sns.pairplot(data=df,
                 y_vars=['f1_test_tm'],
                 x_vars=['label', 'ngrams', 'text'])

    # plt.scatter(df['label'], df['f1_test_tm'], marker='*')
    # plt.scatter(df['ngrams'], df['f1_test_tm'], marker='o')
    # plt.scatter(df['text'], df['f1_test_tm'], marker='^')

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(df['label'], df['ngrams'], df['f1_test_tm'],cmap='viridis', edgecolor='none')

    plt.title(dataset_name)
    plt.xlabel('weight')
    plt.ylabel('f1_test')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('img/{}_topic_num.svg'.format(dataset_name))
    plt.show()
    plt.clf()

    # print(dataset_name, df.iloc[df['f1_test_tm'].idxmax()]['topic_num'])
