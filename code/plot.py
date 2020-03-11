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

df = pd.read_excel('data/reports/R8_class_ids.xlsx')
plt.scatter(df['label'], df['f1_test_tm'])
plt.scatter(df['text'], df['f1_test_tm'], marker='.')
plt.xlabel('num_topics')
plt.ylabel('f1_test')

plt.grid(True)
plt.tight_layout()
plt.savefig('1.svg') # Поддерживаемые форматы: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff