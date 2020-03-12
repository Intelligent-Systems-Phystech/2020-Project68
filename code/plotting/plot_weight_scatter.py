import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../reports/class_ids/20NG_class_ids.csv')

# sns.pairplot(df, x_vars='label,ngrams,text'.split(','), y_vars=['f1_test_tm'])
# sns.pairplot(df['label,ngrams,text'.split(',') + ['f1_test_tm']], hue='f1_test_tm')
# grid = sns.PairGrid(data=df['label,ngrams,text'.split(',') + ['f1_test_tm']],
#                     vars = 'label,ngrams,text'.split(','), size = 4)
df['f1_test_tm'] = df['f1_test_tm'].round(2)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
g = sns.scatterplot(x="label", y="text",
                     hue="f1_test_tm", size="f1_test_tm",
                     sizes=(20, 200), hue_norm=(0.5, 0.6),
                     legend="full", data=df)
box = g.get_position()
g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

# Put a legend to the right side

g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

plt.show()