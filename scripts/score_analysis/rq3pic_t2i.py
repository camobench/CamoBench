import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'Georgia', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.titlesize'] = 13

csv_data = """Background,Hidden Object,IF,VQ,SC,ISC
High-Frequency Texture,Textual Elements,1.497,3.817,0.988,1.738
High-Frequency Texture,Geometry & Shapes,1.421,4.462,0.992,1.721
High-Frequency Texture,Rigid Objects,1.585,4.326,0.979,2.113
High-Frequency Texture,Non-Rigid Objects,1.652,4.15,0.984,2.205
Structured Texture,Textual Elements,1.297,3.775,0.972,1.485
Structured Texture,Geometry & Shapes,1.305,4.308,0.973,1.559
Structured Texture,Rigid Objects,1.408,4.292,0.967,1.797
Structured Texture,Non-Rigid Objects,1.508,3.788,0.96,1.984
Directional Texture,Textual Elements,1.367,4.118,0.969,1.59
Directional Texture,Geometry & Shapes,1.429,4.688,0.987,1.763
Directional Texture,Rigid Objects,1.635,4.556,0.951,2.259
Directional Texture,Non-Rigid Objects,1.663,4.464,0.969,2.253
Low-Frequency Smooth Texture,Textual Elements,1.348,4.125,0.946,1.565
Low-Frequency Smooth Texture,Geometry & Shapes,1.406,4.545,0.952,1.652
Low-Frequency Smooth Texture,Rigid Objects,1.482,4.607,0.926,1.931
Low-Frequency Smooth Texture,Non-Rigid Objects,1.579,4.65,0.934,2.112"""

df = pd.read_csv(io.StringIO(csv_data))

bg_order = ["High-Frequency Texture", "Structured Texture", "Directional Texture", "Low-Frequency Smooth Texture", "Average"]
obj_order = ["Textual Elements", "Geometry & Shapes", "Rigid Objects", "Non-Rigid Objects", "Average"]

bg_labels = ["High\nFrequency", "Structured", "Directional", "Low\nFrequency", "Average"]
obj_labels = ["Textual Elements", "Geometry & Shapes", "Rigid Objects", "Non-Rigid Objects", "Average"]
metrics = ['IF', 'VQ', 'SC', 'ISC']

fig, axes = plt.subplots(2, 2, figsize=(8.5, 9))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    pivot_df = df.pivot(index='Background', columns='Hidden Object', values=metric)
    pivot_df = pivot_df.reindex(index=bg_order, columns=obj_order)
    pivot_df['Average'] = pivot_df.mean(axis=1)
    pivot_df.loc['Average'] = pivot_df.mean(axis=0)
    mat = pivot_df.values
    
    ax = axes[i]
    
    show_y = (i % 2 == 0)  
    show_x = (i >= 2)      
    
    vmin, vmax = np.min(mat), np.max(mat)

    sns.heatmap(mat, 
                annot=True,            
                fmt=".2f",             
                cmap="crest",  
                vmin=vmin, vmax=vmax,       
                annot_kws={"size": 11},
                linewidths=1,        
                linecolor='white',
                cbar=False,            
                square=True,           
                xticklabels=obj_labels if show_x else False, 
                yticklabels=bg_labels if show_y else False,  
                ax=ax)
    
    if show_x:
        ax.set_xticklabels(obj_labels, rotation=20, ha='right')
    if show_y:
        ax.set_yticklabels(bg_labels, rotation=0, ha='right')
    
    metric_full_names = {'IF': 'Illusion Fidelity (IF)', 'VQ': 'Overall Visual Quality (VQ)', 
                         'SC': 'Scene Consistency (SC)', 'ISC': 'Illusion Shape Consistency (ISC)'}
    ax.set_title(metric_full_names[metric], pad=6, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.subplots_adjust(wspace=0.05, hspace=0.1) 

plt.savefig("./data/score_analysis/rq3_heatmaps_t2i.pdf", format='pdf', bbox_inches='tight')
plt.show()