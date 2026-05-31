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
High-Frequency Texture,Textual Elements,1.690,3.604,0.931,1.970
High-Frequency Texture,Geometry & Shapes,1.623,3.994,0.949,2.013
High-Frequency Texture,Rigid Objects,1.671,4.135,0.939,2.104
High-Frequency Texture,Non-Rigid Objects,1.686,4.042,0.953,2.102
Structured Texture,Textual Elements,1.576,3.541,0.906,1.795
Structured Texture,Geometry & Shapes,1.556,3.839,0.895,1.914
Structured Texture,Rigid Objects,1.557,3.958,0.905,1.900
Structured Texture,Non-Rigid Objects,1.567,3.639,0.910,1.942
Directional Texture,Textual Elements,1.573,3.875,0.909,1.847
Directional Texture,Geometry & Shapes,1.606,4.255,0.919,2.008
Directional Texture,Rigid Objects,1.673,4.385,0.918,2.146
Directional Texture,Non-Rigid Objects,1.646,4.291,0.949,2.079
Low-Frequency Smooth Texture,Textual Elements,1.498,3.954,0.902,1.727
Low-Frequency Smooth Texture,Geometry & Shapes,1.570,4.213,0.894,1.880
Low-Frequency Smooth Texture,Rigid Objects,1.595,4.359,0.874,1.988
Low-Frequency Smooth Texture,Non-Rigid Objects,1.596,4.366,0.883,2.042"""

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

plt.savefig("./data/score_analysis/rq3_heatmaps.pdf", format='pdf', bbox_inches='tight')
plt.show()