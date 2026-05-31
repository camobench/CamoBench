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
High-Frequency Texture,Textual Elements,1.309,4.031,0.838,1.509
High-Frequency Texture,Geometry & Shapes,1.244,4.194,0.847,1.478
High-Frequency Texture,Rigid Objects,1.456,4.075,0.784,1.891
High-Frequency Texture,Non-Rigid Objects,1.422,4.156,0.816,1.816
Structured Texture,Textual Elements,1.256,4.106,0.863,1.391
Structured Texture,Geometry & Shapes,1.222,4.281,0.819,1.394
Structured Texture,Rigid Objects,1.328,4.156,0.806,1.619
Structured Texture,Non-Rigid Objects,1.353,3.944,0.856,1.688
Directional Texture,Textual Elements,1.238,4.331,0.894,1.359
Directional Texture,Geometry & Shapes,1.206,4.481,0.884,1.403
Directional Texture,Rigid Objects,1.331,4.594,0.856,1.663
Directional Texture,Non-Rigid Objects,1.3,4.287,0.9,1.587
Low-Frequency Smooth Texture,Textual Elements,1.163,4.463,0.844,1.281
Low-Frequency Smooth Texture,Geometry & Shapes,1.228,4.45,0.809,1.441
Low-Frequency Smooth Texture,Rigid Objects,1.325,4.55,0.797,1.637
Low-Frequency Smooth Texture,Non-Rigid Objects,1.384,4.562,0.825,1.747"""

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

plt.savefig("./data/score_analysis/rq3_heatmaps_editing.pdf", format='pdf', bbox_inches='tight')
plt.show()