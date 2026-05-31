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
High-Frequency Texture,Textual Elements,2.266,2.965,0.891,2.664
High-Frequency Texture,Geometry & Shapes,2.215,3.051,0.938,2.857
High-Frequency Texture,Rigid Objects,1.957,3.84,0.967,2.223
High-Frequency Texture,Non-Rigid Objects,1.912,3.781,0.984,2.1
Structured Texture,Textual Elements,2.264,2.777,0.818,2.588
Structured Texture,Geometry & Shapes,2.205,2.742,0.807,2.861
Structured Texture,Rigid Objects,1.961,3.25,0.857,2.258
Structured Texture,Non-Rigid Objects,1.803,3.188,0.857,2.025
Directional Texture,Textual Elements,2.143,3.164,0.812,2.6
Directional Texture,Geometry & Shapes,2.166,3.355,0.822,2.814
Directional Texture,Rigid Objects,1.953,3.957,0.898,2.25
Directional Texture,Non-Rigid Objects,1.832,3.988,0.945,2.082
Low-Frequency Smooth Texture,Textual Elements,1.971,3.336,0.861,2.289
Low-Frequency Smooth Texture,Geometry & Shapes,2.07,3.484,0.844,2.553
Low-Frequency Smooth Texture,Rigid Objects,1.961,3.805,0.83,2.309
Low-Frequency Smooth Texture,Non-Rigid Objects,1.758,3.746,0.83,2.104"""

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

plt.savefig("./data/score_analysis/rq3_heatmaps_condgen.pdf", format='pdf', bbox_inches='tight')
plt.show()