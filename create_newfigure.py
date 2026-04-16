import matplotlib.pyplot as plt
import numpy as np

strategies = ['bounding_box', 'box_plus_center', 'multiple_points_10', 
              'multiple_points_5', 'multiple_points_3', 'corners_plus_center', 'center_point']
conditions = ['Optimal', 'High Complex', 'Medium Complex', 'Low Complex', 'Very Low']

iou_data = [
    [0.73, 0.21, 0.22, 0.12, 0.11],
    [0.61, 0.34, 0.33, 0.08, 0.07],
    [0.32, 0.14, 0.13, 0.02, 0.01],
    [0.09, 0.06, 0.05, 0.01, 0.00],
    [0.08, 0.05, 0.04, 0.00, 0.00],
    [0.08, 0.05, 0.04, 0.00, 0.00],
    [0.07, 0.05, 0.04, 0.00, 0.00]
]

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(iou_data, cmap='YlOrRd')

ax.set_xticks(np.arange(len(conditions)))
ax.set_yticks(np.arange(len(strategies)))
ax.set_xticklabels(conditions)
ax.set_yticklabels([s.replace('_', ' ').title() for s in strategies])

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('IoU Score', rotation=-90, va='bottom')

for i in range(len(strategies)):
    for j in range(len(conditions)):
        text = ax.text(j, i, f'{iou_data[i][j]:.2f}',
                       ha='center', va='center', 
                       color='white' if iou_data[i][j] > 0.3 else 'black',
                       fontsize=9, fontweight='bold')

ax.set_title('IoU Distribution Across Different Complexity Conditions', fontsize=16, pad=20)
ax.set_xlabel('Scene Complexity Level', fontsize=12)
ax.set_ylabel('Prompt Strategy', fontsize=12)

plt.tight_layout()
plt.savefig('iou_distribution_heatmap.png', dpi=300, bbox_inches='tight')
print('Figure saved as iou_distribution_heatmap.png')
plt.show()