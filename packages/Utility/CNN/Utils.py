import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

def showKernel(model, layer, index_filter = -1, index_channel = -1, figsize = (25, 25)):
    # Mendapatkan weight / kernel
    weights = model.get_layer(layer).get_weights()[0]
    
    # Konfigurasi plotting
    fig = plt.figure(figsize = figsize)
    grid = GridSpec(nrows = weights.shape[3], ncols = 1, figure = fig)
    nested_grid = []

    # Menampilkan weights per kernel
    range_kernel = range(index_filter, index_filter + 1) if index_filter > -1 else range(weights.shape[3])
    for i in range_kernel:
        nrows = int(np.ceil(weights.shape[2] / 6))
        ncols = 6
        nested_grid.append(GridSpecFromSubplotSpec(nrows = nrows, ncols = ncols, subplot_spec = grid[i]))
        
        range_channel = range(index_channel, index_channel + 1) if index_channel > -1 else range(weights.shape[2])
        for j in range_channel:
            row_pos = int(np.ceil((j + 1) / ncols))
            col_pos = int(j + 1 - (6 * (row_pos - 1)))
            
            ax = plt.Subplot(fig, nested_grid[i][row_pos - 1, col_pos - 1])
            ax.imshow(weights[:, :, j, i], cmap='gray')
            ax.title.set_text(f'Kernel {i+1}, channel {j+1}')
            fig.add_subplot(ax)
            plt.xticks([])
            plt.yticks([])

    plt.show()