import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_conf_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, figsize = (5, 5), colorbar = True, percentage = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion Matrix (Dengan Normalisasi)'
        else:
            title = 'Confusion Matrix (Tanpa Normalisasi)'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if percentage:
            cm = cm * 100

    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if colorbar:
        ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='Label sebenarnya',
            xlabel='Label prediksi')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            content = cm[i, j]

            if normalize:
                content = f'{cm[i, j]:.2f}'

                if percentage:
                    content = f'{content} %'

            ax.text(j, i, content,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
