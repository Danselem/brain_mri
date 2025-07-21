import numpy as np
import matplotlib.pyplot as plt

def visualize_batch(train_loader, train_dataset, mean=None, std=None, images_per_row=4, figsize=(15, 15)):
    """
    Visualizes a batch of images from a DataLoader.

    Parameters:
    - train_loader: DataLoader providing batches of images and labels.
    - train_dataset: Dataset object containing class names.
    - mean: List or array of channel-wise means for denormalization. Default is ImageNet mean.
    - std: List or array of channel-wise stds for denormalization. Default is ImageNet std.
    - images_per_row: Number of images to display per row in the grid.
    - figsize: Tuple for the figure size.
    """
    # Default normalization values for ImageNet
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    else:
        mean = np.array(mean)

    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    else:
        std = np.array(std)

    # Load a batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Convert to NumPy and denormalize
    images = (images.numpy().transpose((0, 2, 3, 1)) * std + mean).clip(0, 1)

    num_images = len(images)
    rows = int(np.ceil(num_images / images_per_row))
    fig, axes = plt.subplots(rows, images_per_row, figsize=figsize)

    # Flatten axes for easy iteration
    axes = axes.flat if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(f'Label: {train_dataset.classes[labels[i]]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


