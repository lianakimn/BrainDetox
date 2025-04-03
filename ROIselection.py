import matplotlib

import sys
import matplotlib

if sys.platform == 'win32':
    try:
        import PyQt5  # Try to import PyQt5 for Qt5Agg
        matplotlib.use('Qt5Agg')
    except ImportError:
        # Fallback to TkAgg if PyQt5 is not available
        matplotlib.use('TkAgg')
elif sys.platform == 'darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt



import numpy as np
from matplotlib.path import Path

def roi_selection(image):
    """
    Function to interactively select a single ROI on a given image.

    Args:
        image (np.ndarray): 2D image on which an ROI is to be drawn.

    Returns:
        list: The selected ROI as a list of (x, y) coordinates.
        np.ndarray: Binary mask of the selected ROI.
    """
    roi = []  # Temporary list for the ROI

    def on_click(event):
        nonlocal roi

        if event.dblclick:  # Close ROI on double-click
            if len(roi) > 2:  # Ensure there are enough points for a valid polygon
                print("ROI selection finished.")
                fig.canvas.mpl_disconnect(cid)  # Disconnect the event listener
                redraw_plot(closed=True)  # Finalize the plot with the closed ROI
                plt.close(fig)  # Close the figure after ROI is completed
            else:
                print("Need at least 3 points to close the ROI.")
        elif event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Add the clicked point to the ROI
            roi.append((event.xdata, event.ydata))
            print(f"Clicked at ({event.xdata:.2f}, {event.ydata:.2f})")

            # Redraw the plot
            redraw_plot()

    def redraw_plot(closed=False):
        """Redraw the plot with the current ROI."""
        ax.clear()
        ax.imshow(image, cmap="gray")  # Redraw the image

        if len(roi) > 1:
            x_coords, y_coords = zip(*roi)
            ax.plot(x_coords, y_coords, 'r-o')  # Draw lines between points

            if closed:
                ax.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'r--')  # Close the ROI loop

        fig.canvas.draw()  # Update the figure

    def get_roi_mask(image_shape, roi):
        """Generate a binary mask for the given ROI."""
        path = Path(roi)
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        points = np.vstack((x.flatten(), y.flatten())).T
        mask = path.contains_points(points).reshape(image_shape)
        return mask

    # Show the image for ROI selection
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    plt.title("Draw ROI by left-clicking points. Double-click to finish.")
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    # Generate the mask for the ROI
    if len(roi) > 2:  # Ensure ROI was completed
        mask = get_roi_mask(image.shape, roi)
    else:
        raise ValueError("ROI selection was not completed.")

    return roi, mask
