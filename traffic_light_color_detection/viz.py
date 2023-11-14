import matplotlib.pyplot as plt
import numpy as np



# TODO - Modify visualization code to render images properly in the human visual spetrum
def viz_layer(pics, pic_titles):
    fig = plt.figure(figsize=(20, 20))

    for i in range(len(pics)):
        ax = fig.add_subplot(1, len(pics), i+1, xticks=[], yticks=[])
        plt.axis('off')
        ax.imshow(np.squeeze(pics[i]))
        ax.set_title(pic_titles[i])

    plt.show()