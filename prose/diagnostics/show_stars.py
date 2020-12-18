from prose.blocks.base import Block
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from prose import viz


class ShowStars(Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, image, **kwargs):
        fig = viz.fancy_show_stars(image.data, image.stars_coords)
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image.data = np.asarray(buf)
        plt.close()
