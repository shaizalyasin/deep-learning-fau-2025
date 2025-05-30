import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):  # constrcutor implementation
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size.")
        self.resolution = resolution  # instance variable
        self.tile_size = tile_size
        self.output = []  # this will later hold gernated pattern when draw method is called

    def draw(self):
        tile = np.array([[0, 1], [1, 0]])  # 2x2 numpy array 0 represent black 1 represents white
        reps = self.resolution // (2 * self.tile_size)  # cumber of times tile is supposed to be repeated
        checker = np.tile(tile, (reps, reps))  # make a large 2d bases for the cheker board, based on the number of
        # tiles(actually the reps representing number of tiles)

        self.output = np.repeat(np.repeat(checker, self.tile_size, axis=0), self.tile_size,
                                axis=1)  # Enlarges each tile to a tile_size x tile_size block by repeating rows and columns.
        return self.output.copy()

    def show(self):
        plt.imshow(self.output,
                   cmap='gray')  # cmpa is grayscale whihc convenietnly recognizes 0 and 1 as max and low intensities
        plt.axis('off')
        plt.title('Checkerboard')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):  # variables for postioning and size
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = []

    def draw(self):  # generates the circel in 2d num array
        # following 2 create x and y coordinates for grid array
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)  # mesh for all pixel positions
        distance = np.sqrt((xx - self.position[0]) ** 2 + (
                    yy - self.position[1]) ** 2)  # euclid distance calculated to check each pixel
        # location outside circle radius or inside
        self.output = (distance <= self.radius).astype(
            float)  # so simlar to the checkboard,images inside the circle are set to 1.0(white) and the rest are 0.0(black)
        return self.output.copy()  # returns copy of image

    def show(self):
        plt.imshow(self.output, cmap='gray')  # display image copied in self output instance
        plt.axis('off')
        plt.title('Binary Circle')
        plt.show()


class Spectrum:
    def __init__(self, resolution):  # cosntructor
        self.resolution = resolution
        self.output = []

    def draw(self):
        x = np.linspace(0, 1,
                        self.resolution)  # 1d array spaced evenly from 0-1, resolution gives numbers of pixels(squares)
        y = np.linspace(0, 1, self.resolution)  # same for vertical
        xx, yy = np.meshgrid(x, y)  #
        red = xx  # these are color channels to set the intensity of red color from left to right
        green = yy  # increases top to bottom
        blue = 1 - xx  # blue channel decreases from left to right
        self.output = np.stack((red, green, blue),
                               axis=-1)  # this is the function where a 3d matrix is created and it basically help
                                        # the imshow fucntion recognize the as colors
        return self.output.copy()

    def show(self):
        plt.imshow(
            self.output)  # When you pass a 3D array with shape (height, width, 3) to plt.imshow(), Matplotlib automatically interprets it as an RGB image.
        plt.axis('off')
        plt.title('RGB Spectrum')
        plt.show()
