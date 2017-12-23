# from morphsnakes import morphsnakes
import morphsnakes
#
import skimage.io as io
import numpy as np
from matplotlib import pyplot as ppl


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u


def test_nodule():
    # Load the image.
    img = io.imread("./morphsnakes/testimages/mama07ORI.bmp")[..., 0] / 255.0

    # g(I)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
    mgac.levelset = circle_levelset(img.shape, (100, 126), 20)

    # Visual evolution.
    ppl.figure()
    morphsnakes.evolve_visual(mgac, num_iters=45, background=img)


def test_starfish():
    # Load the image.
    imgcolor = io.imread("./morphsnakes/testimages/seastar2.png") / 255.0
    img = rgb2gray(imgcolor)

    # g(I)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=2)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-1)
    mgac.levelset = circle_levelset(img.shape, (163, 137), 135, scalerow=0.75)

    # Visual evolution.
    ppl.figure()
    morphsnakes.evolve_visual(mgac, num_iters=110, background=imgcolor)


def test_lakes():
    # Load the image.
    imgcolor = io.imread("./morphsnakes/testimages/lakes3.jpg") / 255.0
    img = rgb2gray(imgcolor)

    # MorphACWE does not need g(I)

    # Morphological ACWE. Initialization of the level-set.
    macwe = morphsnakes.MorphACWE(img, smoothing=3, lambda1=1, lambda2=1)
    macwe.levelset = circle_levelset(img.shape, (80, 170), 25)

    # Visual evolution.
    ppl.figure()
    morphsnakes.evolve_visual(macwe, num_iters=190, background=imgcolor)


def test_confocal3d(img):

    if False:
        macwe = morphsnakes.MorphGAC(img, smoothing = 1, threshold = 0.3, balloon = -1)
    else:
        macwe = morphsnakes.MorphACWE(img, smoothing=1, lambda1=1, lambda2=2)
    macwe.levelset = circle_levelset(img.shape, (30, 50, 80), 25)

    if True:
        # Visual evolution.
        morphsnakes.evolve_visual3d(macwe, num_iters=200)
    else:
        fig = ppl.figure(frameon=False)
        for i in range(100):
            macwe.step()
            fd = macwe.levelset
            ppl.imshow(np.max(img, axis=2), cmap='gray')
            ppl.imshow(np.max(fd[:, :, :], axis=2), cmap='jet', alpha=0.3)
            ppl.show()
            ppl.pause(.1)
            ppl.draw()


if __name__ == '__main__':
    print("""""")
    test_nodule()
    test_starfish()
    test_lakes()
    test_confocal3d()
    ppl.show()
