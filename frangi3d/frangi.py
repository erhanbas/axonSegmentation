import numpy as np
# import vigra as vig
from skimage import exposure

from .utils import divide_nonzero
from .hessian import absolute_hessian_eigenvalues
from .utils import absolute_eigenvaluesh
from vigra.filters import hessianOfGaussianEigenvalues as eHoG
import matplotlib.pyplot as plt


def frangi(nd_array, sigmas, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True, window_size = 2.0):

    if not nd_array.ndim == 3:
        raise(ValueError("Only 3 dimensions is currently supported"))

    # from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_frangi.py#L74
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    filtered_array = np.zeros(sigmas.shape + nd_array.shape)
    # window_size = 3.0
    step_size = [1.0, 1.0, 3.0]
    # roi = ((10, 20), (200, 250))
    for i_sigma, sigma in enumerate(np.asarray(sigmas,np.float32)):
        if False:
            eigenvalues = absolute_hessian_eigenvalues(nd_array, sigma=sigma, scale=True)
        else:
            # sigma_d = np.min((np.array([1.0,1.0,1.0]),scale),axis=0)
            scale = sigma*np.array([1,1,1])
            if np.any(scale*window_size >= nd_array.shape):
                break # skip
            eigenvalues_vig = eHoG(nd_array,scale=scale.tolist(), window_size=window_size, step_size=step_size)#sigma_d=sigma_d.tolist())
            # scale eigenvalues
            eigenvalues_vig *= scale**2
            eigenvalues=np.split(eigenvalues_vig,3,axis=3)

        filtered_array[i_sigma] = compute_vesselness(*eigenvalues, alpha=alpha, beta=beta, c=frangi_c,
                                               black_white=black_vessels)
    filtresponse = np.max(filtered_array, axis=0)
    scaleresponse = sigmas[np.argmax(filtered_array, axis=0)]

    # plt.close('all')
    filtres = np.max(filtresponse, axis=2)
    filtres_inds = np.argmax(filtresponse, axis=2)
    # create grid
    np.mgrid((filtres_inds.shape))


    gamma_corrected = exposure.adjust_gamma(filtresponse, .5)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(np.max(nd_array, axis=2).T, cmap='gray')
    ax2.imshow(np.max(filtresponse, axis=2).T, cmap='gray')
    ax3.imshow(np.max(gamma_corrected, axis=2).T, cmap='gray')
    ax4.imshow(np.max(scaleresponse * (filtresponse > .2), axis=2).T, cmap='gray')


    return filtresponse, scaleresponse

def sortbyabs(a, axis=0):
    """Sort array along a given axis by the absolute value
    modified from: http://stackoverflow.com/a/11253931/4067734
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = np.abs(a).argsort(axis)
    return a[index]

def compute_measures(eigen1, eigen2, eigen3):
    """
    RA - plate-like structures
    RB - blob-like structures
    S - background
    """
    Ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    Rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    S = np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
    return Ra, Rb, S


def compute_plate_like_factor(Ra, alpha):
    return 1 - np.exp(np.negative(np.square(Ra)) / (2 * np.square(alpha)))


def compute_blob_like_factor(Rb, beta):
    return np.exp(np.negative(np.square(Rb) / (2 * np.square(beta))))


def compute_background_factor(S, c):
    return 1 - np.exp(np.negative(np.square(S)) / (2 * np.square(c)))


def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    if np.ndim(eigen1)>3:
        eigen1 = np.squeeze(eigen1)
        eigen2 = np.squeeze(eigen2)
        eigen3 = np.squeeze(eigen3)
    Ra, Rb, S = compute_measures(eigen1, eigen2, eigen3)
    plate = compute_plate_like_factor(Ra, alpha)
    blob = compute_blob_like_factor(Rb, beta)
    background = compute_background_factor(S, c)
    return filter_out_background(plate * blob * background, black_white, eigen2, eigen3)


def filter_out_background(voxel_data, black_white, eigen2, eigen3):
    """
    Set black_white to true if vessels are darker than the background and to false if
    vessels are brighter than the background.
    """
    if black_white:
        voxel_data[eigen2 < 0] = 0
        voxel_data[eigen3 < 0] = 0
    else:
        voxel_data[eigen2 > 0] = 0
        voxel_data[eigen3 > 0] = 0
    voxel_data[np.isnan(voxel_data)] = 0
    return voxel_data
