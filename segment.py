import matplotlib.pyplot as plt

import SimpleITK as sitk
from myshow import myshow, myshow3d
import numpy as np
from scipy import ndimage

# Download data to work on
# from downloaddata import fetch_data as fdata

class volumeSeg(object):
    def __init__(self,inputimage,path_array,cost_array=np.ones((1,1,1))*float('nan')):
        self.seeds = path_array
        self.inputimage = inputimage
        self.cost_array = cost_array
        self.path_array_indicies = np.ravel_multi_index(path_array.T, inputimage.shape)

    def convert2itk(self,inputimage,type=sitk.sitkUInt8):
        # converts to rescaled 8bit itk image format
        tmp = sitk.GetImageFromArray(np.swapaxes(inputimage,2,0))
        return (sitk.Cast(sitk.RescaleIntensity(tmp), type))

    def estimateRad(self):
        # apply distance transform
        dist_transform_image = ndimage.distance_transform_edt(self.mask_ActiveContour)
        #estimate along seed
        return(dist_transform_image.flat[self.path_array_indicies])


    def runSeg(self,radius=1):
        inim = self.convert2itk(self.inputimage) # converts to itk u8bit image
        cost = sitk.GetImageFromArray(np.swapaxes(self.cost_array,2,0))
        cc = sitk.GetArrayFromImage(cost)
        # if ~np.all(np.isnan(self.cost_array)):
        #     cost = self.convert2itk(self.cost_array)
        # else:
        #     cost = self.cost_array
        seg = sitk.Image(inim.GetSize(), sitk.sitkUInt8) # holder for initialization
        seg.CopyInformation(inim)
        for idx, seed in enumerate(self.seeds):
            seg[seed.tolist()] = 1
        # Binary dilate enlarges the seed mask by 3 pixels in all directions.
        seg = sitk.BinaryDilate(seg, radius)
        # based on thresholding
        self.mask_Threshold = np.swapaxes(sitk.GetArrayFromImage(self.segmentBasedOnThreshold(inim,seg)),2,0)
        # based on active contours
        self.mask_ActiveContour = np.swapaxes(sitk.GetArrayFromImage(self.segmentBasedOnActiveContours(inim,seg,cost)),2,0)

    def segmentBasedOnActiveContours(self,input,initseg,cost,radius=1):
        # We're going to build the following pipelines:
        # 1. reader -> smoothing -> gradientMagnitude -> sigmoid -> FI
        # 2. fastMarching -> geodesicActiveContour(FI) -> thresholder -> writer
        # The output of pipeline 1 is a feature image that is used by the
        # geodesicActiveContour object.  Also see figure 9.18 in the ITK
        dims = input.GetSize()
        seg = sitk.BinaryDilate(initseg, radius)
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(input, seg)

        # TimeStep=0.125,
        # NumberOfIterations=5,
        # ConductanceParameter=9.0)
        smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
        smoothing.SetTimeStep(0.0625)
        smoothing.SetNumberOfIterations(5)
        smoothing.SetConductanceParameter(9.0)
        smoothed_image = smoothing.Execute(sitk.Cast(input, sitk.sitkFloat32))

        grad = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
        grad_image = grad.Execute(smoothed_image)

        # sigmoid filter:
        # (Max-Min)*1/(1+e^(-(I-\betha)/\alpha))+Min

        beta = stats.GetMedian(1)#np.max((stats.GetMaximum(1) - stats.GetMedian(1),stats.GetMedian(1)))
        alpha = stats.GetSigma(1)/2
        sigmoid = sitk.SigmoidImageFilter()
        sigmoid.SetOutputMinimum(0.0)
        sigmoid.SetOutputMaximum(1.1)
        sigmoid.SetAlpha(alpha)
        sigmoid.SetBeta(beta)
        sigmoid_image = sigmoid.Execute(grad_image)
        # sitk.Show(sigmoid_image)
        if 0:
            fastMarching = sitk.FastMarchingImageFilter()
            for seed in self.seeds:
                fastMarching.AddTrialPoint((seed.tolist()))
            fastMarching_image = sitk.Cast(fastMarching.Execute(sigmoid_image), sitk.sitkFloat32)
        else:
            fastMarching_image = sitk.SignedMaurerDistanceMap(initseg, insideIsPositive=True, useImageSpacing=True)
            if cost.GetSize()==initseg.GetSize():
                fastMarching_image = fastMarching_image*sitk.Cast(cost, sitk.sitkFloat32)

        geoActiveCont = sitk.GeodesicActiveContourLevelSetImageFilter()
        geoActiveCont.SetNumberOfIterations(800)
        geoActiveCont.SetCurvatureScaling(1.0)
        geoActiveCont.SetAdvectionScaling(1.0)
        geoActiveCont.SetMaximumRMSError(0.02)
        geoActiveCont.ReverseExpansionDirectionOn()
        # geoActiveCont.SetPropagationScaling()
        geoActiveCont_image = geoActiveCont.Execute(fastMarching_image,sigmoid_image)

        binary = sitk.BinaryThresholdImageFilter()
        binary.SetLowerThreshold(0)
        binary.SetUpperThreshold(1000)
        binary.SetOutsideValue(0)
        binary.SetInsideValue(255)
        mask = binary.Execute(geoActiveCont_image)

        # estimate radius around initial trace

        return mask

    def segmentBasedOnThreshold(self,inim,seg):
        init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=False, useImageSpacing=True)
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(inim, seg)

        factor = .1
        lower_threshold =  stats.GetMinimum(1)-factor*stats.GetSigma(1)
        upper_threshold = 255  # np.min((255,stats.GetMean(1)+factor*stats.GetSigma(1)))

        lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
        lsFilter.SetLowerThreshold(lower_threshold)
        lsFilter.SetUpperThreshold(upper_threshold)
        lsFilter.SetMaximumRMSError(0.02)
        lsFilter.SetNumberOfIterations(1000)
        lsFilter.SetCurvatureScaling(1.5)
        lsFilter.SetPropagationScaling(1)
        lsFilter.ReverseExpansionDirectionOff()
        ls = lsFilter.Execute(init_ls, sitk.Cast(inim, sitk.sitkFloat32))
        mask = sitk.Cast(255*(ls < 0), sitk.sitkUInt8)
        if 0:
            simg_255 = sitk.Cast(sitk.RescaleIntensity(inim), sitk.sitkUInt8)
            idx = self.seeds[0]
            zslice_offset = 1
            # myshow3d(sitk.LabelOverlay(self.inputimage, seg),
            #          zslices=range(idx[2] - zslice_offset, idx[2] + zslice_offset + 1, zslice_offset), dpi=20, title='init')

            # myshow3d(sitk.LabelOverlay(simg_255, ls <= 0),
            #          zslices=range(idx[2] - zslice_offset, idx[2] + zslice_offset + 1, zslice_offset), dpi=20, title='test')

            myshow3d(sitk.LabelOverlay(self.inputimage, seg),zslices=[17,18,19,20,21], dpi=20, title='init')
            myshow3d(sitk.LabelOverlay(simg_255, mask),zslices=[17,18,19,20,21], dpi=20, title='init')

        return mask
