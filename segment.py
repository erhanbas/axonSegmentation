import matplotlib.pyplot as plt

import SimpleITK as sitk
from myshow import myshow, myshow3d
import numpy as np
# Download data to work on
# from downloaddata import fetch_data as fdata

class volumeSeg(object):
    def __init__(self,inputimage,path_array):
        self.seeds = path_array
        self.inputimage = inputimage

    def convert2itk(self,inputimage):
            tmp = sitk.GetImageFromArray(np.swapaxes(inputimage,2,0))
            return (sitk.Cast(sitk.RescaleIntensity(tmp), sitk.sitkUInt8))

    def runSeg(self,radius=1):
        inim = self.convert2itk(self.inputimage) # converts to itk u8bit image
        seg = sitk.Image(inim.GetSize(), sitk.sitkUInt8) # holder for initialization
        seg.CopyInformation(inim)
        for idx, seed in enumerate(self.seeds):
            seg[seed.tolist()] = 1
        # Binary dilate enlarges the seed mask by 3 pixels in all directions.
        seg = sitk.BinaryDilate(seg, radius)
        # based on thresholding
        self.mask = self.segmentBasedOnThreshold(inim,seg)
        # based on active contours
        self.mask = self.segmentBasedOnActiveContours(inim,seg)

    def segmentBasedOnActiveContours(self,input,initseg):
        # We're going to build the following pipelines:
        # 1. reader -> smoothing -> gradientMagnitude -> sigmoid -> FI
        # 2. fastMarching -> geodesicActiveContour(FI) -> thresholder -> writer
        # The output of pipeline 1 is a feature image that is used by the
        # geodesicActiveContour object.  Also see figure 9.18 in the ITK
        dims = input.GetSize()

        # TimeStep=0.125,
        # NumberOfIterations=5,
        # ConductanceParameter=9.0)
        smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
        smoothing.SetTimeStep(0.0625)
        smoothing.SetNumberOfIterations(5)
        smoothing.SetConductanceParameter(9.0)
        smoothed_image = smoothing.Execute(sitk.Cast(input, sitk.sitkFloat32))

        grad = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
        grad.SetSigma(1.0)
        grad_image = grad.Execute(smoothed_image)

        sigmoid = sitk.SigmoidImageFilter()
        sigmoid.SetOutputMinimum(0.0)
        sigmoid.SetOutputMaximum(1.1)
        sigmoid.SetAlpha(-.05)
        sigmoid.SetBeta(3)
        sigmoid_image = sigmoid.Execute(grad_image)

        fastMarching = sitk.FastMarchingImageFilter()
        for seed in self.seeds:
            fastMarching.AddTrialPoint((seed.tolist()))
        fastMarching_image = fastMarching.Execute(sigmoid_image)

    # geodesicActiveContour = itk.GeodesicActiveContourLevelSetImageFilter[
    #     InternalImageType, InternalImageType, InternalPixelType].New(fastMarching, sigmoid,
    #                                                                  PropagationScaling=float(argv[9]),
    #                                                                  CurvatureScaling=1.0,
    #                                                                  AdvectionScaling=1.0,
    #                                                                  MaximumRMSError=0.02,
    #                                                                  NumberOfIterations=800
    #                                                                  )
    #
    # thresholder = itk.BinaryThresholdImageFilter[InternalImageType, OutputImageType].New(geodesicActiveContour,
    #                                                                                      LowerThreshold=-1000,
    #                                                                                      UpperThreshold=0,
    #                                                                                      OutsideValue=0,
    #                                                                                      InsideValue=255)

    def segmentBasedOnThreshold(self,inim,seg):
        init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=False, useImageSpacing=True)
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(inim, seg)

        factor = .5
        lower_threshold =  stats.GetMean(1)-factor*stats.GetSigma(1)
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
        mask = ls < 0
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

        return (mask)


            # def tmp(self):
    #     idx = tuple(initLocation)
    #     pt = simg.TransformIndexToPhysicalPoint(idx)
    #     seg = sitk.Image(simg.GetSize(), sitk.sitkUInt8)
    #     seg.CopyInformation(simg)
    #
    #     seg[idx] = 1
    #     seg = sitk.BinaryDilate(seg, 1)
    #
    #     stats = sitk.LabelStatisticsImageFilter()
    #     stats.Execute(simg, seg)
    #
    #     factor = 2
    #     lower_threshold = 25  # stats.GetMean(1)
    #     upper_threshold = 255  # np.min((255,stats.GetMean(1)+factor*stats.GetSigma(1)))
    #
    #     init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=False, useImageSpacing=True)
    #     myshow3d(init_ls, zslices=range(idx[2] - zslice_offset, idx[2] + zslice_offset + 1, zslice_offset), dpi=20,
    #              title=t)
    #
    #     lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    #     lsFilter.SetLowerThreshold(lower_threshold)
    #     lsFilter.SetUpperThreshold(upper_threshold)
    #     lsFilter.SetMaximumRMSError(0.02)
    #     lsFilter.SetNumberOfIterations(1000)
    #     lsFilter.SetCurvatureScaling(1.5)
    #     lsFilter.SetPropagationScaling(1)
    #     lsFilter.ReverseExpansionDirectionOn()
    #     ls = lsFilter.Execute(init_ls, sitk.Cast(simg, sitk.sitkFloat32))
    #
    #     t = "LevelSet after " + str(lsFilter.GetNumberOfIterations()) + " iterations"
    #     zslice_offset = 1
    #     sitk.Show(ls > 0)
    #     myshow3d(sitk.LabelOverlay(simg_255, ls > 0),
    #              zslices=range(idx[2] - zslice_offset, idx[2] + zslice_offset + 1, zslice_offset), dpi=20, title=t)
        #
        # gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
        # gradientMagnitude.SetSigma(2)
        # featureImage = sitk.BoundedReciprocal( gradientMagnitude.Execute(simg))
        # geodesicActiveContour = sitk.GeodesicActiveContourLevelSetImageFilter()
        # geodesicActiveContour.SetPropagationScaling(1)
        # geodesicActiveContour.SetCurvatureScaling(.5)
        # geodesicActiveContour.SetAdvectionScaling(1.0)
        # geodesicActiveContour.SetMaximumRMSError(0.01)
        # geodesicActiveContour.SetNumberOfIterations(1000)
        #
        # init_ls = sitk.Cast(init_ls, featureImage.GetPixelID()) * -1 + 0.5
        # levelset = geodesicActiveContour.Execute(init_ls, featureImage )
        #
        # print( "RMS Change: ", geodesicActiveContour.GetRMSChange() )
        # print( "Elapsed Iterations: ", geodesicActiveContour.GetElapsedIterations() )
        # contour = sitk.BinaryContour( sitk.BinaryThreshold( levelset, -1000, 0 ) )
        # # sitk.Show(sitk.LabelOverlay(simg, contour), "Levelset Countour")
        # sitk.Show(contour, "Levelset Countour")


