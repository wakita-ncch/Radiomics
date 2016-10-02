import vtk
import itk
import dicom
from multiprocessing import Pool
import re
import os
import csv
from scipy import interpolate
import matplotlib.pyplot as plt

import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _point_in_polygon import _point_in_polygon
from _make_masked_image import _make_masked_image

from profiling_tools import *

def makeCoordinationDictionary(ds, structure_id):
    
    structure = ds.ROIContourSequence[structure_id]

    coord_dict = {}
        
    for i, plane in enumerate(structure.ContourSequence):
        
        if not plane.ContourData[2] in coord_dict:
            
            contours = []
            a_contour = []
            
            for j in range(plane.NumberOfContourPoints):
                
                point = ()
                point = (float(plane.ContourData[3*j]), float(plane.ContourData[3*j+1]), float(plane.ContourData[3*j+2]))
                a_contour.append(point)
                
            contours.append(a_contour)
            coord_dict[plane.ContourData[2]] = contours
            
        elif plane.ContourData[2] in coord_dict:
            
            a_contour = []
                
            for j in range(plane.NumberOfContourPoints):
                
                point = ()
                point = (float(plane.ContourData[3*j]), float(plane.ContourData[3*j+1]), float(plane.ContourData[3*j+2]))
                a_contour.append(point)
                
            coord_dict[plane.ContourData[2]].append(a_contour)

    return coord_dict

@time
def makeTriangulatedMesh(coordination_dictionary, division_spacing, sigma, num_threads):

    max_X = -1e10
    min_X = 1e10
    max_Y = -1e10
    min_Y = 1e10
    
    for key in sorted(coordination_dictionary.keys()):
        
        plane = coordination_dictionary[key]
        
        for a_contour in plane:
            
            for point in a_contour:
                
                if point[0] > max_X:
                    max_X = point[0]
                if point[0] < min_X:
                    min_X = point[0]
                if point[1] > max_Y:
                    max_Y = point[1]
                if point[1] < min_Y:
                    min_Y = point[1]
                    
    max_X = int(max_X) + 1
    min_X = int(min_X) - 1
    max_Y = int(max_Y) + 1
    min_Y = int(min_Y) - 1
    
    image_X = max_X - min_X + 10
    image_Y = max_Y - min_Y + 10
    
    args = []
    min_Z = min(coordination_dictionary.keys())
    
    for key in sorted(coordination_dictionary.keys()):
        
        plane = coordination_dictionary[key]
        
        for a_contour in plane:
            
            image = np.zeros((image_X, image_Y), dtype = float)
            
            arg = (a_contour, image, min_X, min_Y, key)
            
            args.append(arg)

    p = Pool(int(num_threads))

    imagesWithKey = p.map(makeBinaryImageWighTag, [a for a in args])

    imageDict = {}
    
    for imageWithKey in imagesWithKey:
        
        if not imageWithKey[1] in imageDict.keys():
            
            imageDict[imageWithKey[1]] = imageWithKey[0]
            
        else:
            
            imageDict[imageWithKey[1]] += imageWithKey[0]
            
    spacer_image = np.zeros((image_X, image_Y, 5), dtype = float)
    binary_image = spacer_image
    
    keyList = sorted(imageDict.keys())
    
    for i, key in enumerate(keyList):
        
        a_image = imageDict[key]
        
        try:
            
            nextKey = keyList[i+1]
            
            for j in range(int(float(nextKey) - float(key))):
                                        
                binary_image = np.dstack((binary_image, a_image))
                
        except:
            
            binary_image = np.dstack((binary_image, a_image))
        
    binary_image = np.dstack((binary_image, spacer_image))

    itkImage = numpyToITK3D(binary_image)
    
    CharPixelType = itk.UC
    RealPixelType = itk.F
    Dimension = 3
    
    CharImageType = itk.Image[CharPixelType, Dimension]
    RealImageType = itk.Image[RealPixelType, Dimension]
    
    smoothFilter = itk.SmoothingRecursiveGaussianImageFilter[RealImageType, RealImageType]
    smoothFilter = smoothFilter.New()
    smoothFilter.SetInput(itkImage)
    smoothFilter.SetSigma(float(sigma))
    smoothFilter.Update()
    
    rescaleFilter = itk.RescaleIntensityImageFilter[RealImageType, CharImageType]
    rescale = rescaleFilter.New()
    rescale.SetOutputMinimum(0)
    rescale.SetOutputMaximum(255)    
    rescale.SetInput(smoothFilter.GetOutput())    
    rescale.Update()
    
    converter = itk.ImageToVTKImageFilter[CharImageType].New()
    converter.SetInput(rescale.GetOutput())
    converter.Update()
    
    surface = vtk.vtkMarchingCubes()
    surface.SetInput(converter.GetOutput())
    surface.ComputeNormalsOn()
    surface.SetValue(0, 80.0)
    
    transform = vtk.vtkTransform()
    transform.Translate(min_X - 5, min_Y - 5, min_Z - 5)
    
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(surface.GetOutputPort())
    tf.SetTransform(transform)
    tf.Update()
    
    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputConnection(tf.GetOutputPort())
    triangleFilter.Update()
    
    decimate = vtk.vtkQuadricClustering()
    decimate.SetDivisionSpacing(float(division_spacing), float(division_spacing), float(division_spacing))
    decimate.SetInput(triangleFilter.GetOutput())
    
    return decimate.GetOutput()

def makeBinaryImageWighTag(arg):

    poly = arg[0]
    image = arg[1]
    minX = arg[2]
    minY = arg[3]
    tag = arg[4]

    for xs in range(0, image.shape[0]):
        for ys in range(0, image.shape[1]):
            image[xs][ys] = c_PointInPolygon(xs + minX - 5, ys + minY - 5, poly)
            
    return [image, tag]

def numpyToITK3D(numpyArray):
   
    itkImage = makeITKImage(numpyArray.shape, itk.F)

    index = []
    for i in range(len(numpyArray.shape)):
        index.append(None)

    for i in range(numpyArray.shape[0]):
        for j in range(numpyArray.shape[1]):
            for k in range(numpyArray.shape[2]):
                index[0] = i
                index[1] = j
                index[2] = k
                itkImage.SetPixel(index, numpyArray[i,j,k])

    return itkImage

def PointInPolygon(x, y, poly):

    n = len(poly)
    inside = False
    p1x, p1y, p1z = poly[0]
    for i in range(n+1):
        p2x, p2y, p2z = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside              
        p1x,p1y = p2x,p2y
        
    return inside

def c_PointInPolygon(x, y, poly):

    #poly = np.array(poly)

    return False if _point_in_polygon(x, y, poly) < 0 else True

def makeITKImage(dimensions, pixelType):
    
    Dimension = len(dimensions)
    ImageType = itk.Image[pixelType, Dimension]
    
    index = itk.Index[Dimension]()
    for i in range(Dimension):
        index.SetElement(i, 0)
    
    size = itk.Size[Dimension]()
    for i in range(Dimension):
        size.SetElement(i, dimensions[i])
    
    imageRegion = itk.ImageRegion[Dimension]()
    imageRegion.SetSize(size)
    imageRegion.SetIndex(index)
    
    image = ImageType.New()
    
    image.SetRegions(imageRegion)
    image.Allocate()
    image.FillBuffer(0)
    
    return image

### Functions for vtk ###

def makeActor(polyData, color):
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polyData)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    
    return actor   

def showPolydata(polydata1, polydata2 = None):

    actor = makeActor(polydata1, (1, 0, 0))
    actor.GetProperty().SetOpacity(1.0)

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    ren.SetBackground(1, 1, 1)

    ren.AddActor(actor)

    if polydata2:

        actor2 = makeActor(polydata2, (0, 1, 0))
        actor2.GetProperty().SetOpacity(0.5)

        ren.AddActor(actor2)

    iren = vtk.vtkRenderWindowInteractor()    
    iren.SetRenderWindow(renWin)
                
    iren.Initialize()
    renWin.Render()
    iren.Start()

class StructureLoader(object):

    def __init__(self, struct_dcm, struct_id):

        self.coord_dict = None
        self.surface = None

        _division_spacing = 2.0
        _sigma = 1.0
        _num_threads = 4

        self.coord_dict = makeCoordinationDictionary(struct_dcm, struct_id)

        if False:

            self.surface = makeTriangulatedMesh(self.coord_dict, _division_spacing, _sigma, _num_threads)
            showPolydata(self.surface)

    def getZRange(self):

        return self.coord_dict.keys()

#class ImageLoader(object):

#    def __init__(self, image_path):

#        self.reader = vtk.vtkDICOMImageReader()
#        self.reader.SetDirectoryName(image_path)
#        self.reader.Update()

#        self.image = self.reader.GetOutput()

#        self.showImage(self.image)

#    def showImage(self, polydata = None):

#        viewer = vtk.vtkImageViewer2()

#        viewer.SetInput(self.image)
#        #viewer.SetColorWindow(1500.0)
#        #viewer.SetColorLevel(-500)

#        istyle = vtk.vtkInteractorStyleImage()
#        iren = vtk.vtkRenderWindowInteractor()
#        viewer.SetupInteractor(iren)
#        iren.SetInteractorStyle(istyle)

#        minSlice = viewer.GetSliceMin()
#        maxSlice = viewer.GetSliceMax()
#        sliceNum = {'slice' : minSlice}

#        def MouseWheelForwardEvent(caller, event):
        
#            if sliceNum['slice'] < maxSlice :
#                sliceNum['slice'] += 1
#                viewer.SetSlice(sliceNum['slice'])

#                viewer.Render()
        
#        def MouseWheelBackwardEvent(caller, event):
       
#            if sliceNum['slice'] > minSlice :
#                sliceNum['slice'] -= 1
#                viewer.SetSlice(sliceNum['slice'])

#                viewer.Render()

#        istyle.AddObserver("MouseWheelForwardEvent", MouseWheelForwardEvent)
#        istyle.AddObserver("MouseWheelBackwardEvent", MouseWheelBackwardEvent)

#        viewer.Render()
#        iren.Initialize()
#        iren.Start()

def getDicomFiles(path):
    
    files = os.listdir(path)
    dcm_files = []

    for i, file in enumerate(files):

        root, ext = os.path.splitext(path + '/' + file)

        if ext == '.dcm':

            dcm_files.append(path + '/' + file)

    return dcm_files

class ImageLoader(object):

    def __init__(self, dcm_dir_path):

        self.sorted_images = None
        self.images = None
        self.imagePositionPatientX = None
        self.imagePositionPatientY = None
        self.pixelSpacingX = None
        self.pixelSpacingY = None
        self.sliceThickness = None
        self.rows = None
        self.columns = None

        # sort images according to z slice location

        unsorted_images = getDicomFiles(dcm_dir_path)
        unsorted_positions = []
        images = {}

        for image in unsorted_images:

            dcm = dicom.read_file(image)

            if not (self.imagePositionPatientX and self.imagePositionPatientY and self.pixelSpacingX and self.pixelSpacingY 
                    and self.sliceThickness and self.rows and self.columns):

                self.imagePositionPatientX = dcm.ImagePositionPatient[0]
                self.imagePositionPatientY = dcm.ImagePositionPatient[1]
                self.pixelSpacingX = dcm.PixelSpacing[0]
                self.pixelSpacingY = dcm.PixelSpacing[1]
                self.sliceThickness = dcm.SliceThickness
                self.rows = dcm.Rows
                self.columns = dcm.Columns

            assert dcm.SliceLocation == dcm.ImagePositionPatient[2]

            unsorted_positions.append(dcm.ImagePositionPatient[2])

            images[dcm.ImagePositionPatient[2]] = dcm.pixel_array

        assert len(unsorted_positions) == len(set(unsorted_positions))

        unsorted_positions = np.array(unsorted_positions)

        sorted_images = [None] * len(unsorted_images)

        for i, val in enumerate(np.argsort(unsorted_positions)):

            sorted_images[i] = unsorted_images[val]

        self.sorted_images = sorted_images
        self.images = images

@time
def makeMaskedImage(image, contours, imageLoader):

    assert len(contours) == 1

    #val_max = np.max(image)
    #val_min = np.min(image)

    #for contour in contours:

    #    for i, point in enumerate(contour):

    #        (X, Y, Z) = point

    #        x = int((X - imageLoader.imagePositionPatientX) / imageLoader.pixelSpacingX)
    #        y = int((Y - imageLoader.imagePositionPatientY) / imageLoader.pixelSpacingY)
    #        z = int(Z)

    #        if i > 0:

    #            (X1, Y1, Z1) = contour[i-1]

    #            x1 = int((X1 - imageLoader.imagePositionPatientX) / imageLoader.pixelSpacingX)
    #            y1 = int((Y1 - imageLoader.imagePositionPatientY) / imageLoader.pixelSpacingY)
    #            z1 = int(Z1)         

    #            plt.plot([x, x1], [y, y1], 'r-')

    #masked_image = np.copy(image)

    #ys = np.arange(masked_image.shape[0])
    #xs = np.arange(masked_image.shape[1])

    for y in range(image.shape[0]):

        for x in range(image.shape[1]):

            X = x * imageLoader.pixelSpacingX + imageLoader.imagePositionPatientX
            Y = y * imageLoader.pixelSpacingY + imageLoader.imagePositionPatientY

            for contour in contours:

                if not c_PointInPolygon(X, Y, contour):

                    image[y, x] = 0

    return image

@time
def c_makeMaskedImage(image, contours, shape0, shape1, psX, psY, ippX, ippY):

    return _make_masked_image(image, contours, shape0, shape1, psX, psY, ippX, ippY)

if __name__ == '__main__':

    dir_path = 'C:\Users\Kaz\Desktop\ST'

    struct_dcm = dicom.read_file(dir_path + '\struct.dcm')

    for i, structure in enumerate(struct_dcm.StructureSetROISequence):

        name = structure.ROIName

        print i, name

    selected = int(raw_input("Select the structure for mesh creation: "))

    structureLoader = StructureLoader(struct_dcm, selected)

    imageLoader = ImageLoader('./data/fusedMRI')

    for slice_num in structureLoader.getZRange():

        print "slice num: ", slice_num

        image = imageLoader.images[slice_num]
        contours = structureLoader.coord_dict[slice_num]

        #masked_image = makeMaskedImage(image, contours, imageLoader)
        masked_image = c_makeMaskedImage(image, contours, image.shape[0], image.shape[1], 
                                         imageLoader.pixelSpacingX, imageLoader.pixelSpacingY, 
                                         imageLoader.imagePositionPatientX, imageLoader.imagePositionPatientY)

        #plt.close()
        #plt.gray()
        #plt.imshow(masked_image)
        #plt.show()

        stacked_image = np.vstack((stacked_image, masked_image)) if 'stacked_image' in locals() else masked_image

    print stacked_image