import vtk
import os
import numpy as np
import tifffile as tf

def load_tif_sequence(directory):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')])
    image_list = [tf.imread(file) for file in files]
    return np.stack(image_list, axis=0)

def main():
    data_dir = 'C:/Users/mmabo/Downloads/Image_reconstruction/'
    volume_data = load_tif_sequence(data_dir)

    # Compute the range of data intensities
    min_intensity = volume_data.min()
    max_intensity = volume_data.max()

    # Convert numpy array to VTK format
    data_importer = vtk.vtkImageImport()
    data_importer.CopyImportVoidPointer(volume_data, volume_data.nbytes)
    data_importer.SetDataScalarTypeToUnsignedShort()
    data_importer.SetNumberOfScalarComponents(1)

    # Set the data dimensions
    w, h, d = volume_data.shape
    data_importer.SetDataExtent(0, d-1, 0, h-1, 0, w-1)
    data_importer.SetWholeExtent(0, d-1, 0, h-1, 0, w-1)

    # Define opacity
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(min_intensity, 0.0)
    alphaChannelFunc.AddPoint(max_intensity, 0.2)

    # Define color map
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(min_intensity, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(max_intensity, 1.0, 1.0, 1.0)

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    # Define the volume mapper using GPU ray casting
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(data_importer.GetOutputPort())

    # Define the volume (final visualization object)
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # Renderer and render window
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    # Connect render window to interactor (for mouse & keyboard interactions)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)
    renderWin.SetSize(400, 400)

    # Initialize and start the interactor
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()

if __name__ == '__main__':
    main()
