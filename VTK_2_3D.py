import vtk
import numpy as np
import tifffile
from vtk.util import numpy_support
def load_tif_sequence(path, num_slices):
    """Load tif images into a 3D numpy array."""
    slices = [tifffile.imread(f"{path}Z_Fish_{i:04d}.tif") for i in range(num_slices)]
    return np.stack(slices, axis=-1)

# Load your tif image sequence into a 3D numpy array
data_path = "C:/Users/mmabo/Downloads/Image_reconstruction/"
volume_np = load_tif_sequence(data_path, 113)  # Adjust the number 113 based on your data

# Convert numpy array to VTK format
data = vtk.vtkImageData()
data.SetDimensions(volume_np.shape)
flattened_volume = volume_np.ravel()
vtk_data_array = numpy_support.numpy_to_vtk(flattened_volume, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
data.GetPointData().SetScalars(vtk_data_array)

# Define opacity transfer function
alpha_channel_func = vtk.vtkPiecewiseFunction()
alpha_channel_func.AddPoint(0, 0.0)
alpha_channel_func.AddPoint(255, 0.8)

# Define color transfer function
color_func = vtk.vtkColorTransferFunction()
color_func.AddRGBPoint(0, 0, 0, 0)
color_func.AddRGBPoint(255, 1, 1, 1)

# Combine the two functions to describe how volume is rendered
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_func)
volume_property.SetScalarOpacity(alpha_channel_func)
volume_property.SetInterpolationTypeToLinear()
volume_property.ShadeOn()
volume_property.SetAmbient(0.4)

# Define the volume mapper
volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
volume_mapper.SetInputData(data)
volume_mapper.SetBlendModeToComposite()

# Connect everything to the volume renderer
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Set up the renderer
renderer = vtk.vtkRenderer()
render_win = vtk.vtkRenderWindow()
render_win.AddRenderer(renderer)
render_interactor = vtk.vtkRenderWindowInteractor()
render_interactor.SetRenderWindow(render_win)
renderer.AddVolume(volume)
renderer.SetBackground(0, 0, 0)
render_win.SetSize(800, 800)

# Initialize and start the rendering loop
render_interactor.Initialize()
render_win.Render()
render_interactor.Start()
