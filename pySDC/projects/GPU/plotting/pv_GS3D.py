# state file generated using paraview version 5.13.2
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [801, 744]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [0.5, 0.5, 0.5]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [2.431851652578137, 2.4318516525781364, 2.4318516525781373]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.5]
renderView1.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.8660254037844386
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.PolarGrid = 'Polar Grid Actor'
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [801, 744]
renderView2.AxesGrid = 'Grid Axes 3D Actor'
renderView2.CenterOfRotation = [0.75, 0.75, 0.25]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [1.1101682224564904, 1.1101682224564902, 0.6101682224564906]
renderView2.CameraFocalPoint = [0.1442423961674227, 0.1442423961674228, -0.35575760383257704]
renderView2.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 0.4330127018922193
renderView2.LegendGrid = 'Legend Grid Actor'
renderView2.PolarGrid = 'Polar Grid Actor'
renderView2.UseColorPaletteForBackground = 0
renderView2.Background = [1.0, 1.0, 1.0]
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitHorizontal(0, 0.500000)
layout1.AssignView(1, renderView1)
layout1.AssignView(2, renderView2)
layout1.SetSize(1603, 744)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Image Data Reader'
idx = 80
path = f'/p/project1/ccstma/baumann7/pySDC/pySDC/projects/GPU/vtk_data/GrayScottLarge-res_2304_{idx:06d}.vti'
print(f'Plotting {path}', flush=True)
data = XMLImageDataReader(registrationName='data.vti', FileName=[path])
data.CellArrayStatus = ['values']
data.TimeArray = 'None'

# create a new 'Extract Subset'
extractSubset1 = ExtractSubset(registrationName='ExtractSubset1', Input=data)
extractSubset1.VOI = [180, 360, 180, 360, 0, 180]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from grayScottLargeres_2304_000040vti
display = Show(data, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'values'
valuesTF2D = GetTransferFunction2D('values')
valuesTF2D.ScalarRangeInitialized = 1
valuesTF2D.Range = [-4.714955210161051e-05, 0.42375012117331917, 0.0, 1.0]

# get color transfer function/color map for 'values'
valuesLUT = GetColorTransferFunction('values')
valuesLUT.TransferFunction2D = valuesTF2D
valuesLUT.RGBPoints = [-4.714955210161051e-05, 0.231373, 0.298039, 0.752941, 0.21559500151153457, 0.865003, 0.865003, 0.865003, 0.43123715257517076, 0.705882, 0.0156863, 0.14902]
valuesLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'values'
valuesPWF = GetOpacityTransferFunction('values')
valuesPWF.Points = [-4.714955210161051e-05, 0.0, 0.5, 0.0, 0.3092452883720398, 0.008928571827709675, 0.5, 0.0, 0.37578630447387695, 0.6383928656578064, 0.5, 0.0, 0.43123715257517076, 1.0, 0.5, 0.0]
valuesPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
display.Representation = 'Volume'
display.ColorArrayName = ['CELLS', 'values']
display.LookupTable = valuesLUT
display.SelectNormalArray = 'None'
display.SelectTangentArray = 'None'
display.SelectTCoordArray = 'None'
display.TextureTransform = 'Transform2'
display.OSPRayScaleFunction = 'Piecewise Function'
display.Assembly = ''
display.SelectedBlockSelectors = ['']
display.SelectOrientationVectors = 'None'
display.ScaleFactor = 0.1
display.SelectScaleArray = 'values'
display.GlyphType = 'Arrow'
display.GlyphTableIndexArray = 'values'
display.GaussianRadius = 0.005
display.SetScaleArray = [None, '']
display.ScaleTransferFunction = 'Piecewise Function'
display.OpacityArray = [None, '']
display.OpacityTransferFunction = 'Piecewise Function'
display.DataAxesGrid = 'Grid Axes Representation'
display.PolarAxes = 'Polar Axes Representation'
display.ScalarOpacityUnitDistance = 0.00481125224324688
display.ScalarOpacityFunction = valuesPWF
display.TransferFunction2D = valuesTF2D
display.OpacityArrayName = ['CELLS', 'values']
display.ColorArray2Name = ['CELLS', 'values']
display.IsosurfaceValues = [0.21559500151153457]
display.SliceFunction = 'Plane'
display.Slice = 180
display.SelectInputVectors = [None, '']
display.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
display.SliceFunction.Origin = [0.5, 0.5, 0.5]

# show data from extractSubset1
extractSubset1Display = Show(extractSubset1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
extractSubset1Display.Representation = 'Surface'
extractSubset1Display.ColorArrayName = ['CELLS', 'values']
extractSubset1Display.LookupTable = valuesLUT
extractSubset1Display.SelectNormalArray = 'None'
extractSubset1Display.SelectTangentArray = 'None'
extractSubset1Display.SelectTCoordArray = 'None'
extractSubset1Display.TextureTransform = 'Transform2'
extractSubset1Display.OSPRayScaleFunction = 'Piecewise Function'
extractSubset1Display.Assembly = ''
extractSubset1Display.SelectedBlockSelectors = ['']
extractSubset1Display.SelectOrientationVectors = 'None'
extractSubset1Display.ScaleFactor = 0.05
extractSubset1Display.SelectScaleArray = 'values'
extractSubset1Display.GlyphType = 'Arrow'
extractSubset1Display.GlyphTableIndexArray = 'values'
extractSubset1Display.GaussianRadius = 0.0025
extractSubset1Display.SetScaleArray = [None, '']
extractSubset1Display.ScaleTransferFunction = 'Piecewise Function'
extractSubset1Display.OpacityArray = [None, '']
extractSubset1Display.OpacityTransferFunction = 'Piecewise Function'
extractSubset1Display.DataAxesGrid = 'Grid Axes Representation'
extractSubset1Display.PolarAxes = 'Polar Axes Representation'
extractSubset1Display.ScalarOpacityUnitDistance = 0.00481125224324688
extractSubset1Display.ScalarOpacityFunction = valuesPWF
extractSubset1Display.TransferFunction2D = valuesTF2D
extractSubset1Display.OpacityArrayName = ['CELLS', 'values']
extractSubset1Display.ColorArray2Name = ['CELLS', 'values']
extractSubset1Display.IsosurfaceValues = [0.21185148581060878]
extractSubset1Display.SliceFunction = 'Plane'
extractSubset1Display.Slice = 90
extractSubset1Display.SelectInputVectors = [None, '']
extractSubset1Display.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
extractSubset1Display.SliceFunction.Origin = [0.75, 0.75, 0.75]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from extractSubset1
extractSubset1Display_1 = Show(extractSubset1, renderView2, 'UniformGridRepresentation')

# trace defaults for the display properties.
extractSubset1Display_1.Representation = 'Volume'
extractSubset1Display_1.ColorArrayName = ['CELLS', 'values']
extractSubset1Display_1.LookupTable = valuesLUT
extractSubset1Display_1.SelectNormalArray = 'None'
extractSubset1Display_1.SelectTangentArray = 'None'
extractSubset1Display_1.SelectTCoordArray = 'None'
extractSubset1Display_1.TextureTransform = 'Transform2'
extractSubset1Display_1.OSPRayScaleFunction = 'Piecewise Function'
extractSubset1Display_1.Assembly = ''
extractSubset1Display_1.SelectedBlockSelectors = ['']
extractSubset1Display_1.SelectOrientationVectors = 'None'
extractSubset1Display_1.ScaleFactor = 0.1
extractSubset1Display_1.SelectScaleArray = 'values'
extractSubset1Display_1.GlyphType = 'Arrow'
extractSubset1Display_1.GlyphTableIndexArray = 'values'
extractSubset1Display_1.GaussianRadius = 0.005
extractSubset1Display_1.SetScaleArray = [None, '']
extractSubset1Display_1.ScaleTransferFunction = 'Piecewise Function'
extractSubset1Display_1.OpacityArray = [None, '']
extractSubset1Display_1.OpacityTransferFunction = 'Piecewise Function'
extractSubset1Display_1.DataAxesGrid = 'Grid Axes Representation'
extractSubset1Display_1.PolarAxes = 'Polar Axes Representation'
extractSubset1Display_1.ScalarOpacityUnitDistance = 0.00481125224324688
extractSubset1Display_1.ScalarOpacityFunction = valuesPWF
extractSubset1Display_1.TransferFunction2D = valuesTF2D
extractSubset1Display_1.OpacityArrayName = ['CELLS', 'values']
extractSubset1Display_1.ColorArray2Name = ['CELLS', 'values']
extractSubset1Display_1.IsosurfaceValues = [0.21559500151153457]
extractSubset1Display_1.SliceFunction = 'Plane'
extractSubset1Display_1.Slice = 180
extractSubset1Display_1.SelectInputVectors = [None, '']
extractSubset1Display_1.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
extractSubset1Display_1.SliceFunction.Origin = [0.5, 0.5, 0.5]

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = [renderView1, renderView2]
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

# ----------------------------------------------------------------
# restore active source
SetActiveSource(extractSubset1)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
save_path = f"plots/GS_large/GS3D_{idx:06d}.png"
SaveScreenshot(save_path, GetLayout())
print(f'Saved {save_path}', flush=True)
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------
