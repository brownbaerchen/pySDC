# state file generated using paraview version 5.13.2
import paraview

paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import (
    GetMaterialLibrary,
    CreateView,
    SetActiveView,
    CreateLayout,
    XMLImageDataReader,
    ExtractSubset,
    Show,
    GetTransferFunction2D,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetTimeTrack,
    GetTimeKeeper,
    SetActiveSource,
    GetAnimationScene,
    SaveScreenshot,
    GetLayout,
)

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1856, 1104]
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
renderView2.ViewSize = [1856, 1104]
renderView2.AxesGrid = 'Grid Axes 3D Actor'
renderView2.CenterOfRotation = [0.390625, 0.4782986111111111, 0.5477430555555556]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [0.5026408771490182, 0.5903144882601291, 0.6597589327045738]
renderView2.CameraFocalPoint = [0.32117817260657244, 0.4088517837176832, 0.4782962281621271]
renderView2.CameraViewUp = [-0.4082482904638631, 0.816496580927726, -0.40824829046386296]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 0.08134750500301806
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
# path = f'/p/project1/ccstma/baumann7/pySDC/pySDC/projects/GPU/vtk_data/GrayScottLarge-res_2304_{args.idx:06d}.vti'
path = f'/p/scratch/ccstma/baumann7/GS3D_vti/GrayScottLarge-res_2304_{args.idx:06d}.vti'
print(f'Plotting {path}', flush=True)
data = XMLImageDataReader(registrationName='data.vti', FileName=[path])
data.CellArrayStatus = ['values']
data.TimeArray = 'None'

# create a new 'Extract Subset'
extractSubset1 = ExtractSubset(registrationName='ExtractSubset1', Input=data)
# extractSubset1.VOI = [180, 360, 180, 360, 0, 180]
extractSubset1.VOI = [800, 1100, 924, 1200, 1124, 1420]

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
valuesLUT.RGBPoints = [
    -0.00010914247462833595,
    0.0,
    0.0,
    0.34902,
    0.01387773478635173,
    0.039216,
    0.062745,
    0.380392,
    0.027864612047331795,
    0.062745,
    0.117647,
    0.411765,
    0.04185148930831186,
    0.090196,
    0.184314,
    0.45098,
    0.055838366569291926,
    0.12549,
    0.262745,
    0.501961,
    0.069825243830272,
    0.160784,
    0.337255,
    0.541176,
    0.08381212109125206,
    0.2,
    0.396078,
    0.568627,
    0.09779899835223213,
    0.239216,
    0.454902,
    0.6,
    0.1117858756132122,
    0.286275,
    0.521569,
    0.65098,
    0.12577275287419226,
    0.337255,
    0.592157,
    0.701961,
    0.1397596301351723,
    0.388235,
    0.654902,
    0.74902,
    0.1537465073961524,
    0.466667,
    0.737255,
    0.819608,
    0.16773338465713244,
    0.572549,
    0.819608,
    0.878431,
    0.18172026191811252,
    0.654902,
    0.866667,
    0.909804,
    0.19570713917909258,
    0.752941,
    0.917647,
    0.941176,
    0.20969401644007266,
    0.823529,
    0.956863,
    0.968627,
    0.2236808937010527,
    0.941176,
    0.984314,
    0.988235,
    0.2236808937010527,
    0.988235,
    0.960784,
    0.901961,
    0.23263249514807996,
    0.988235,
    0.945098,
    0.85098,
    0.2415840965951072,
    0.980392,
    0.898039,
    0.784314,
    0.25165464822301287,
    0.968627,
    0.835294,
    0.698039,
    0.2656415254839929,
    0.94902,
    0.733333,
    0.588235,
    0.27962840274497297,
    0.929412,
    0.65098,
    0.509804,
    0.29361528000595305,
    0.909804,
    0.564706,
    0.435294,
    0.30760215726693313,
    0.878431,
    0.458824,
    0.352941,
    0.32158903452791315,
    0.839216,
    0.388235,
    0.286275,
    0.33557591178889323,
    0.760784,
    0.294118,
    0.211765,
    0.3495627890498733,
    0.701961,
    0.211765,
    0.168627,
    0.3635496663108534,
    0.65098,
    0.156863,
    0.129412,
    0.3775365435718334,
    0.6,
    0.094118,
    0.094118,
    0.3915234208328135,
    0.54902,
    0.066667,
    0.098039,
    0.4055102980937936,
    0.501961,
    0.05098,
    0.12549,
    0.41949717535477365,
    0.45098,
    0.054902,
    0.172549,
    0.4334840526157537,
    0.4,
    0.054902,
    0.192157,
    0.44747092987673376,
    0.34902,
    0.070588,
    0.211765,
]
valuesLUT.ColorSpace = 'Lab'
valuesLUT.NanColor = [0.25, 0.0, 0.0]
valuesLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'values'
valuesPWF = GetOpacityTransferFunction('values')
valuesPWF.Points = [
    -0.00010914247462833595,
    0.0,
    0.5,
    0.0,
    0.21217168867588043,
    0.0,
    0.5,
    0.0,
    0.3758581280708313,
    0.9,
    0.5,
    0.0,
    0.44747092987673376,
    0.95,
    0.5,
    0.0,
]

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
save_path = f"/p/project1/ccstma/baumann7/pySDC/pySDC/projects/GPU/plots/GS_large/GS3D_{args.idx:06d}.png"
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
