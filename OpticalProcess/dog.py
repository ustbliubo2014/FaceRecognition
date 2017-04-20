#See http://javadoc.imagej.net/Fiji/index.html for import
from __future__ import division #So dividing 2 int will give a float if needed
import os
from loci.plugins import BF
from ij.io import OpenDialog
from ij import IJ, ImagePlus, ImageStack
from ij.process import ImageConverter, AutoThresholder
from ij3d import Image3DUniverse
from javax.vecmath import Color3f, Point3f
from script.imglib.analysis import DoGPeaks
from script.imglib import ImgLib
#from script.imglib.color import Red

#This script will open a fluorescent 1 channel stack, compute the difference of Gaussian and then plot all found points
# with the orthoslice view
#Parameters available :
# - cellDiameter : Select the cell/nucleus diameter (in unut of the Image Stack)
# - minPeak : The min intensity for a peak to be considered
# - plotType : plot Points, Icospheres or limit to the first 3495 points due to some slowdown with Icospheres
# - cellIcosophere : The size of the point/icosphere use to visualize each detected point.
#
#TODO: 
# -Apply 3D Watershed segmentation using the detected points has seed
#  (see http://imagejdocu.tudor.lu/doku.php?id=plugin:segmentation:3d_spots_segmentation:start#d_watershed )
# -Allow multiple Channels in the stack
# -Add a dialog for the parameters
# -Add interface to measure nucleus/cell parameters
#
#Authors : Benjamin Pavie & Radoslaw Ejsmont (aka Radek)

#Static parameters
USE_POINT=1
USE_ICOSPHERE=2
USE_ICOSPHERE_WITH_LIMIT=3 #Plot all points, may be slow over 3495
LIMIT_SPHERE_PER_MESH=3495 #Limit of number of icospheres per Mesh

#############################
# USER PARAMETERS TO CHANGE
#############################
#Diameter in microns
cellDiameter=3.0
# The minimum intensity for a peak to be considered
minPeak=0
#Use point(1), icosphere(2), or icosphere but limited to the first 3495 points in X (3)
plotType=USE_POINT
# diameter Cell visualization
cellIcosophere=3.0

#Open the image and duplicate it to get an 8 Bits version use to plot
od = OpenDialog("Open image file", None)
srcFile = od.getPath()
srcDir = od.getDirectory()
imp = BF.openImagePlus(srcFile)[0]
imp8Bits = imp.duplicate()
ImageConverter(imp8Bits).convertToGray8()
print "Image " + srcFile + " open successfully!"
#Create the coordinate.txt file to ouput the point coordinate
coordinateFile = open(srcDir + "coordinate.txt", "w")
print "Created " +srcDir + "coordinate.txt"
# If minPeak is set to 0, set it to automatically.
if minPeak == 0:
	minPeak = AutoThresholder().getThreshold("Percentile", imp.getStatistics().histogram); 

#Get the pixel calibration
cal=imp.getCalibration()
#Set Gaussian Sigma parameters for the Difference of Gaussian
sigmaLarge=[cellDiameter/cal.pixelWidth,cellDiameter/cal.pixelHeight,cellDiameter/cal.pixelDepth]
sigmaSmall=[a/2 for a in sigmaLarge]
print "Cell Diameter: XY-%f Z-%f in pixel" % (cellDiameter/cal.pixelWidth, cellDiameter/cal.pixelDepth)
print "Minimum peak value: %f" % (minPeak)
print "Sigma Large  : %f %f  %f in pixel" % (cellDiameter/cal.pixelWidth, cellDiameter/cal.pixelHeight,cellDiameter/cal.pixelDepth)
print "Sigma Small  : %f %f  %f in pixel" % (cellDiameter/cal.pixelWidth/2, cellDiameter/cal.pixelHeight/2,cellDiameter/cal.pixelDepth/2)
#peaks=DoGPeaks(Red(ImgLib.wrap(imp)),sigmaLarge,sigmaSmall,minPeak,1)
#Difference of Gaussians
peaks=DoGPeaks(ImgLib.wrap(imp),sigmaLarge,sigmaSmall,minPeak,1)
print "Found", len(peaks), "peaks"


#########################################################
# Show the peaks as spheres in 3D, along with orthoslices
#########################################################

#Get point list from the DoGPeaks rescaled according to the pixel size
points = peaks.asPoints([cal.pixelWidth,cal.pixelHeight,cal.pixelDepth])

#Sort the coordinate on x axis
points = sorted(points, key=lambda point: point.x)

#Write all points in a text file
for point in points:
	coordinate = "%f, %f, %f \n" % (point.x, point.y, point.z)
	coordinateFile.write(coordinate)
coordinateFile.close()

#Create the Image3D	
univ=Image3DUniverse(512,512)

#Plot all points as Point , Icosphere or if too many points (>3495), only the first 3495
if plotType == USE_POINT:
	univ.addPointMesh(points[0:len(points)-1],Color3f(1,0,0),cellIcosophere, 'Cells').setLocked(True)
elif plotType == USE_ICOSPHERE:
	#Plot all points, splitted if needed in multiple mesh containing each a maximum of 3495 Icosphere
	a, b= divmod(len(peaks), LIMIT_SPHERE_PER_MESH)
	start=0
	for i in range(0,a):
		univ.addIcospheres(points[start:start+LIMIT_SPHERE_PER_MESH],Color3f(1,0,0),2,cellIcosophere/2, 'Cells'+str(i)).setLocked(True)
		print "Plot point %f to %f" % (start, start+LIMIT_SPHERE_PER_MESH)
		start=start+LIMIT_SPHERE_PER_MESH
	univ.addIcospheres(points[start:len(points)-1],Color3f(1,0,0),2,cellIcosophere/2, 'Cells'+str(a)).setLocked(True)  
	print "Plot point %f to %f" % (start+1, len(points))
elif plotType == USE_ICOSPHERE_WITH_LIMIT:
	if len(points) < LIMIT_SPHERE_PER_MESH:
		univ.addIcospheres(points[0:len(points)-1],Color3f(1,0,0),2,cellIcosophere/2, 'Cells').setLocked(True)
	else:
		univ.addIcospheres(points[0:LIMIT_SPHERE_PER_MESH-1],Color3f(1,0,0),2,cellIcosophere/2, 'Cells').setLocked(True)
		print "Plot only the first", LIMIT_SPHERE_PER_MESH, "points"
else: #Default is plot points
	univ.addPointMesh(points[0:len(points)-1],Color3f(1,0,0),6, 'Cells').setLocked(True)
	
#Add the orthoslice of the stack
#Orhoslices can be manipulated using the context menu (right click once the orthoslice has been selected then >Adjust Slices
univ.addOrthoslice(imp8Bits).setLocked(True)

#Add the stack and display it in 3D
#univ.addVoltex(imp8Bits).setLocked(True)

#Display the 3D View
univ.show()
print "Done"
