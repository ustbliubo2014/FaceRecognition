import os
import string
from javax.vecmath import Color3f, Point3f
from ij.io import OpenDialog
from loci.plugins import BF
from ij import IJ, ImagePlus, ImageStack
from ij.process import ImageConverter
from ij3d import Image3DUniverse
from script.imglib import ImgLib

#Static parameters
USE_POINT=1
USE_ICOSPHERE=2
USE_ICOSPHERE_WITH_LIMIT=3 #Plot all points, may be slow over 3495
LIMIT_SPHERE_PER_MESH=3495 #Limit of number of icospheres per Mesh


#Use point(1), icosphere(2), or icosphere but limited to the first 3495 points in X (3)
plotType=USE_POINT
# diameter Cell visualization
cellIcosophere=3.0

#Open the image and duplicate it to get an 8 Bits version use to plot
od = OpenDialog("Open coordinate file", None)
srcFile = od.getPath()
fp = open(srcFile)

points=[]
try:
  for line in fp:
  	result=string.split(line,",")
  	for coor in result:
  	  point = Point3f(float(result[0]),float(result[1]),float(result[2]))
  	  #print "add point  : %f %f  %f" % (float(result[0]),float(result[1]),float(result[2]))
  	  points.append(point)
  	  #print result[0],
finally:
  fp.close()




#Open the image and duplicate it to get an 8 Bits version use to plot
od = OpenDialog("Open image file", None)
srcFile = od.getPath()
srcDir = od.getDirectory()
imp = BF.openImagePlus(srcFile)[0]
#imp8Bits = imp.duplicate()
ImageConverter(imp).convertToGray8()

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
#univ.addOrthoslice(imp).setLocked(True)

#Add the stack and display it in 3D
univ.addVoltex(imp).setLocked(True)

#Display the 3D View
univ.show()
print "Done"  
#List<Object>