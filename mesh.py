#!/usr/bin/env python

# This example shows how to use Delaunay3D with alpha shapes.

import vtk
import numpy as np
Nx=4
Ny=Nx
Nz=Nx
L=1.
P=L
H=L
# beware the x,y slicing is not the same as in matlab
X, Y, Z = np.mgrid[0: L: Nx * 1.j,
                   0: P: Ny * 1.j,
                   0: H: Nz * 1.j]


# The points to be triangulated are generated randomly in the unit
# cube located at the origin. The points are then associated with a
# vtkPolyData.
math = vtk.vtkMath()
points = vtk.vtkPoints()
l = 0
for i in xrange(X.shape[0]):
  for j in xrange(X.shape[1]):
    for k in xrange(X.shape[2]):
      points.InsertPoint(l, X[i,j,k], Y[i,j,k], Z[i,j,k])
      l=l+1
profile = vtk.vtkPolyData()
profile.SetPoints(points)

# Delaunay3D is used to triangulate the points. The Tolerance is the
# distance that nearly coincident points are merged
# together. (Delaunay does better if points are well spaced.) The
# alpha value is the radius of circumcircles, circumspheres. Any mesh
# entity whose circumcircle is smaller than this value is output.
delny = vtk.vtkDelaunay3D()
delny.SetInput(profile)
delny.SetTolerance(0.01)
#delny.SetAlpha(0.2)
delny.BoundingTriangulationOff()

# Shrink the result to help see it better.
shrink = vtk.vtkShrinkFilter()
shrink.SetInputConnection(delny.GetOutputPort())
shrink.SetShrinkFactor(0.6)

mymap = vtk.vtkDataSetMapper()
mymap.SetInputConnection(shrink.GetOutputPort())

triangulation = vtk.vtkActor()
triangulation.SetMapper(mymap)
triangulation.GetProperty().SetColor(1, 0, 0)

# Create graphics stuff
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set the background and size
ren.AddActor(triangulation)
ren.SetBackground(0, 0, 0)
renWin.SetSize(250, 250)
renWin.Render()

cam1 = ren.GetActiveCamera()
cam1.Zoom(1.5)

iren.Initialize()
renWin.Render()
iren.Start()
