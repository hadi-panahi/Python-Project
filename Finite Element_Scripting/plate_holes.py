"""
This script was developed by Miguel Herraez
TecnoDigital School
For more information visit: https://tecnodigitalschool.com/course-abaqus-scripting/
"""


from abaqus import *
from abaqusConstants import *
from caeModules import *

# Plate dimensions
L = 150.
H = 50.
th = 2.

# Holes
columns = 6
rows = 3
radius = 4.

# Mesh
elem_size = 4.

# Peak stress
max_stress = 300.  # MPa



# New model
Mdb()
model = mdb.models['Model-1']

# Sketch and part
s = model.ConstrainedSketch(name='sketch-plate', sheetSize=200.0)
s.rectangle(point1=(0.0, 0.0), point2=(L, H))
p = model.Part(name='Plate', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# Sets and surface
e = p.edges
edges = e.getSequenceFromMask(mask=('[#8 ]', ), )
p.Set(edges=edges, name='LEFT')
edges = e.getSequenceFromMask(mask=('[#1 ]', ), )
p.Set(edges=edges, name='BOTTOM')
edges = e.getSequenceFromMask(mask=('[#2 ]', ), )
p.Set(edges=edges, name='RIGHT')
edges = e.getSequenceFromMask(mask=('[#4 ]', ), )
p.Set(edges=edges, name='TOP')
f = p.faces
faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
p.Set(faces=faces, name='ALL')
side1Edges = e.getSequenceFromMask(mask=('[#2 ]', ), )
p.Surface(side1Edges=side1Edges, name='Surf-1')

# Holes
s1 = model.ConstrainedSketch(name='sketch-holes', sheetSize=200.0)

dH = H / (rows + 1.)
dL = L / (columns + 1.)

# Aligned rows of holes
# y0 = dH
# for i in range(rows):
	# x0 = dL
	# for j in range(columns):
		# s1.CircleByCenterPerimeter(center=(x0, y0), point1=(x0, y0 + radius))
		# x0 += dL
	# y0 += dH

# Zig-zagging rows of holes	
y0 = dH
for i in range(rows):
	if i % 2 == 0:  # Even row
		x0 = dL
		for j in range(columns):
			s1.CircleByCenterPerimeter(center=(x0, y0), point1=(x0, y0 + radius))
			x0 += dL
	else:  # Odd row
		x0 = 1.5 * dL
		for j in range(columns-1):
			s1.CircleByCenterPerimeter(center=(x0, y0), point1=(x0, y0 + radius))
			x0 += dL
	y0 += dH

p.Cut(sketch=s1)

# Material and section
material = model.Material(name='STEEL')
material.Elastic(table=((210000.0, 0.3), ))
model.HomogeneousSolidSection(name='Section-steel', material='STEEL', thickness=th)
region = p.sets['ALL']
p.SectionAssignment(region=region, sectionName='Section-steel', offset=0.0, 
	offsetType=MIDDLE_SURFACE, thicknessAssignment=FROM_SECTION)

# Mesh   
p.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
elemType1 = mesh.ElemType(elemCode=CPS4, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD)
p.setElementType(regions=p.sets['ALL'], elemTypes=(elemType1, elemType2))
p.generateMesh()

# Assembly
a = model.rootAssembly
instance = a.Instance(name=p.name, part=p, dependent=ON)
session.viewports[session.currentViewportName].setValues(displayedObject=a)

# Step
step = model.StaticStep(name='Step-1', previous='Initial')

# Boundary condition
region = instance.sets['LEFT']
model.XsymmBC(name='BC-1', createStepName=step.name, region=region)

# Load
region = instance.surfaces['Surf-1']
# model.Pressure(name='Load', createStepName=step.name, region=region, magnitude=-0.5)
analyt_field = model.ExpressionField(name='AnalyticalField-1', expression='-Y * Y / ' + str(H*H))
model.Pressure(name='Load', createStepName=step.name, region=region, 
               distributionType=FIELD, field=analyt_field.name, magnitude=max_stress, amplitude=UNSET)
    
# Job
job = mdb.Job(name='Job-1', model=model.name, type=ANALYSIS, resultsFormat=ODB)

# Save model
mdb.saveAs(pathName='plate_with_holes')

# Run simulation
job.submit(consistencyChecking=OFF)