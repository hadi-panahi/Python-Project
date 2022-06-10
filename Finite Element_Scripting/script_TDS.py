
modelName = 'Impact'

# Full-plate dimensions
L = 100.  # mm
H = 100.  # mm
thickness = 0.2  # mm

# Projectile
D = 5.  # mm (diameter)
v_0 = 80.e3  # mm/s  (impact velocity)

# Visualization
my_view = session.Viewport(name='Viewport: 1')
my_view.makeCurrent()
my_view.maximize()
executeOnCaeStartup()

# Create new model
Mdb()
model = mdb.models['Model-1']

# Composite material lamina
compMat = model.Material(name='Composite')
compMat.Density(table=((2e-09,),))
compMat.Elastic(type=ENGINEERING_CONSTANTS, table=((90000.0, 90000.0, 90000.0, 0.2, 0.2,
                                                    0.4, 30000.0, 30000.0, 10000.0),))
compMat.HashinDamageInitiation(table=((500.0, 200.0, 500.0, 200.0, 80.0, 80.0),))
compMat.hashinDamageInitiation.DamageEvolution(
    type=ENERGY, table=((100.0, 50.0, 100.0, 50.0),))
model.HomogeneousShellSection(name='Section-Composite',
                              preIntegrate=OFF, material='Composite', thicknessType=UNIFORM,
                              thickness=thickness, thicknessField='', idealization=NO_IDEALIZATION,
                              poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT,
                              useDensity=OFF, integrationRule=SIMPSON, numIntPts=3)

# Projectile material
projMat = model.Material(name='STEEL')
projMat.Density(table=((8e-09,),))
projMat.Elastic(table=((250000.0, 0.3),))
model.HomogeneousSolidSection(name='Section-Steel', material='STEEL', thickness=None)

# Create target part
L_ = 0.5 * L
H_ = 0.5 * H

s = model.ConstrainedSketch(name='Target', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
# s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(0.0, 0.0), point2=(L_, H_))
p = model.Part(name='TARGET', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# Create sets of the edges
p.Set(faces=p.faces, name='ALL')
p.Set(edges=p.edges.findAt(((0.5 * L_, H_, 0.),)), name='TOP')
p.Set(edges=p.edges.findAt(((0.5 * L_, 0., 0.),)), name='BOTTOM')
p.Set(edges=p.edges.findAt(((0., 0.5 * H_, 0.),)), name='LEFT')
p.Set(edges=p.edges.findAt(((L_, 0.5 * H_, 0.),)), name='RIGHT')

# Create surface
p.Surface(side1Faces=p.sets['ALL'].faces, name='SURF')

# Assign material section
p.SectionAssignment(region=p.sets['ALL'], sectionName='Section-Composite', offset=0.0,
                    offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# Assign orientation (lamina properties)
p.MaterialOrientation(region=p.sets['ALL'], orientationType=GLOBAL, axis=AXIS_1,
                      additionalRotationType=ROTATION_NONE, localCsys=None, fieldName='')

# Mesh target part
# Assign mesh seeds
p.seedPart(size=5.0, deviationFactor=0.1, minSizeFactor=0.1)

p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=p.sets['LEFT'].edges, minSize=0.5,
                 maxSize=5.0, constraint=FINER)
p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=p.sets['BOTTOM'].edges, minSize=0.5,
                 maxSize=5.0, constraint=FINER)
# Select finite elements
elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=EXPLICIT, hourglassControl=DEFAULT,
                          elemDeletion=ON, maxDegradation=0.999)
elemType2 = mesh.ElemType(elemCode=S3R, elemLibrary=EXPLICIT)
p.setElementType(regions=p.sets['ALL'], elemTypes=(elemType1, elemType2))

# Modify mesh controls
p.setMeshControls(regions=p.faces, allowMapped=False)
# Generate the mesh
p.generateMesh()

#################################################
# PHASE II: Create the PROJECTILE part and mesh
#################################################

D_ = 0.5 * D
s1 = model.ConstrainedSketch(name='PROJECTILE', sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
# s1.setPrimaryObject(option=STANDALONE)
s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
# s1.FixedConstraint(entity=g[2])
s1.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, D_), point2=(0.0, -D_), direction=CLOCKWISE)
s1.Line(point1=(0.0, -D_), point2=(0.0, D_))
# s1.VerticalConstraint(entity=g[4], addUndoState=False)
# s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
p = model.Part(name='PROJECTILE', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidRevolve(sketch=s1, angle=90.0, flipRevolveDirection=OFF)

# Create sets
p.Set(cells=p.cells, name='ALL')
p.Set(vertices=p.vertices.findAt(((0., D_, 0.),)), name='TIP')
p.Set(faces=p.faces.findAt(((0.5 * D_, 0., 0.),)), name='FACE-1')
p.Set(faces=p.faces.findAt(((0., 0., 0.5 * D_),)), name='FACE-2')
p.Set(faces=p.faces.findAt(((D_ * cos(pi / 4.), 0., D_ * sin(pi / 4.)),)), name='SURF')

# Create surface
p.Surface(side1Faces=p.sets['SURF'].faces, name='SURF')

# Assign material
p.SectionAssignment(region=p.sets['ALL'], sectionName='Section-Steel', offset=0.0,
                    offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# Make partition to improve meshing
planeXZ = p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
p.PartitionCellByDatumPlane(datumPlane=p.datums[planeXZ.id], cells=p.cells)

# Mesh Projectile
# Mesh seeds
p.seedPart(size=D_ * 0.25, deviationFactor=0.1, minSizeFactor=0.1)
# Assign element types
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN,
                          secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
p.setElementType(regions=p.sets['ALL'], elemTypes=(elemType1, elemType2, elemType3))
# Generate mesh
p.generateMesh()

#####################################################
# PHASE III: Create Assembly with instances
#####################################################

a = model.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

# Instance Target
p = model.parts['TARGET']
a.Instance(name='TARGET', part=p, dependent=ON)

# Instance Projectile
p = model.parts['PROJECTILE']
a.Instance(name='PROJECTILE', part=p, dependent=ON)

# Rotate and trasnlate Projectile
a.rotate(instanceList=('PROJECTILE',), axisPoint=(0.0, 0.0, 0.0),
         axisDirection=(L, 0.0, 0.0), angle=-90.0)
a.translate(instanceList=('PROJECTILE',), vector=(0.0, 0.0, D_ + 0.1))

#####################################################
# PHASE IV: Step and Boundary conditions
#####################################################

# Create step
model.ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=0.0005)

# Boundary conditions of the Target
at = a.instances['TARGET']
model.PinnedBC(name='BC-support-1', createStepName='Step-1', region=at.sets['RIGHT'])
model.PinnedBC(name='BC-support-2', createStepName='Step-1', region=at.sets['TOP'])
model.XsymmBC(name='BC-symX', createStepName='Step-1', region=at.sets['LEFT'])
model.YsymmBC(name='BC-symY', createStepName='Step-1', region=at.sets['BOTTOM'])

# Boundary conditions of the Projectile
ap = a.instances['PROJECTILE']
model.XsymmBC(name='BC-proj-symX', createStepName='Step-1', region=ap.sets['FACE-2'])
model.YsymmBC(name='BC-proj-symY', createStepName='Step-1', region=ap.sets['FACE-1'])
# Initial velocity of the projectile
model.Velocity(name='Velocity', region=ap.sets['ALL'], field='', distributionType=MAGNITUDE,
               velocity1=0.0, velocity2=0.0, velocity3=-v_0, omega=0.0)

# Define interaction property for contact between projectile and target
contProp = model.ContactProperty('IntProp-1')
contProp.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                        constraintEnforcementMethod=DEFAULT)
contProp.TangentialBehavior(formulation=FRICTIONLESS)
# Assign interaction
interaction = model.ContactExp(name='Int-2', createStepName='Initial')
r11 = ap.surfaces['SURF']
r12 = at.surfaces['SURF']
interaction.includedPairs.setValuesInStep(stepName='Initial', useAllstar=OFF, addPairs=((r11, r12),))
interaction.contactPropertyAssignments.appendInStep(
    stepName='Initial', assignments=((GLOBAL, SELF, 'IntProp-1'),))

#####################################################
# PHASE V: Output, Job and save model
#####################################################

# Field output
model.fieldOutputRequests['F-Output-1'].setValues(variables=('A', 'CSTRESS', 'LE', 'S', 'U', 'V'))

model.FieldOutputRequest(name='F-Output-Target', createStepName='Step-1',
                         variables=('DAMAGEFT', 'DAMAGEFC', 'DAMAGEMT', 'DAMAGEMC', 'DAMAGESHR', 'SDEG', 'STATUS'),
                         region=at.sets['ALL'], sectionPoints=DEFAULT, rebar=EXCLUDE)

# History output
model.HistoryOutputRequest(name='H-Output-Tip', createStepName='Step-1',
                           variables=('U3', 'V3', 'A3'), region=ap.sets['TIP'],
                           sectionPoints=DEFAULT, rebar=EXCLUDE)

model.HistoryOutputRequest(name='H-Output-3', createStepName='Step-1',
                           variables=('ALLKE',), region=ap.sets['ALL'],
                           sectionPoints=DEFAULT, rebar=EXCLUDE)

# Create job
my_job = mdb.Job(name=modelName, model='Model-1', numCpus=1)

# Visualization of the model
a = mdb.models['Model-1'].rootAssembly
my_view.setValues(displayedObject=a)
my_view.assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
my_view.enableMultipleColors()
my_view.setColor(initialColor='#BDBDBD')
cmap = my_view.colorMappings['Part instance']
my_view.setColor(colorMapping=cmap)
my_view.disableMultipleColors()

# Save model
mdb.saveAs(pathName=modelName)


#-----------------------------------------------------------------------------
btn = getWarningReply('Do you want the script to submit the simulation and postprocess it?', buttons=(YES, NO))

if btn == YES:
    # Run simulation
    my_job.submit()
    my_job.waitForCompletion()

    # Open odb
    odb = session.openOdb(modelName + '.odb')
    my_view.setValues(displayedObject=odb)

    # Read Kinetic energy of the projectile
    projectile = 'PROJECTILE'
    ap = odb.rootAssembly.instances[projectile]

    ke_data = xyPlot.XYDataFromHistory(odb=odb, steps=('Step-1',),
                                       outputVariableName='Kinetic energy: ALLKE PI: ' + projectile + ' in ELSET ALL')
    _, ke = zip(*ke_data)

    # Read velocity of the projectile tip
    v_data = xyPlot.XYDataFromHistory(odb=odb, steps=('Step-1',),
                                      outputVariableName='Spatial velocity: V3 PI: ' + projectile + ' Node ' + str(
                                          ap.nodeSets['TIP'].nodes[0].label) + ' in NSET TIP')
    time, v = zip(*v_data.data)
    v_0 = -v[0] / 1e3
    v_f = -v[-1] / 1e3

    if v_f < 0.:
        v_f2 = 0.
        print("The projectile was arrested for v_0 = {0:.1f} m/s".format(v_0))
    else:
        v_f2 = v_0 * sqrt(ke[-1] / ke[0])
        print("The projectile penetrated the target: v_0 = {0:.1f} m/s, v_f = {1:.1f} m/s".format(v_0, v_f2))
    
    # Visualization
    my_view.odbDisplay.basicOptions.setValues(mirrorAboutXzPlane=True, mirrorAboutYzPlane=True)
    my_view.odbDisplay.commonOptions.setValues(visibleEdges=FEATURE, deformationScaling=UNIFORM, uniformScaleFactor=1)
    my_view.odbDisplay.setPrimaryVariable(variableLabel='DAMAGESHR', outputPosition=INTEGRATION_POINT, )
    my_view.odbDisplay.display.setValues(plotState=CONTOURS_ON_DEF)
    my_view.view.setValues(session.views['Iso'])
    # my_view.viewportAnnotationOptions.setValues(triad=OFF, legend=OFF, annotations=OFF, compass=OFF)
    session.animationController.setValues(animationType=TIME_HISTORY, viewports=(my_view.name, ))
    session.animationController.animationOptions.setValues(frameRate=40)
    session.animationController.play(duration=UNLIMITED)
    
    
else:

    getWarningReply("You can submit the job manually", buttons=(YES,))
