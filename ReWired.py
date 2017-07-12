import pygame, math
import numpy as np
import random
# The README is also included here below.
# I will definetly encourage you to read it if you havent done it allready
# and ask me if you need further detailed comments on anything.
def translationMatrix(dx=0,dy=0,dz=0):
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[dx,dy,dz,1]])
def translateAlongVectorMatrix(vector,distance):
    unit_vector = np.hstack([unitVector(vector) * distance,1])
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],unit_vector])
def scaleMatrix(s,cx=0,cy=0,cz=0):
    return np.array([[s,0,0,0],[0,s,0,0],[0,0,s,0],[cx*(1-s),cy*(1-s),cz*(1-s),1]])
def rotateXMatrix(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1,0, 0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
def rotateYMatrix(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[ 0,0,0,1]])
def rotateZMatrix(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s,0,0],[s,c,0,0],[0, 0,1,0],[0, 0,0,1]])
def rotateAboutVector(gx, gy, gz, x,y,z, radians):
    rotZ = np.arctan2(y, x)
    rotZ_matrix = rotateZMatrix(rotZ)
    (x, y, z, _) = np.dot(np.array([x,y,z,1]), rotZ_matrix)
    rotY = np.arctan2(x, z)
    matrix = translationMatrix(dx=-gx, dy=-gy, dz=-gz)
    matrix = np.dot(matrix, rotZ_matrix)
    matrix = np.dot(matrix, rotateYMatrix(rotY))
    matrix = np.dot(matrix, rotateZMatrix(radians))
    matrix = np.dot(matrix, rotateYMatrix(-rotY))
    matrix = np.dot(matrix, rotateZMatrix(-rotZ))
    matrix = np.dot(matrix, translationMatrix(dx=cx, dy=cy, dz=cz))
    return matrix
class Wireframe:
    def __init__(self, nodes=None):
        self.nodes = np.zeros((0,4))
        self.edges = []
        self.faces = []
        if nodes:
            self.addNodes(nodes)
    def addNodes(self, node_array):
        ones_added = np.hstack((node_array, np.ones((len(node_array),1))))
        self.nodes = np.vstack((self.nodes, ones_added))
    def addEdges(self, edge_list):
        self.edges += [edge for edge in edge_list if edge not in self.edges]
    def addFaces(self, face_list, face_colour=(255,255,255)):
        for node_list in face_list:
            num_nodes = len(node_list)
            if all((node < len(self.nodes) for node in node_list)):
                self.faces.append((node_list, np.array(face_colour, np.uint8)))
                self.addEdges([(node_list[n-1], node_list[n]) for n in range(num_nodes)])
    def output(self):
        if len(self.nodes) > 1:
            self.outputNodes()
        if self.edges:
            self.outputEdges()
        if self.faces:
            self.outputFaces()  
    def outputNodes(self):
        for i, (x,y,z,_) in enumerate(self.nodes):
            print("Node{}:({},{},{})".format(i, x, y, z))
    def outputEdges(self):
        for i, (node1, node2) in enumerate(self.edges):
            print("Edge{}:{}->{}".format(i, node1, node2)) 
    def outputFaces(self):
        for i, nodes in enumerate(self.faces):
            print("Face{}:({})".format(i, ", ".join(['{}'.format(n for n in nodes)])))
    def transform(self, transformation_matrix):
        self.nodes = np.dot(self.nodes, transformation_matrix)
    def findCentre(self):
        min_values = self.nodes[:,:-1].min(axis=0)
        max_values = self.nodes[:,:-1].max(axis=0)
        return 0.5*(min_values + max_values)
    def sortedFaces(self):
        return sorted(self.faces, key=lambda face: min(self.nodes[f][2] for f in face[0]))
    def update(self):
        pass
class WireframeGroup:
    def __init__(self):
        self.wireframes = {}
    def addWireframe(self, name, wireframe):
        self.wireframes[name] = wireframe
    def output(self):
        for name, wireframe in self.wireframes.items():
            print (name)
            wireframe.output()    
    def outputNodes(self):
        for name, wireframe in self.wireframes.items():
            print (name)
            wireframe.outputNodes()
    def outputEdges(self):
        for name, wireframe in self.wireframes.items():
            print (name)
            wireframe.outputEdges()
    def findCentre(self):
        min_values = np.array([wireframe.nodes[:,:-1].min(axis=0) for wireframe in self.wireframes.values()]).min(axis=0)
        max_values = np.array([wireframe.nodes[:,:-1].max(axis=0) for wireframe in self.wireframes.values()]).max(axis=0)
        return 0.5*(min_values + max_values)
    def transform(self, matrix):
        for wireframe in self.wireframes.values():
            wireframe.transform(matrix)
    def update(self):
        for wireframe in self.wireframes.values():
            wireframe.update()
def Cuboid(x,y,z,w,h,d):
    cuboid = Wireframe()
    cuboid.addNodes(np.array([[nx,ny,nz] for nx in (x,x+w) for ny in (y,y+h) for nz in (z,z+d)]))
    cuboid.addFaces([(0,1,3,2), (7,5,4,6), (4,5,1,0), (2,3,7,6), (0,2,6,4), (5,7,3,1)])
    return cuboid
def Spheroid(x,y,z,rx,ry,rz,resolution=10):
    spheroid   = Wireframe()
    latitudes  = [n*np.pi/resolution for n in range(1,resolution)]
    longitudes = [n*2*np.pi/resolution for n in range(resolution)]
    spheroid.addNodes([(x + rx*np.sin(n)*np.sin(m), y - ry*np.cos(m), z - rz*np.cos(n)*np.sin(m)) for m in latitudes for n in longitudes])
    num_nodes = resolution*(resolution-1)
    spheroid.addFaces([(m+n, (m+resolution)%num_nodes+n, (m+resolution)%resolution**2+(n+1)%resolution, m+(n+1)%resolution) for n in range(resolution) for m in range(0,num_nodes-resolution,resolution)])
    spheroid.addNodes([(x, y+ry, z),(x, y-ry, z)])
    spheroid.addFaces([(n, (n+1)%resolution, num_nodes+1) for n in range(resolution)])
    start_node = num_nodes-resolution
    spheroid.addFaces([(num_nodes, start_node+(n+1)%resolution, start_node+n) for n in range(resolution)])
    return spheroid
def HorizontalGrid(x,y,z,dx,dz,nx,nz):
    grid = Wireframe()
    grid.addNodes([[x+n1*dx, y, z+n2*dz] for n1 in range(nx+1) for n2 in range(nz+1)])
    grid.addEdges([(n1*(nz+1)+n2,n1*(nz+1)+n2+1) for n1 in range(nx+1) for n2 in range(nz)])
    grid.addEdges([(n1*(nz+1)+n2,(n1+1)*(nz+1)+n2) for n1 in range(nx) for n2 in range(nz+1)])
    return grid
def FractalLandscape(origin=(0,0,0), dimensions=(400,400), iterations=4, height=40):
    def midpoint(nodes):
        m = 1.0/ len(nodes)
        x = m * sum(n[0] for n in nodes) 
        y = m * sum(n[1] for n in nodes) 
        z = m * sum(n[2] for n in nodes) 
        return [x,y,z]
    (x,y,z) = origin
    (dx,dz) = dimensions
    nodes = [[x, y, z], [x+dx, y, z], [x+dx, y, z+dz], [x, y, z+dz]]
    edges = [(0,1), (1,2), (2,3), (3,0)]
    size = 2
    for i in range(iterations):
        for (n1, n2) in edges:
            nodes.append(midpoint([nodes[n1], nodes[n2]]))
        squares = [(x+y*size, x+y*size+1, x+(y+1)*size+1, x+(y+1)*size) for y in range(size-1) for x in range(size-1)]
        for (n1,n2,n3,n4) in squares:
            nodes.append(midpoint([nodes[n1], nodes[n2], nodes[n3], nodes[n4]]))
        nodes.sort(key=lambda node: (node[2],node[0]))
        size = size*2-1
        edges = [(x+y*size, x+y*size+1) for y in range(size) for x in range(size-1)]
        edges.extend([(x+y*size, x+(y+1)*size) for x in range(size) for y in range(size-1)])
        scale = height/2**(i*0.8)
        for node in nodes:
            node[1] += (random.random()-0.5)*scale
    grid = Wireframe(nodes)
    grid.addEdges(edges)
    grid = FractalLandscape(origin = (0,400,0), iterations=1)
    grid.output()
ROTATION_AMOUNT = np.pi/36
MOVEMENT_AMOUNT = 2
key_to_function = {
    pygame.K_LEFT:   (lambda x: x.transform(translationMatrix(dx=-MOVEMENT_AMOUNT))),
    pygame.K_RIGHT:  (lambda x: x.transform(translationMatrix(dx= MOVEMENT_AMOUNT))),
    pygame.K_UP:     (lambda x: x.transform(translationMatrix(dy=-MOVEMENT_AMOUNT))),
    pygame.K_DOWN:   (lambda x: x.transform(translationMatrix(dy= MOVEMENT_AMOUNT))),
    pygame.K_EQUALS: (lambda x: x.scale(1.25)),
    pygame.K_MINUS:  (lambda x: x.scale(0.8)),
    pygame.K_q:      (lambda x: x.rotate('x', ROTATION_AMOUNT)),
    pygame.K_w:      (lambda x: x.rotate('x',-ROTATION_AMOUNT)),
    pygame.K_a:      (lambda x: x.rotate('y', ROTATION_AMOUNT)),
    pygame.K_s:      (lambda x: x.rotate('y',-ROTATION_AMOUNT)),
    pygame.K_z:      (lambda x: x.rotate('z', ROTATION_AMOUNT)),
    pygame.K_x:      (lambda x: x.rotate('z',-ROTATION_AMOUNT))
}
light_movement = {
    pygame.K_q:      (lambda x: x.transform(rotateXMatrix(-ROTATION_AMOUNT))),
    pygame.K_w:      (lambda x: x.transform(rotateXMatrix( ROTATION_AMOUNT))),
    pygame.K_a:      (lambda x: x.transform(rotateYMatrix(-ROTATION_AMOUNT))),
    pygame.K_s:      (lambda x: x.transform(rotateYMatrix( ROTATION_AMOUNT))),
    pygame.K_z:      (lambda x: x.transform(rotateZMatrix(-ROTATION_AMOUNT))),
    pygame.K_x:      (lambda x: x.transform(rotateZMatrix( ROTATION_AMOUNT)))
}
class WireframeViewer(WireframeGroup):
    def __init__(self, width, height, name="Wireframe Viewer"):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(name)
        self.wireframes = {}
        self.wireframe_colours = {}
        self.object_to_update = []
        self.displayNodes = False
        self.displayEdges = True
        self.displayFaces = True
        self.perspective = False
        self.eyeX = self.width/2
        self.eyeY = 100
        self.view_vector = np.array([0, 0, -1])
        self.light = Wireframe()
        self.light.addNodes([[0, -1, 0]])
        self.min_light = 0.02
        self.max_light = 1.0
        self.light_range = self.max_light - self.min_light 
        self.background = (10,10,50)
        self.nodeColour = (250,250,250)
        self.nodeRadius = 4
        self.control = 0
    def addWireframe(self, name, wireframe):
        self.wireframes[name] = wireframe
        self.wireframe_colours[name] = (250,250,250)
    def addWireframeGroup(self, wireframe_group):
        for name, wireframe in wireframe_group.wireframes.items():
            self.addWireframe(name, wireframe)
    def scale(self, scale):
        scale_matrix = scaleMatrix(scale, self.width/2, self.height/2, 0)
        self.transform(scale_matrix)
    def rotate(self, axis, amount):
        (x, y, z) = self.findCentre()
        translation_matrix1 = translationMatrix(-x, -y, -z)
        translation_matrix2 = translationMatrix(x, y, z)
        if axis == 'x':
            rotation_matrix = rotateXMatrix(amount)
        elif axis == 'y':
            rotation_matrix = rotateYMatrix(amount)
        elif axis == 'z':
            rotation_matrix = rotateZMatrix(amount)
        rotation_matrix = np.dot(np.dot(translation_matrix1, rotation_matrix), translation_matrix2)
        self.transform(rotation_matrix)
    def display(self):
        self.screen.fill(self.background)
        light = self.light.nodes[0][:3]
        spectral_highlight = self.light.nodes[0][:3] + self.view_vector
        spectral_highlight /= np.linalg.norm(spectral_highlight)
        for name, wireframe in self.wireframes.items():
            nodes = wireframe.nodes
            if self.displayFaces:
                for (face, colour) in wireframe.sortedFaces():
                    v1 = (nodes[face[1]] - nodes[face[0]])[:3]
                    v2 = (nodes[face[2]] - nodes[face[0]])[:3]
                    normal = np.cross(v1, v2)
                    towards_us = np.dot(normal, self.view_vector)
                    if towards_us > 0:
                        normal /= np.linalg.norm(normal)
                        theta = np.dot(normal, light)
                        c = 0
                        if theta < 0:
                            shade = self.min_light *  colour
                        else:
                            shade = (theta * self.light_range + self.min_light) *  colour
                        pygame.draw.polygon(self.screen, shade, [(nodes[node][0], nodes[node][1]) for node in face], 0)
                if self.displayEdges:
                    for (n1, n2) in wireframe.edges:
                        if self.perspective:
                            if wireframe.nodes[n1][2] > -self.perspective and nodes[n2][2] > -self.perspective:
                                z1 = self.perspective/ (self.perspective + nodes[n1][2])
                                x1 = self.width/2  + z1*(nodes[n1][0] - self.width/2)
                                y1 = self.height/2 + z1*(nodes[n1][1] - self.height/2)                    
                                z2 = self.perspective/ (self.perspective + nodes[n2][2])
                                x2 = self.width/2  + z2*(nodes[n2][0] - self.width/2)
                                y2 = self.height/2 + z2*(nodes[n2][1] - self.height/2)
                                pygame.draw.aaline(self.screen,  (255,255,255), (x1, y1), (x2, y2), 1)
                        else:
                            pygame.draw.aaline(self.screen,  (255,255,255), (nodes[n1][0], nodes[n1][1]), (nodes[n2][0], nodes[n2][1]), 1)
            if self.displayNodes:
                for node in nodes:
                    pygame.draw.circle(self.screen, (255,255,255), (int(node[0]), int(node[1])), self.nodeRadius, 0)
        pygame.display.flip()
    def keyEvent(self, key):
        if key in key_to_function:
            key_to_function[key](self)
    def run(self):
        running = True
        key_down = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key_down = event.key
                elif event.type == pygame.KEYUP:
                    key_down = None
            if key_down:
                self.keyEvent(key_down)
            self.display()
            self.update()
        pygame.quit()
#There you go dear lovely git folks! 
# This is sort of a tribute to the pygame module envisioned by Petercollingridge 
# I named it ReWired as its corresponding Main class was elegantly named Wireframe. 
# This module is now capable of visualising self made three dimensional shapes,
# by using the py Pygame module, declaring and groupping nodes, 
# generating vectors and matrices, and finally applying unit vectors and 
# common matrix operations we get to display predefined 3d objects, 
# along with customising the scaling, rotation, and lightning factor.
# Peters old python2 code, due to technicallities, and/or lack of involvement, 
# according to his last contrib (2012) and the git comments, couldnt work at all.
#I have myself invested quite some time in setting up such a functional 3d-Engine, 
# with the allready existing pygame, Vec3D methods and various documentations. 
# After looking a bit further on github i stumbled upon his old yet wonderfull Repo, 
# and by applying my own code drafts, calculus and linear algebra knowledge, 
# quickly clarified a lot of the "hard" questions ive previously had difficulty with 
# involving setting up such an environment, especially with PyGame.
#As i wont forget his substantial addition to my understanding, 
# and to the allready existing and wonderfull github community,
#I have thereby created this entirely Python3 converted repo, as a tribute.
# It should hopefully fix a great deal of issues, and as i did have to change and thereby correct a lot of errors.
#YET! There are still some things that has changed in my newer version "GridMod"
# and should preferably be changed if forking this repo for best practice purposes...
#Such as:
#- variable_names/functions(): they should prefferably allways be lowercased in order to distinguish them from Classes
#                               or other Important methods and Handles from dependencies and imported packages 
#                              such as the sweet pygame were using (ex: Key_Attributes = K_up)
#- Renaming most function names, making their meaning more familliar (ex: translationMatrix() = translate_identity())
#- Incorporating all var declarations and functions into their corresponding Classes
#- Transforming more lists into tupples...
#- Changing Key Binding
#Feel free to let me know if you want to add or ask anything,
# perhaps you could have a look at my newest related project called GridMod! 
# The GridMod Project is all initialized by classes, and other personal additions/changes.
# And it is be the main 3d-engine concept im working on at the moment, 
# trying to implement kinetics and motion...
#Suggessting what could further be accomplished, to make this happen would be awesome, 
#Since this really took some great ammount of time as well to convert anf fix properly,
#you are definetly also welcome to simply tell me if you liked it!!
#Ruf.
def testWireframe():
    print ("Tetrahedron")
    triangle = Wireframe([[100,200,10], [200,200,10], [125,100,500]])
    triangle.addEdges([[0,1],[1,2],[2,0]])
    triangle.output()
    print ("Cuboid")
    cuboid = Cuboid(100,100,10,20,30,40)
    cuboid.output()
def testTranslate():
    cuboid = Cuboid(100,100,10,20,30,40)
    cuboid.outputNodes()  
    print("Translate cuboid along a vector")
    cuboid.transform(translationMatrix(4, 3, 1))
    cuboid.outputNodes()
def testScale():
    cuboid = Cuboid(100,100,10,20,30,40)
    cuboid.outputNodes()
    print ("> Scale cuboid by 2, centred at P(100,150,200)")
    cuboid.transform(scaleMatrix(2, 100, 150, 200))
    cuboid.outputNodes()
def testRotate():
    cuboid = Cuboid(100,100,10,20,30,40)
    cuboid.outputNodes()
    (x,y,z) = cuboid.findCentre()    
    translation_matrix = translationMatrix(-x, -y, -z)
    rotation_matrix = np.dot(translation_matrix, rotateXMatrix(math.pi/2))
    rotation_matrix = np.dot(rotation_matrix, -translation_matrix)
    print ("Rotate cuboid around its centre and the x-axis")
    cuboid.transform(rotation_matrix)
    cuboid.outputNodes()
def testWireframeGroup():
    groupped_frames = WireframeGroup()
    groupped_frames.addWireframe('cube1', Cuboid(100,100,10,20,30,40))
    groupped_frames.addWireframe('cube2', Cuboid(10,200,10,10,40,20))        
    groupped_frames.output()
def testWireframeDisplay():
    viewer = WireframeViewer(600, 400)
    viewer.addWireframe('cube', Cuboid(80,150,0,200,200,200))
    viewer.run()
def testSurfaceDisplayWithSphere():
    resolution = 10
    viewer = WireframeViewer(1200, 860)
    viewer.addWireframe('sphere', Spheroid(300, 200, 20, 160, 160, 160, resolution))
    faces = viewer.wireframes['sphere'].faces
    for i in range(int(resolution/4)):
        for j in range(int(resolution*2-4)):
            f = i*(resolution*4-8) +j
            faces[f][1][1] = 0
            faces[f][1][2] = 0
    print("Created a sphere with {} faces.".format(len(viewer.wireframes['sphere'].faces)))
    viewer.run()
def testWireframeDisplay3():
    viewer = WireframeViewer(1200, 860)
    viewer.addWireframe('grid',  HorizontalGrid(20,400,0,40,30,14,20))
    viewer.addWireframe('cube1', Cuboid(200,100,400,20,30,40))
    viewer.addWireframe('cube2', Cuboid(100,360, 20,10,40,20))
    viewer.addWireframe('sphere', Spheroid(250,300,100,20,30,40))
    viewer.run()
def chooseExample():
    examples = ['testWireframe',
                'testTranslate',
                'testScale',
                'testRotate',
                'testWireframeGroup',
                'testWireframeDisplay',
                'testSurfaceDisplayWithSphere',
                'testWireframeDisplay3',
                'exit']
    print("Options: ")
    for i, e in enumerate(examples, 1):
        print(" {}. {}".format(i, e))
    choice = input("Choose an option: ")
    print("> {}".format(examples[int(choice)-1]))
    exec("{}()".format(examples[int(choice)-1]))
chooseExample()
#Pygame Display keys:
# Q: rotate down 
# W: rotate up
# A: rotate left 
# S: rotate right 
# Arrows: position
