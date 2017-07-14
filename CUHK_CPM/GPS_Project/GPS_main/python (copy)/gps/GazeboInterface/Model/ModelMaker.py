#!/usr/bin/env python
__author__ = 'dyz'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    "CreateBoxModel",
    "CreateBoxVisual",
    "CreateCylinderModel",
    "CreateCylinderVisual",
    "CreateSphereModel",
    "CreateSphereVisual",
]

import lxml.etree as ltr


# for create box visual
def __BoxVisual(root, visual_name, x, y, z, Colour_name):
    VISUAL = ltr.SubElement(root, "visual", name=visual_name + "_visual")
    GEOMETRY = ltr.SubElement(VISUAL, "geometry")
    BOX = ltr.SubElement(GEOMETRY, "box")
    SIZE = ltr.SubElement(BOX, "size")
    SIZE.text = str(x) + " " + str(y) + " " + str(z)
    MATERIAL = ltr.SubElement(VISUAL, "material")
    SCRIPT = ltr.SubElement(MATERIAL, "script")
    URI = ltr.SubElement(SCRIPT, "uri")
    URI.text = "model://materials/scripts/Pi.material"
    NAME = ltr.SubElement(SCRIPT, "name")
    NAME.text = "Pi/" + str(Colour_name)
    return root


# for create box collision
def __BoxCollision(root, collision_name, x, y, z):
    COLLISION = ltr.SubElement(root, "collision", name=collision_name + "_collision")
    GEOMETRY = ltr.SubElement(COLLISION, "geometry")
    BOX = ltr.SubElement(GEOMETRY, "box")
    SIZE = ltr.SubElement(BOX, "size")
    SIZE.text = str(x) + " " + str(y) + " " + str(z)
    return root


# for create box inertial
def __BoxInertial(root, mass, x, y, z):
    INERTIAL = ltr.SubElement(root, "inertial")
    MASS = ltr.SubElement(INERTIAL, "mass")
    MASS.text = str(mass)
    INERTIA = ltr.SubElement(INERTIAL, "inertia")
    IXX = ltr.SubElement(INERTIA, "ixx")
    IXX.text = str(mass / 12.0 * (y ** 2 + z ** 2))
    IXY = ltr.SubElement(INERTIA, "ixy")
    IXY.text = str(0)
    IXZ = ltr.SubElement(INERTIA, "ixz")
    IXZ.text = str(0)
    IYY = ltr.SubElement(INERTIA, "iyy")
    IYY.text = str(mass / 12.0 * (x ** 2 + z ** 2))
    IYZ = ltr.SubElement(INERTIA, "iyz")
    IYZ.text = str(0)
    IZZ = ltr.SubElement(INERTIA, "izz")
    IZZ.text = str(mass / 12.0 * (y ** 2 + x ** 2))
    return root


# for create Cylinder collision
def __CylinderCollision(root, collision_name, radius, length):
    COLLISION = ltr.SubElement(root, "collision", name=collision_name + "_collision")
    GEOMETRY = ltr.SubElement(COLLISION, "geometry")
    CYLINDER = ltr.SubElement(GEOMETRY, "cylinder")
    RADIUS = ltr.SubElement(CYLINDER, "radius")
    LENGTH = ltr.SubElement(CYLINDER, "length")
    RADIUS.text = str(radius)
    LENGTH.text = str(length)
    return root


# for create Cylinder visual
def __CylinderVisual(root, visual_name, radius, length, Colour_name):
    VISUAL = ltr.SubElement(root, "visual", name=visual_name + "_visual")
    GEOMETRY = ltr.SubElement(VISUAL, "geometry")
    CYLINDER = ltr.SubElement(GEOMETRY, "cylinder")
    RADIUS = ltr.SubElement(CYLINDER, "radius")
    LENGTH = ltr.SubElement(CYLINDER, "length")
    RADIUS.text = str(radius)
    LENGTH.text = str(length)
    MATERIAL = ltr.SubElement(VISUAL, "material")
    SCRIPT = ltr.SubElement(MATERIAL, "script")
    URI = ltr.SubElement(SCRIPT, "uri")
    URI.text = "model://materials/scripts/Pi.material"
    NAME = ltr.SubElement(SCRIPT, "name")
    NAME.text = "Pi/" + str(Colour_name)
    return root


# for create Cylinder inertial
def __CylinderInertial(root, mass, radius, length):
    INERTIAL = ltr.SubElement(root, "inertial")
    MASS = ltr.SubElement(INERTIAL, "mass")
    MASS.text = str(mass)
    INERTIA = ltr.SubElement(INERTIAL, "inertia")
    IXX = ltr.SubElement(INERTIA, "ixx")
    IXX.text = str(1.0 / 12.0 * mass * (3 * radius ** 2 + length ** 2))
    IXY = ltr.SubElement(INERTIA, "ixy")
    IXY.text = str(0)
    IXZ = ltr.SubElement(INERTIA, "ixz")
    IXZ.text = str(0)
    IYY = ltr.SubElement(INERTIA, "iyy")
    IYY.text = str(1.0 / 12.0 * mass * (3 * radius ** 2 + length ** 2))
    IYZ = ltr.SubElement(INERTIA, "iyz")
    IYZ.text = str(0)
    IZZ = ltr.SubElement(INERTIA, "izz")
    IZZ.text = str(1.0 / 2.0 * mass * length ** 2)
    return root


# for create Sphere Collision
def __SphereCollision(root, collision_name, radius):
    COLLISION = ltr.SubElement(root, "collision", name=collision_name + "_collision")
    GEOMETRY = ltr.SubElement(COLLISION, "geometry")
    SPHERE = ltr.SubElement(GEOMETRY, "sphere")
    RADIUS = ltr.SubElement(SPHERE, "radius")
    RADIUS.text = str(radius)


# for create Sphere Visual
def __SphereVisual(root, visual_name, radius, Colour_name):
    VISUAL = ltr.SubElement(root, "visual", name=visual_name + "_visual")
    GEOMETRY = ltr.SubElement(VISUAL, "geometry")
    SPHERE = ltr.SubElement(GEOMETRY, "sphere")
    RADIUS = ltr.SubElement(SPHERE, "radius")
    RADIUS.text = str(radius)
    MATERIAL = ltr.SubElement(VISUAL, "material")
    SCRIPT = ltr.SubElement(MATERIAL, "script")
    URI = ltr.SubElement(SCRIPT, "uri")
    URI.text = "model://materials/scripts/Pi.material"
    NAME = ltr.SubElement(SCRIPT, "name")
    NAME.text = "Pi/" + str(Colour_name)


# for create Sphere inertial
def __SphereInertial(root, mass, radius):
    INERTIAL = ltr.SubElement(root, "inertial")
    MASS = ltr.SubElement(INERTIAL, "mass")
    MASS.text = str(mass)
    INERTIA = ltr.SubElement(INERTIAL, "inertia")
    IXX = ltr.SubElement(INERTIA, "ixx")
    IXX.text = str(2.0 / 3.0 * mass * radius ** 2)
    IXY = ltr.SubElement(INERTIA, "ixy")
    IXY.text = str(0)
    IXZ = ltr.SubElement(INERTIA, "ixz")
    IXZ.text = str(0)
    IYY = ltr.SubElement(INERTIA, "iyy")
    IYY.text = str(2.0 / 3.0 * mass * radius ** 2)
    IYZ = ltr.SubElement(INERTIA, "iyz")
    IYZ.text = str(0)
    IZZ = ltr.SubElement(INERTIA, "izz")
    IZZ.text = str(2.0 / 3.0 * mass * radius ** 2)
    return root


def CreateBoxModel(mass, x, y, z, color, static=False):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="box")
    if static:
        PLUGIN = ltr.SubElement(MODEL, "plugin", name="box_static_plugin", filename="libstatic_body.so")
        PLUGIN.text = "\n"
    Link = ltr.SubElement(MODEL, "link", name="box_link")
    __BoxInertial(Link, mass, x, y, z)
    __BoxCollision(Link, "box", x, y, z)
    __BoxVisual(Link, "box", x, y, z, color)
    return ltr.tostring(ROOT, pretty_print=True)


def CreateBoxVisual(x, y, z, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="box")
    PLUGIN = ltr.SubElement(MODEL, "plugin", name="box_static_plugin", filename="libstatic_body.so")
    PLUGIN.text = "\n"
    Link = ltr.SubElement(MODEL, "link", name="box_link")
    __BoxVisual(Link, "box", x, y, z, color)
    return ltr.tostring(ROOT, pretty_print=True)


def CreateCylinderModel(mass, radius, length, color, static=False):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="Cylinder")
    if static:
        PLUGIN = ltr.SubElement(MODEL, "plugin", name="Cylinder_static_plugin", filename="libstatic_body.so")
        PLUGIN.text = "\n"
    Link = ltr.SubElement(MODEL, "link", name="Cylinder_link")
    __CylinderInertial(Link, mass, radius, length)
    __CylinderCollision(Link, "cylinder", radius, length)
    __CylinderVisual(Link, "cylinder", radius, length, color)
    return ltr.tostring(ROOT, pretty_print=True)


def CreateCylinderVisual(radius, length, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="Cylinder")
    PLUGIN = ltr.SubElement(MODEL, "plugin", name="Cylinder_static_plugin", filename="libstatic_body.so")
    PLUGIN.text = "\n"
    Link = ltr.SubElement(MODEL, "link", name="Cylinder_link")
    __CylinderVisual(Link, "cylinder", radius, length, color)
    return ltr.tostring(ROOT, pretty_print=True)


def CreateSphereModel(mass, radius, color, static=False):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="Sphere")
    if static:
        PLUGIN = ltr.SubElement(MODEL, "plugin", name="Sphere_static_plugin", filename="libstatic_body.so")
        PLUGIN.text = "\n"
    Link = ltr.SubElement(MODEL, "link", name="Sphere_link")
    __SphereInertial(Link, mass, radius)
    __SphereCollision(Link, "sphere", radius)
    __SphereVisual(Link, "sphere", radius, color)
    return ltr.tostring(ROOT, pretty_print=True)


def CreateSphereVisual(radius, color):
    ROOT = ltr.Element("sdf", version="1.5")
    MODEL = ltr.SubElement(ROOT, "model", name="Sphere")
    PLUGIN = ltr.SubElement(MODEL, "plugin", name="Sphere_static_plugin", filename="libstatic_body.so")
    PLUGIN.text = "\n"
    Link = ltr.SubElement(MODEL, "link", name="Sphere_link")
    __SphereVisual(Link, "sphere", radius, color)
    return ltr.tostring(ROOT, pretty_print=True)
