#!/usr/bin/env python
################################################################################
#
# Simple repeater for gen and xyz files.
#
################################################################################
#
# Copyright (c) 2014, Balint Aradi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################
import re
import sys
import os
import numpy as np

# Starting line of a gen formatted structure
PAT_GENS_BEGIN = re.compile(r"^[ \t]*\d+[ \t]+(?:s|S|c|C)", re.MULTILINE)
PAT_XYZS_BEGIN = re.compile(r"^[ \t]*\d+[ \t]*$", re.MULTILINE)


class Geometry(object):
  """Only a data container with constructor for storing a geometry"""


  def __init__(self, species, indexes, coords, latVecs=None):
    """Constructs a geometry object.
  species -- Name of the species present in the structure.
  indexes -- For each atom the sequence number of its specie.
  coords  -- Coordinates (nAtom, 3).
  latVecs -- Translation vectors, if structure is periodic.
"""
    self.species = species[:]
    self.nSpecie = len(self.species)
    self.indexes = np.array(indexes)
    self.nAtom = len(indexes)
    self.coords = np.array(coords, dtype=float)
    if (self.coords.shape != (self.nAtom, 3)):
      raise "Bad coordinate array"
    if not latVecs:
      self.periodic = False
      self.latVecs = None
    else:
      self.latVecs = np.array(latVecs, dtype=float)
      if self.latVecs.shape != (3, 3):
        raise "Bad cell vectors"
      self.periodic = True


def gen2geo(input):
  """Converts the content of a gen file into a geometry object
  input  -- Content of a gen file as a string.
  return -- Initialized Geometry object.
"""
  lines = input.splitlines()
  words = lines[0].split()
  nAtom = int(words[0])
  periodic = (words[1] == "S") or (words[1] == "F")
  if periodic:
    fractional = words[1] == "F"
  species = lines[1].split()
  indexes = []
  coords = []
  for line in lines[2:2+nAtom]:
    words = line.split()
    indexes.append(int(words[1])-1)
    coords.append([ float(ww) for ww in words[2:5]])
  
  if periodic:
    latVecs = []
    for line in lines[2+nAtom+1:2+nAtom+4]:
      words = line.split()
      latVecs.append([ float(ww) for ww in words[0:3]])
    if fractional:
      coords = np.dot(coords, latVecs)

  if periodic:
    return Geometry(species, indexes, coords, latVecs)
  else:
    return Geometry(species, indexes, coords)



def xyz2geo(input, latVecs=None):
  """Converts the content of an xyz file into a geometry object
  input -- Content of the xyz file as string
  latVecs -- Text representation of the lattice vectors (optional)
  return -- Initialised Geometry object
  """

  lines = input.strip().splitlines()
  words = lines[0].split()
  nAtom = int(words[0])
  coords = np.zeros((nAtom, 3), dtype=float)
  indexes = np.zeros((nAtom,), dtype=int)
  species = []
  for ii in range(nAtom):
    words = lines[2+ii].split()
    specie = words[0].lower()
    specie = specie[0].upper() + specie[1:]
    try:
      ind = species.index(specie)
    except ValueError:
      species.append(specie)
      ind = len(species) - 1
    indexes[ii] = ind
    coords[ii][0] = float(words[1])
    coords[ii][1] = float(words[2])
    coords[ii][2] = float(words[3])
  if latVecs == None:
    ii += 2 + 1
  else:
    lines = latVecs.strip().splitlines()
    ii = 0
  if ii + 3 <= len(lines):
    periodic = True
    latVecs = np.zeros((3, 3), dtype=float)
    for jj in range(3):
      words = lines[ii+jj].split()
      latVecs[jj] = [ float(s) for s in words ]
  else:
    periodic = False
  if periodic:
    return Geometry(species, indexes, coords, latVecs)
  else:
    return Geometry(species, indexes, coords)
    


def geo2gen(geo):
  """Converts a geometry object to a gen string
  geo    -- Geometry object.
  return -- String containing the geometry in gen format.
"""

  result = []
  if geo.periodic:
    mode = "S"
  else:
    mode = "C"
  result.append("%5d %2s" % (geo.nAtom, mode))
  result.append(("%2s "*geo.nSpecie) % tuple(geo.species))

  for ii in range(geo.nAtom):
    result.append("%5d %3d %16.8f %16.8f %16.8f"
                  % (ii+1, geo.indexes[ii]+1, geo.coords[ii][0],
                     geo.coords[ii][1], geo.coords[ii][2]))
  if geo.periodic:
    result.append("%16.8f %16.8f %16.8f" % (0.0, 0.0, 0.0))
    for latv in geo.latVecs:
      result.append("%16.8f %16.8f %16.8f" % tuple(latv))

  return "\n".join(result)


def geo2xyz(geo, printLatVec=False):
  """Converts a geometry object to xyz format
  geo    -- Geometry object
  printLatVec -- If lattice vectors should be appended (default: False)
  return -- String containing the geometry in xyz format
"""
  result = []
  result.append("%5d" % geo.nAtom)
  result.append(("%2s "*geo.nSpecie) % tuple(geo.species))
  for ii in range(geo.nAtom):
    result.append(" %-3s %16.8f %16.8f %16.8f" % (geo.species[geo.indexes[ii]],
                                                  geo.coords[ii][0],
                                                  geo.coords[ii][1],
                                                  geo.coords[ii][2]))
  if geo.periodic and printLatVec:
    #result.append("%16.8f %16.8f %16.8f" % (0.0, 0.0, 0.0))
    for latv in geo.latVecs:
      result.append("%16.8f %16.8f %16.8f" % tuple(latv))

  return "\n".join(result)



def geos2gens(geos):
  """Converts a list of Geometry objects to string containing the geometries
in gen format.
  geos   -- List of Geometry objects.
  return -- String containing the geometries in gen format, separated by
    blank lines.
"""

  result = []
  for geo in geos:
    result.append(geo2gen(geo))
  return "\n\n".join(result)


def geos2xyzs(xyzs):
  """Converts a list of Geometry objects to string containing the geometries
in xyz format.
  geos   -- List of Geometry objects.
  return -- String containing the geometries in xyz format separated by newline
"""

  result = [ geo2xyz(geo) for geo in geos ]
  return "\n".join(result)


def gens2geos(gens):
  """Converts a string containing gen-formatted structures to a list of
Geometry objects.
  gens   -- String containing the gen-formatted structures. The structures
    may or may not be separated by empty lines.
  return -- List of Geometry objects.
"""

  # Split string by searching for the characteristic 1st line of the gen format.
  result = []
  start = 0
  match = PAT_GENS_BEGIN.search(gens)
  while match:
    prevStart = match.start()
    match = PAT_GENS_BEGIN.search(gens, pos=match.end())
    if match:
      gen = gens[prevStart:match.start()]
    else:
      gen = gens[prevStart:]
    result.append(gen2geo(gen))

  return result


def xyzs2geos(xyzs):
  """Converts a string containing xyz-formatted structures to a list of
Geometry objects.
  xyzs   -- String containing the xyz-formatted structures. The structures
    may or may not be separated by empty lines.
  return -- List of Geometry objects.
"""
  # Split string by searching for the characteristic 1st line of the gen format.
  result = []
  start = 0
  match = PAT_XYZS_BEGIN.search(xyzs)
  while match:
    prevStart = match.start()
    match = PAT_XYZS_BEGIN.search(xyzs, pos=match.end())
    if match:
      xyz = xyzs[prevStart:match.start()]
    else:
      xyz = xyzs[prevStart:]
    result.append(xyz2geo(xyz))

  return result



scriptName = os.path.basename(sys.argv[0])
genformat = scriptName == "repeatgen.py"

printLatVec = False
if not genformat and len(sys.argv) > 1 and sys.argv[1] == "-L":
  printLatVec = True
  del sys.argv[1]

periodic = False
if not genformat and len(sys.argv) > 2 and sys.argv[1] == "-l":
  del sys.argv[1]
  f = open(sys.argv[1], "r")
  latVecsTxt = f.read()
  f.close()
  lines = latVecsTxt.strip().splitlines()
  latVecs = np.zeros((3, 3), dtype=float)
  for ii in range(3):
    words = lines[ii].split()
    latVecs[ii] = [ float(s) for s in words ]
  periodic = True
  del sys.argv[1]

if len(sys.argv) < 5:
  print >> sys.stderr, "%s: Bad nr. of arguments"  % scriptName
  if genformat:
    print >> sys.stderr, "Usage: %s <geometry> <n1> <n2> <n3>\n" % scriptName
  else:
    print >> sys.stderr, (
      "Usage: %s [ -L ] [ -l latvec ] <geometry> <n1> <n2> <n3>\n"
      " -L for printing lattice vectors into the xyz-file\n"
      " -l for specifying a file containing the lattice vectors" % scriptName)
  sys.exit(1)

fileName = sys.argv[1]
ff = open(fileName, "r")
if genformat:

  geos = gens2geos(ff.read())
else:
  geos = xyzs2geos(ff.read())
ff.close()


if periodic:
  for geo in geos:
    geo.periodic = True
    geo.latVecs = latVecs

factor = np.array([ int(s) for s in sys.argv[2:5] ], dtype=int)
nCell = np.product(factor)
if nCell < 1:
  print >> sys.stderr, "Product of the repeat factors must be greater than one."
  sys.exit(1)

for geo in geos:

  if not geo.periodic:

    print >> sys.stderr, "Provided structure is not periodic"
    sys.exit(1)


  geo.indexes = np.resize(geo.indexes, (len(geo.indexes)*nCell,))
  geo.nAtom = geo.nAtom * nCell
  newCoords = []
  coeffs = np.zeros((3,1), dtype=float)
  for i1 in range(factor[0]):
    coeffs[0][0] = float(i1)
    for i2 in range(factor[1]):
      coeffs[1][0] = float(i2)
      for i3 in range(factor[2]):
        coeffs[2][0] = float(i3)
        transl = sum(coeffs * geo.latVecs, 0)
        newCoords.append(geo.coords + transl)
  geo.coords = np.reshape(newCoords, (-1, 3))
  geo.latVecs = (coeffs + [[1.0], [1.0], [1.0]]) * geo.latVecs

  if genformat:
    
    print(geo2gen(geo))
  else:

    print(geo2xyz(geo, printLatVec=printLatVec))
