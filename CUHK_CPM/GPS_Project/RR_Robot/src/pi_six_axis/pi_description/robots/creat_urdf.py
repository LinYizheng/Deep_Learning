#!/usr/bin/env python
import os

FilePath = "pi.urdf"
# find = "prismatic"
# replace = "fixed"
if __name__ == '__main__':
	os.system("rosrun xacro xacro.py pi.urdf.xacro >"+FilePath)
	# with open(FilePath) as f:
	# 	s = f.read()
	# 	s = s.replace(find, replace)
	# with open(FilePath, "w") as f:
	# 	f.write(s)
	print "finish!"
