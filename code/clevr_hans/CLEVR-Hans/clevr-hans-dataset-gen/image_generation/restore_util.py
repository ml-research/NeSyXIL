objnameout_objname = {
  "cube": "SmoothCube_v2",
  "sphere": "Sphere",
  "cylinder": "SmoothCylinder"
}

sizes = {
  "large": 0.7,
  "small": 0.35
}

mat = {
      "rubber": "Rubber",
      "metal": "MyMetal"
}

colors={
  "gray": [87, 87, 87],
  "red": [173, 35, 35],
  "blue": [42, 75, 215],
  "green": [29, 105, 20],
  "brown": [129, 74, 25],
  "purple": [129, 38, 192],
  "cyan": [41, 208, 208],
  "yellow": [255, 238, 51]
}


def get_obj_name(obj_name_out):
    return objnameout_objname[obj_name_out]


def get_by_size(size_name):
	return sizes[size_name]


def get_by_mat_name_out(mat_name_out):
	return mat[mat_name_out]


def get_by_colorname(colorname):
	return colors[colorname]
