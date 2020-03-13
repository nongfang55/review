import source.CExtend.FPS.FPS
from ctypes import *


class Point(Structure):
    _fields_= [("x", c_float), ("y", c_float)]

if __name__ == "__main__":
    print(source.CExtend.FPS.FPS.div(-1, 1 ))
    dll = CDLL("Project1.dll")
    dll.addf.restype = c_float
    dll.addf.argtypes = [c_float, c_float]
    print(dll.addf(10, 30))
    p = Point(2,5)
    print(p.x, p.y)

    a = c_int(66)
    b = pointer(a)
    c = POINTER(c_int)(a)

    # print(b)
    # print(c)
    # print(type(b.contents))
    #
    # dll.print_point.argtypes = (c_int, POINTER(Point))
    # dll.print_point.restype = None
    p = Point(1, 2)
    # dll.print_point(1, byref(p))

    n = 5
    points = []
    for i in range(0, n):
        points.append(Point(i, i))

    g = (Point * n)(*points)
    print(g)

    dll.print_point.argtypes = (POINTER(Point), c_int)
    dll.print_point.restype = None
    dll.print_point(g, 5)






