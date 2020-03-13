#include<Python.h>

static PyObject* example_mul(PyObject* self, PyObject *args)
{
   float a,b;
   if(!PyArg_ParseTuple(args, "ff", &a, &b))
   {
      return NULL;
   }
   return Py_BuildValue("f", a*b);
}

static PyObject* example_div(PyObject* self, PyObject *args)
{
    float a,b;
    if(!PyArg_ParseTuple(args, "ff", &a, &b))
    {
       return NULL;
    }
    return Py_BuildValue("f", a/b);
}


static char mul_docs[] = "mul(a,b):return a*b\n";
static char div_docs[] = "div(a,b):return a/b\n";


static PyMethodDef example_methods[] =
{
   {"mul", (PyCFunction)example_mul, METH_VARARGS, mul_docs},
   {"div", (PyCFunction)example_div, METH_VARARGS, div_docs},
   {NULL, NULL, 0, NULL}
};

//void PyMODINIT_FUNC initexample(void)
//{
//    Py_InitModule3("example", example_methods, "Extension module example!");
//}

static struct PyModuleDef exmaplemodule = {
    PyModuleDef_HEAD_INIT,
    "FPS", /* module name */
    NULL, /* module documentation, may be NULL */
    -1,
    example_methods /* the methods array */
};

PyMODINIT_FUNC PyInit_FPS(void)
{
    return PyModule_Create(&exmaplemodule);
}





