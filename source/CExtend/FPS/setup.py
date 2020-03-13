from distutils.core import setup, Extension
setup(name="ReviewRecommendFPS", version="1.0", ext_modules=[Extension("FPS", ["FPSAlgorithm.c"])])