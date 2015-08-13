#Use :
# - to install
# python3 setup.py install
# - to create tar.gz for distribution :
# python3 setup.py sdist
# - to build in place
# python3 setup.py build_ext --inplace


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

extensions = [
    Extension("rf.Geodesic", sources= ["rf/Geodesic.pyx"])
]

setup(
    name='rf',
    version='1.9',
    cmdclass = {'build_ext':build_ext},
    ext_modules = extensions,
    py_modules=['rf','rf.Splitter', 'rf.Criterion', 'rf.FeatureFunction', 'rf.TrainSamples', 'rf.Forest', 'rf.DecisionTree', 'rf.tree', 'rf.Jungle']
)




