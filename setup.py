from distutils.core import setup

setup(
    name='rf',
    version='1.5',
    py_modules=['rf','rf.Splitter', 'rf.Criterion', 'rf.FeatureFunction', 'rf.TrainSamples', 'rf.Forest', 'rf.DecisionTree', 'rf.tree', 'rf.Jungle']
)
