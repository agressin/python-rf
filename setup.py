from distutils.core import setup

setup(
    name='rf',
    version='1.2',
    py_modules=['rf','rf.Splitter', 'rf.Criterion', 'rf.FeatureFunction', 'rf.TrainSamples', 'rf.Forest', 'rf.DecisionTree', 'rf.tree']
)
