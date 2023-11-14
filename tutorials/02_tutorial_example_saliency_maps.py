import os

import tensorflow.compat.v1 as tf
from omegaconf import OmegaConf

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from tutorials import paths

# Demo: saliency map analysis
# This notebook provides an overview of the phonetic similarity analysis from Fig. 3, using the baseline RNN and eval
# data provided on Dryad.

# Initialize RNN model and datasets
ckptDir = paths.PATH_RNN_BASELINE_RELEASE

args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
args['loadDir'] = ckptDir
args['mode'] = 'infer'
args['loadCheckpointIdx'] = None

for x in range(len(args['dataset']['datasetProbabilityVal'])):
    args['dataset']['datasetProbabilityVal'][x] = 0.0

for sessIdx in range(4,19):
    args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
    args['dataset']['dataDir'][sessIdx] = paths.PATH_DERIVED_TFRECORDS
args['testDir'] = 'test'

# Initialize model
tf.compat.v1.reset_default_graph()
nsd = NeuralSequenceDecoder(args)