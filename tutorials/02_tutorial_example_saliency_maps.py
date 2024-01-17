import os

import numpy as np
from omegaconf import OmegaConf
import tensorflow as tf
# import tensorflow.compat.v1 as tf

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from tutorials import paths


# Demo: saliency map analysis
# This notebook provides an overview of the phonetic similarity analysis from Fig. 3, using the baseline RNN and eval
# data provided on Dryad.


class GradientGetter:
    '''Simple object for handling saliency map computation. Accepts inputs:

        rnnModel (NeuralSequenceDecoder object) - model to analyze
        rnnInputs (2D float) - time x channels of neural data to use
        n_perturbations (int) - # of noise perturbations, see below
        noise_level (float)   - noise strength in [0, 1], see below

        The paper uses SmoothGrad (https://arxiv.org/pdf/1706.03825.pdf), a method for
        cleaning up saliency maps by averaging gradients over noisy perturbations of the input
        data. Using n_perturbations = 1 and noise_level = 0 yields the standard saliency map
        strategy. In the speech decoding paper, we used SmoothGrad to obtain a small boost
        in performance but for demo purposes we've turned it off by default.
        '''

    def __init__(self, rnnModel, rnnInputs, n_perturbations=1, noise_level=0):
        self.rnnModel = rnnModel
        self.rnnInputs = rnnInputs

        assert n_perturbations >= 1, "N_perturbations must be >= 1"
        assert noise_level >= 0, "noise_level must be >= 0"
        # Set n_perturbations = 1, noise_level = 0 for vanilla saliency maps
        # Set n_perturbations = 20, noise_level = 0.1 for our results

        self.n_perturbations = n_perturbations  # SmoothGrad perturbations per timepoint
        self.noise_level = noise_level  # noise SD as fraction of data range (0.1-0.2 in SmoothGrad paper)
        self.print_fraction = 0.1  # print progress

    def __call__(self):
        '''Heavy lifting done here. Step through the input neural data and calculate jacobians
           of outputs with respect to input channels.

           Returns:

               gradients (list of 4D float) - entries are extra dim x outputs x input kernel x 256 channels
        '''

        timelen = self.rnnInputs.shape[0]
        delta = self.rnnModel.model.stack_kwargs['kernel_size']
        stride = self.rnnModel.model.stack_kwargs['strides']

        noise_sd = self.noise_level * (np.percentile(self.rnnInputs, 98) - np.percentile(self.rnnInputs, 2))

        t = 0
        gradients = list()
        while t <= timelen - delta:
            step_data = self.rnnInputs[t:(t + delta), :]

            noise_grads = list()
            for i in range(self.n_perturbations):
                noisy_data = step_data + tf.random.normal(step_data.shape, 0, noise_sd)
                noise_grads.append(self.computeTimestepSaliency(noisy_data))

            step_grads = tf.math.reduce_mean(tf.concat(noise_grads, axis=0), axis=0, keepdims=True)
            gradients.append(step_grads)
            t += delta + stride

            # if (timeline - delta) * self.print_fraction
            print(t, '/', timelen - delta)
        return gradients

    @tf.function
    def computeTimestepSaliency(self, rnnInput):
        '''Runs a single datapoint (32 consecutive timepoints) through model. Inputs are:

        rnnInput (2D tensor) - timepoints x channels'''

        rnnStates = [self.rnnModel.model.initStates, None, None, None, None, None]

        with tf.GradientTape(persistent=True) as g:
            # x = data
            x = tf.tile(rnnInput[tf.newaxis, :, :], [1, 1, 1])
            x = tf.cast(x, tf.float32)
            g.watch(x)

            for t in range(5):
                output = self.rnnModel.normLayers[layerIdx](x)
                output = self.rnnModel.inputLayers[layerIdx](output)
                output, rnnStates = self.rnnModel.model(output, rnnStates, training=False, returnState=True)

            logitOut = tf.squeeze(output, 0)
            y = tf.cast(logitOut, tf.float32)

        jacobian = g.batch_jacobian(y, x, experimental_use_pfor=False)
        return jacobian


def getSimilarity(vectors, subtractMean=False, metric='similarity'):
    '''Inputs are:

        vectors (2D float)  - classes x feature matrix
        subtractMean (Bool) - toggle across-class mean subtraction
        metric (str)        - either 'similarity' (cosine sim) or a metric for
                              sklearn's pdist function

        Returns <dists> which is a classes x classes distance/similarity matrix
    '''

    if subtractMean:
        vectors -= np.mean(vectors, axis=0, keepdims=True)

    if metric == 'similarity':
        normed = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        dists = normed @ normed.T
    else:
        dists = squareform(pdist(vecs, metric))

    return dists


def main():
    # Initialize RNN model and datasets
    ckptDir = paths.PATH_RNN_BASELINE_RELEASE

    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    for sessIdx in range(4, 19):
        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = paths.PATH_DERIVED_TFRECORDS
    args['testDir'] = 'test'

    # Initialize model
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    # Let's load some example data from a single session and compute gradients of the output phone probabilities with
    # respect to the input channels:
    datasetIdx = -2
    dataset = list(nsd.tfValDatasets[datasetIdx])[0]
    layerIdx = nsd.args['dataset']['datasetToLayerMap'][datasetIdx]
    gradients = list()
    for idx in range(dataset['inputFeatures'].shape[0]):
        data = dataset['inputFeatures'][idx, :dataset['nTimeSteps'][idx], :]
        getter = GradientGetter(nsd, data)
        gradients.extend(getter())

        print(idx + 1, '/', dataset['inputFeatures'].shape[0])


if __name__ == '__main__':
    main()
