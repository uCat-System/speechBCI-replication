import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d

from tutorials import paths


def meanResamples(trlConcat, nResamples):
    resampleMeans = np.zeros((nResamples, trlConcat.shape[1], trlConcat.shape[2]))
    for rIdx in range(nResamples):
        resampleIdx = np.random.randint(0, trlConcat.shape[0], trlConcat.shape[0])
        resampleTrl = trlConcat[resampleIdx, :, :]
        resampleMeans[rIdx, :, :] = np.sum(resampleTrl, axis=0) / trlConcat.shape[0]

    return resampleMeans


def triggeredAvg(features, eventIdx, eventCodes, window, smoothSD=0, computeCI=True, nResamples=100):
    winLen = window[1] - window[0]
    codeList = np.unique(eventCodes)

    featAvg = np.zeros([len(codeList), winLen, features.shape[1]])
    featCI = np.zeros([len(codeList), winLen, features.shape[1], 2])
    allTrials = []

    for codeIdx in range(len(codeList)):
        trlIdx = np.squeeze(np.argwhere(eventCodes == codeList[codeIdx]))
        trlSnippets = []
        for t in trlIdx:
            if (eventIdx[t] + window[0]) < 0 or (eventIdx[t] + window[1]) >= features.shape[0]:
                continue
            trlSnippets.append(features[(eventIdx[t] + window[0]):(eventIdx[t] + window[1]), :])

        trlConcat = np.stack(trlSnippets, axis=0)
        allTrials.append(trlConcat)

        if smoothSD > 0:
            trlConcat = gaussian_filter1d(trlConcat, smoothSD, axis=1)

        featAvg[codeIdx, :, :] = np.mean(trlConcat, axis=0)

        if computeCI:
            tmp = np.percentile(meanResamples(trlConcat, nResamples), [2.5, 97.5], axis=0)
            featCI[codeIdx, :, :, :] = np.transpose(tmp, [1, 2, 0])

    return featAvg, featCI, allTrials


def meanSubtract(dat):
    dat['binnedTX'] = dat['binnedTX'].astype(np.float32)
    blockList = np.squeeze(np.unique(dat['blockNum']))
    for b in blockList:
        loopIdx = np.squeeze(dat['blockNum'] == b)
        dat['binnedTX'][loopIdx, :] -= np.mean(dat['binnedTX'][loopIdx, :], axis=0, keepdims=True)
    return dat


def plotPreamble():
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['svg.fonttype'] = 'none'


def main():
    subtractMeansWithinBlock = False
    baseDir = paths.PATH_BASE

    phonemesDat = scipy.io.loadmat(paths.PATH_TUNING_TASKS / 't12.2022.04.21_phonemes.mat')
    orofacialDat = scipy.io.loadmat(paths.PATH_TUNING_TASKS / 't12.2022.04.21_orofacial.mat')
    fiftyWordDat = scipy.io.loadmat(paths.PATH_TUNING_TASKS / 't12.2022.05.03_fiftyWordSet.mat')

    # mean subtraction
    if subtractMeansWithinBlock:
        phonemesDat = meanSubtract(phonemesDat)
        orofacialDat = meanSubtract(orofacialDat)
        fiftyWordDat = meanSubtract(fiftyWordDat)

    # get triggered averages for making PSTHs
    fAvg_phones, fCI_phones, trials_phones = triggeredAvg(phonemesDat['tx2'].astype(np.float32),
                                                          phonemesDat['goTrialEpochs'][:, 0],
                                                          np.squeeze(phonemesDat['trialCues']), [-100, 100], smoothSD=4)

    fAvg_orofacial, fCI_orofacial, trials_orofacial = triggeredAvg(orofacialDat['tx2'].astype(np.float32),
                                                                   orofacialDat['goTrialEpochs'][:, 0],
                                                                   np.squeeze(orofacialDat['trialCues']), [-100, 100],
                                                                   smoothSD=4)

    fAvg_fiftyWord, fCI_fiftyWord, trials_fiftyWord = triggeredAvg(fiftyWordDat['tx2'].astype(np.float32),
                                                                   fiftyWordDat['goTrialEpochs'][:, 0],
                                                                   np.squeeze(fiftyWordDat['trialCues']), [-100, 100],
                                                                   smoothSD=4)

    # define the conditions that go into each PSTH panel
    eyebrowSet = [5, 6, 7, 8]
    eyeSet = [9, 10, 11, 12]
    jawSet = [13, 14, 17, 18]
    larynxSet = [19, 20, 21, 22]
    lipsSet = [23, 24, 27, 28]
    tongueSet = [29, 30, 31, 32]
    orofacialSets = [eyebrowSet, eyeSet, jawSet, lipsSet, tongueSet, larynxSet]
    phonemeSet = [0, 5, 30, 39]
    fiftyWordSet = [4, 7, 20, 31]

    setTitles = ['Forehead', 'Eyelids', 'Jaw', 'Lips', 'Tongue', 'Larynx', 'Phonemes', 'Words']

    channelIdx = 69  # 69
    timeAxis = np.arange(-100, 100) * 0.02

    legends = [['Furrow', 'Raise', 'Raise Left', 'Raise Right'],
               ['Close', 'Open Wide', 'Wink Left', 'Wink Right'],
               ['Jaw Clench', 'Jaw Drop', 'Jaw Left', 'Jaw Right'],
               ['Frown', 'Pucker', 'Smile', 'Tuck'],
               ['Tongue Down', 'Tongue Up', 'Tongue Left', 'Tongue Right'],
               ['Hum High', 'Hum Loud', 'Hum Low', 'Hum Soft'],
               ['B', 'G', 'IH', 'AE'],
               ['Bring', 'Comfortable', 'Help', 'Need']]

    plotPreamble()

    if subtractMeansWithinBlock:
        yLimit = [-10, 25]
    else:
        yLimit = [0, 35]

    plt.figure(figsize=(len(setTitles) * (4 / 5), 0.7), dpi=300)
    for setIdx in range(len(orofacialSets)):
        conIdx = orofacialSets[setIdx]
        plt.subplot(1, len(orofacialSets) + 2, setIdx + 1)
        lines = []
        for c in range(len(conIdx)):
            tmp = plt.plot(timeAxis, 50 * fAvg_orofacial[conIdx[c], :, channelIdx], linewidth=1)
            lines.append(tmp[0])
            plt.fill_between(timeAxis,
                             50 * fCI_orofacial[conIdx[c], :, channelIdx, 0],
                             50 * fCI_orofacial[conIdx[c], :, channelIdx, 1], alpha=0.3)
        plt.ylim(yLimit)
        plt.plot([0, 0], plt.gca().get_ylim(), '--k', linewidth=0.75)
        if setIdx > 0:
            plt.gca().set_yticklabels([])
        else:
            if subtractMeansWithinBlock:
                plt.ylabel('Δ TX Rate (Hz)')
            else:
                plt.ylabel('TX Rate (Hz)')

        ax = plt.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.75)
        ax.tick_params(length=2)

        plt.xlabel('Time (s)')
        plt.title(setTitles[setIdx])
        plt.legend(lines, legends[setIdx], loc='upper left', bbox_to_anchor=(0.0, -0.75), fontsize=4, frameon=False)
        # plt.xlim([-0.5,1.0])

    plt.subplot(1, len(orofacialSets) + 2, len(orofacialSets) + 1)
    conIdx = phonemeSet
    for c in range(len(conIdx)):
        plt.plot(timeAxis, 50 * fAvg_phones[conIdx[c], :, channelIdx], linewidth=1)
        plt.fill_between(timeAxis,
                         50 * fCI_phones[conIdx[c], :, channelIdx, 0],
                         50 * fCI_phones[conIdx[c], :, channelIdx, 1], alpha=0.3)
        plt.ylim(yLimit)
        plt.plot([0, 0], plt.gca().get_ylim(), '--k', linewidth=0.75)
        plt.title(setTitles[-2])
        plt.gca().set_yticklabels([])
        plt.xlabel('Time (s)')
        ax = plt.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.75)
        ax.tick_params(length=2)
        plt.legend(lines, legends[-2], loc='upper left', bbox_to_anchor=(0.0, -0.75), fontsize=4, frameon=False)

    plt.subplot(1, len(orofacialSets) + 2, len(orofacialSets) + 2)
    conIdx = fiftyWordSet
    for c in range(len(conIdx)):
        plt.plot(timeAxis, 50 * fAvg_fiftyWord[conIdx[c], :, channelIdx], linewidth=1)
        plt.fill_between(timeAxis,
                         50 * fCI_fiftyWord[conIdx[c], :, channelIdx, 0],
                         50 * fCI_fiftyWord[conIdx[c], :, channelIdx, 1], alpha=0.3)
        plt.ylim(yLimit)
        plt.plot([0, 0], plt.gca().get_ylim(), '--k', linewidth=0.75)
        plt.title(setTitles[-1])
        plt.gca().set_yticklabels([])
        plt.xlabel('Time (s)')
        ax = plt.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.75)
        ax.tick_params(length=2)
        plt.legend(lines, legends[-1], loc='upper left', bbox_to_anchor=(0.0, -0.75), fontsize=4, frameon=False)

    plt.show()


if __name__ == '__main__':
    main()
