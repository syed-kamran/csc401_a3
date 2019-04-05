from a3_gmm import train, log_b_m_x, logLik
from sklearn.model_selection import train_test_split
import numpy as np
import os
import fnmatch
import random

# dataDir = '/u/cs401/A3/data/'
dataDir = '/Users/kamran/Documents/CSC401/csc401_a3/data/'


def experiment(speakers, M, max_iter, epsilon=0):
    print('Exp | Speakers: {}, Max Iters: {}, M: {}, Epsilon: {}'.format(
        speakers,
        max_iter,
        M,
        epsilon
    ))
    d = 13
    k = 5
    trainThetas = []
    testMFCCs = []
    f = open('gmmLiks.M.{}.mi.{}.s.{}.e.{}.txt'.format(
        M, max_iter, speakers, epsilon
    ), 'w')
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs[:speakers]:
            files = fnmatch.filter(os.listdir(
                os.path.join(dataDir, speaker)),
                '*npy'
            )
            random.shuffle(files)
            testMFCC = np.load(
                os.path.join(dataDir, speaker, files.pop())
            )
            testMFCCs.append(testMFCC)
            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(
                    os.path.join(dataDir, speaker, file)
                )
                X = np.append(X, myMFCC, axis=0)
            trainThetas.append(train(speaker, X, M, epsilon, max_iter))

    num_correct = 0
    for i in range(len(testMFCCs)):
        correct, log_likelihoods, log_names = test(
            testMFCCs[i], i, trainThetas, k
        )
        num_correct += correct
        f.write('Actual ID: {}\n'.format(trainThetas[i].name))
        for j in range(k):
            f.write('{} {}\n'.format(
                log_names[log_likelihoods[j]][1], log_likelihoods[j])
            )
    accuracy = num_correct/len(testMFCCs)
    f.write('TOTAL ACCURACY: {}'.format(accuracy))


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the
    correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods
        in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or
        exponent) does not matter
    '''
    bestModel = -1
    log_likelihoods = []
    log_names = {}
    T = mfcc.shape[0]
    for i in range(len(models)):
        M = models[i].omega.shape[0]
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m][:] = log_b_m_x(m, mfcc, models[i])
        likelihood = logLik(log_Bs, models[i])
        log_likelihoods.append(likelihood)
        log_names[likelihood] = [i, models[i].name]
    log_likelihoods = sorted(log_likelihoods, reverse=True)
    # Fix
    print('Actual ID: {}'.format(models[correctID].name))
    for i in range(k):
        print('{} {}'.format(
            log_names[log_likelihoods[i]][1], log_likelihoods[i])
        )
    bestModel = log_names[log_likelihoods[0]][0]
    correct = 1 if (bestModel == correctID) else 0
    return correct, log_likelihoods, log_names


if __name__ == '__main__':
    exp_speakers = [32, 20, 10, 5]
    exp_m = [8, 6, 4, 2]
    exp_max_iter = [20, 10, 15, 5, 0]

    # Perform m reduction experiment
    for m in exp_m:
        experiment(exp_speakers[0], m, exp_max_iter[0])
    # Perform max_iter reduction experiment
    for max_iter in exp_max_iter:
        experiment(exp_speakers[0], exp_m[0], max_iter)
    # Perform speaker reduction experiment
    for speakers in exp_speakers:
        experiment(speakers, exp_m[0], exp_max_iter[0])
