from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import numpy as np
import os
import fnmatch
import random


dataDir = '/u/cs401/A3/data/'
# dataDir = '/Users/kamran/Documents/CSC401/csc401_a3/data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only
        component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something
        for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    if x.ndim == 1:
        d = x.shape[0]
        prob = -np.sum(np.divide(
            np.multiply(x - myTheta.mu[m], x - myTheta.mu[m]),
            2*myTheta.Sigma[m])
        )
    else:
        d = x.shape[1]
        prob = -np.sum(np.divide(
            np.multiply(x - myTheta.mu[m], x - myTheta.mu[m]),
            2*myTheta.Sigma[m]),
            axis=1
        )
    precompute = - d/2 * np.log(2*np.pi)
    precompute -= 1/2 * np.log(
        np.prod(
            myTheta.Sigma[m]
        )
    )
    # print('log_b_m_x', np.exp(prob + precompute))
    return prob + precompute


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional
        vector x, and model myTheta
        See equation 2 of handout
    '''
    probs = np.empty((myTheta.omega.shape[0], x.shape[0]))
    for i in range(myTheta.omega.shape[0]):
        probs[i] = log_b_m_x(i, x, myTheta)
    denum = logsumexp(probs, axis=0, b=myTheta.omega)
    num = np.log(myTheta.omega[m][0]) + probs[m]
    # raise Exception
    # print(len(probs[0]))
    # print('log_p_m_x', np.log(num/denum))
    return num-denum


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed
        MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead
        pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which
        is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    # iterate over all samples
    log_probs = logsumexp(log_Bs, b=myTheta.omega, axis=0)
    return np.sum(log_probs)


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu,
        sigma)'''
    myTheta = theta(speaker, M, X.shape[1])
    # Initialize myTheta (set omega values to 1/M)
    myTheta.omega = 1/M * np.ones((M, 1))
    # Initialize myTheta (set Sigma to Identity)
    myTheta.Sigma = M*np.ones((M, X.shape[1]))
    # Initialize myTheta (set mu to a random vector from the data)
    myTheta.mu = X[:M][:]
    # Implement Training
    i = 0
    prev_l = -np.inf
    improvement = np.inf
    while i < maxIter and improvement >= epsilon:
        print (i)
        # Compute Intermediate Results
        log_Bs = np.zeros((M, X.shape[0]))
        log_Ps = np.zeros((M, X.shape[0]))
        for j in range(M):
            log_Bs[j][:] = log_b_m_x(j, X, myTheta)
            log_Ps[j][:] = log_p_m_x(j, X, myTheta)
        L = logLik(log_Bs, myTheta)
        print (L)

        probs = np.sum(np.exp(log_Ps), axis=1).reshape(M, 1)
        myTheta.omega = probs/X.shape[0]

        mu_update = np.dot(np.exp(log_Ps), X)
        sigma_update = np.dot(np.exp(log_Ps), np.multiply(X,X))

        myTheta.mu = np.divide(
            mu_update, probs
        )
        sigma_update = np.divide(
            sigma_update, probs
        )
        sigma_update = sigma_update - np.multiply(myTheta.mu, myTheta.mu)
        myTheta.sigma = sigma_update
        improvement = L - prev_l
        prev_l = L
        i += 1
        # print(myTheta.omega)
        # print(myTheta.mu)
        # print(myTheta.sigma)
    return myTheta


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
    for i in range(len(models)):
        M = models[i].omega.shape[0]
        log_Bs = np.zeros((M, 1))
        for i in range(M):
            log_Bs[i][1] = log_b_m_x(i, mfcc, models[i])
        likelihood = logLik(log_Bs, models[i])
        log_likelihoods.append(likelihood)
        log_names[likelihood] = [i, models[i].name]
    log_likelihoods = sorted(log_likelihoods, reverse=True)
    # Fix
    print('Correct Id: {}, Correct Name {}'.format(
        correctID, models[correctID].name)
    )
    for i in range(k):
        print('{} {}'.format(
            models[i].name, log_likelihoods[i])
        )
    bestModel = log_names[log_likelihoods[0]][0]
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

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

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0*numCorrect/len(testMFCCs)
