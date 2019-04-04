import os
import numpy as np
import re

# dataDir = '/u/cs401/A3/data/'
dataDir = '/Users/kamran/Documents/CSC401/csc401_a3/data/'


def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions,
    insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    n, m = len(r), len(h)
    R, B = np.zeros((n+1, m+1)), np.zeros((n+1, m+1))
    # OPTIMIZE: Initialize is so bad rn
    for i in range(1):
        for j in range(m+1):
            R[i][j] = max(i, j)
            if j != 0:
                B[i][j] = 2
    for i in range(n+1):
        for j in range(1):
            R[i][j] = max(i, j)
            if i != 0:
                B[i][j] = 1
    for i in range(1, n+1):
        for j in range(1, m+1):

            del_e = R[i-1][j] + 1
            sub_e = R[i-1][j-1] if r[i-1] == h[j-1] else R[i-1][j-1] + 1
            ins_e = R[i][j-1] + 1
            R[i][j] = min(del_e, sub_e, ins_e)
            if R[i][j] == del_e:
                B[i][j] = 1
            elif R[i][j] == ins_e:
                B[i][j] = 2
            elif R[i][j] == sub_e and r[i-1] == h[j-1]:
                B[i][j] = 4
            else:
                B[i][j] = 3

    # Lets find the individual errors
    i, j = n, m
    sub_errors = 0
    insert_errors = 0
    deletion_errors = 0
    while B[i][j] != 0:
        if B[i][j] == 4:
            i, j = i-1, j-1
        elif B[i][j] == 3:
            sub_errors += 1
            i, j = i-1, j-1
        elif B[i][j] == 2:
            insert_errors += 1
            j -= 1
        else:
            deletion_errors += 1
            i -= 1
    if n == 0:
        return (np.inf, sub_errors, insert_errors, deletion_errors)
    return (R[n][m]/n, sub_errors, insert_errors, deletion_errors)


if __name__ == "__main__":
    punc_to_remove = r'([!"#$%&\\()*+,-/:;<=>?@^_`{|}~])'
    for root, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # Reading the lines
            ref_fh = open(
                '{}{}{}'.format(root, speaker, '/transcripts.txt'), 'r'
            )
            reference_lines = ref_fh.readlines()
            if (len(reference_lines)) == 0:
                ref_fh.close()
                continue
            google_fh = open(
                '{}{}{}'.format(root, speaker, '/transcripts.Google.txt'), 'r'
            )
            google_lines = google_fh.readlines()
            google_fh.close()
            kaldi_fh = open(
                '{}{}{}'.format(root, speaker, '/transcripts.Kaldi.txt'), 'r'
            )
            kaldi_lines = kaldi_fh.readlines()
            kaldi_fh.close()
            # Sanitizing the lines
            for i in range(len(reference_lines)):
                ref = re.sub(
                    punc_to_remove, '', reference_lines[i]
                ).lower().split()[2:]
                google = re.sub(
                    punc_to_remove, '', google_lines[i]
                ).lower().split()[2:]
                kaldi = re.sub(
                    punc_to_remove, '', kaldi_lines[i]
                ).lower().split()[2:]
                g_score = Levenshtein(ref, google)
                k_score = Levenshtein(ref, kaldi)
                g_output = '{} Google {} {} S:{}, I:{}, D:{}'.format(
                    speaker,
                    i,
                    g_score[0],
                    g_score[1],
                    g_score[2],
                    g_score[3]
                )
                print(g_output)
                k_output = '{} Kaldi {} {} S:{}, I:{}, D:{}'.format(
                    speaker,
                    i,
                    k_score[0],
                    k_score[1],
                    k_score[2],
                    k_score[3]
                )
                print(k_output)
