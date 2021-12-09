import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import linalg
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


def get_amino_count(amino_count, group):
    for key in group.keys():
        seq = group[key]
        for i in range(len(seq)):
            if i in amino_count.keys():
                if seq[i] in amino_count[i].keys():
                    amino_count[i][seq[i]] += 1
                else:
                    amino_count[i][seq[i]] = 1
            else:
                amino_count[i] = dict()
                amino_count[i][seq[i]] = 1


def fill_vectors(vectors, tot_amino_counts, grp, tot_seq_count):
    idx = 0
    imp_char = dict()
    for key in grp.keys():
        for j in range(len(vectors[idx])):
            val = grp[key][j]
            count = tot_amino_counts[j][val]
            pct = count / tot_seq_count
            if pct > 0.47 and pct < 0.53:
                if j in imp_char.keys():
                    if imp_char[j] == val:
                        pct = pct
                    else:
                        pct = -pct
                else:
                    imp_char[j] = val
                    pct = pct
            elif pct > 0.80:
                pct = pct / 3
            else:
                pct = pct * 2 / 3
            vectors[idx, j] = pct
        idx += 1


def main():
    group = dict()
    aln = open('sequence[1182]-mod2.fasta', 'r')
    read = True
    while read:
        name = aln.readline()
        if not name:
            read = False
            break
        key = name[0:name.index(" ")]
        seq = ""
        for n in range(3):
            nxt = aln.readline()
            seq += nxt[0:nxt.index("\n")]
        group[key] = seq
    aln.close()


    tot_amino_count = dict()
    get_amino_count(tot_amino_count, group)

    aln_positions = len(tot_amino_count)
    tot_seq_count = len(group)

    vectors = np.zeros((tot_seq_count, aln_positions))
    fill_vectors(vectors, tot_amino_count, group, tot_seq_count)

    # working
    rbf_param = 0.7
    K = np.exp(-rbf_param * distance.cdist(vectors, vectors, metric='sqeuclidean'))

    D = K.sum(axis=1)
    D = np.sqrt(1/D)
    M = np.multiply(D[np.newaxis, :], np.multiply(K, D[:, np.newaxis]))

    U, Sigma, _ = linalg.svd(M, full_matrices=False, lapack_driver='gesvd')
    Usubset = U[:, 0:2]
    y_pred = KMeans(n_clusters=2).fit_predict(normalize(Usubset))

    # SVD to reduce dimensions of "vectors"
    U, S, VT = linalg.svd(vectors, full_matrices=False)
    rank = 2
    U_sub = U[:, :rank]

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.scatter(U_sub[0:48, 0], U_sub[0:48, 1], color="black", s=20)
    plt.scatter(U_sub[48:, 0], U_sub[48:, 1], color="orange", s=20)
    plt.title("Colored according to Group")

    plt.subplot(122)
    plt.scatter(U_sub[:, 0], U_sub[:, 1], c=y_pred, s=20)
    plt.title("Labels returned by Spectral Clustering")

    plt.show()

if __name__ == '__main__':
    main()
