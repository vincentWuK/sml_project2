import ijson
import numpy as np


def read_data():
    path = "../Data/train.json"
    with open(path, 'r', encoding='utf-8') as f:
        # initialize variables
        obj = ijson.kvitems(f, 'item')
        data = list()
        row_ind = list()
        col_ind = list()
        year = list()
        venue = list()
        title = list()
        abstract = list()
        ind = 0
        authors = []
        # read data in file
        for k, v in obj:
            if k == "authors":
                # data.extend([1 for _ in range(len(v))])
                # row_ind.extend([ind for _ in range(len(v))])
                # col_ind.extend(v)
                authors.append(v)
                # ind += 1
            if k == "year":
                year.append(v)
            if k == "venue":
                if v == '':
                    venue.append(int(0))
                else:
                    venue.append((int(v) + 1))
            if k == "title":
                title.append(v)
            if k == "abstract":
                abstract.append(v)
        # author_sparse = csr_matrix((data, (row_ind, col_ind)), shape=(ind, 21246))
        # Y = author_sparse.toarray()
        X = []
        Y = []
        keep = []
        inds = [x for x in range(100)]
        for author in authors:
            for a in author:
                if a in inds:
                    keep.append(author)
                    break
        for index, k in enumerate(keep):
            temp = []
            co = []
            for au in k:
                if au in inds:
                    temp.append(au)
                else:
                    co.append(au)
            X.append(co)
            Y.append(temp)
        tx = np.zeros((len(X), 21246))
        for i in range(len(X)):
            tx[i][X[i]] += 1
        X = tx
        ty = np.zeros((len(Y), 100))
        for i in range(len(Y)):
            ty[i][Y[i]] += 1
        Y = ty
        return [X, Y]


def read_test_data():
    path = "../Data/test.json"
    with open(path, 'r', encoding='utf-8') as f:
        # initialize variables
        obj = ijson.kvitems(f, 'item')
        coauthors = []
        ind = 0
        authors = []
        # read data in file
        for k, v in obj:
            if k == "coauthors":
                coauthors.append(v)
        X = np.zeros((len(coauthors), 21246))
        for i in range(len(coauthors)):
            X[i][coauthors[i]] += 1
        return X
