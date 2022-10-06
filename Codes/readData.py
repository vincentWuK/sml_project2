import numpy as np
from scipy.sparse import csr_matrix
import ijson
import pickle

MAX_AUTHOR = 21246  # range{0, 21245}
MAX_TITLE = 5000  # range{1,4999}
MAX_ABSTRACT = 5000  # range{1,4999}


def read_data():
    file_name_train = '../Data/train.json'
    with open(file_name_train, 'r', encoding='utf-8') as f:
        obj = ijson.kvitems(f, 'item')
        data = list()
        row_ind = list()
        col_ind = list()
        ind = 0
        year = list()
        venue = list()
        title = list()
        abstract = list()
        for k, v in obj:
            if k == "authors":
                data.extend([1 for _ in range(len(v))])
                row_ind.extend([ind for _ in range(len(v))])
                col_ind.extend(v)
                ind += 1
            if k == "year":
                year.append(v)
            if k == "venue":
                venue.append(v)
            if k == "title":
                title.append(v)
            if k == "abstract":
                abstract.append(v)
        author_sparse = csr_matrix((data, (row_ind, col_ind)), shape=(ind, MAX_AUTHOR)).toarray()
        X = np.column_stack((year, venue, title, abstract))
        return [X, author_sparse]


if __name__ == "__main__":
    data = read_data()
