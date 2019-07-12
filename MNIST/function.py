



def get_k_fold_data(k, i, x, y):
    
    n = x.shape[0]
    train_x, train_y = None, None
    valid_x, valid_y = None, None
    for j in range(k):
        idx = slice(j * n // k, (j+1) * n // k)
        fold_x, fold_y = x[idx], y[idx]
        
        if j == i:
            valid_x, valid_y = fold_x, fold_y
        elif train_x is None:
            train_x, train_y = fold_x, fold_y
        else:
            train_x = nd.concat(train_x, fold_x, dim = 0)
            train_y = nd.concat(train_y, fold_y, dim = 0)
    
    return train_x, train_y, valid_x, valid_y
