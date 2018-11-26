import numpy as np

def sigmoid(x):
     return .5 * (1 + np.tanh(.5 * x))

def gibbs_sample_RBM(weight_matrix, bias_hidden, bias_visible, sample_visible,
                     sample_hidden, init_visible=None, init_hidden=None):
    """
    weight_matrix is num_hidden x num_visible matrix.
    bias_hidden is num_hidden x 1 matrix (real)
    bias_visible is num_visible x 1 matrix (real)
    init_hidden is num_hidden x 1 matrix ({0,1}-valued)
    init_visible is num_visible x 1 matrix ({0,1}-valued)

    returns hidden -- a num_hidden x 1 shape {0,1} matrix if sample_hidden=True
    returns visible -- a num_visible x 1 shape {0,1} matrix if sample_visible=True

    $P(H,V) = \frac{1}{Z} \exp(H^\top W V + \b^\top H + \c^\top V)$
    """

    num_hidden=len(bias_hidden)
    num_visible=len(bias_visible)

    if sample_hidden:
        expected_hidden=sigmoid(np.dot(weight_matrix, init_visible)+bias_hidden)
        rand_vec=np.random.uniform(0,1,num_hidden).reshape(num_hidden,1)
        hidden=rand_vec<expected_hidden
        hidden=hidden.astype(int)
        return hidden

    if sample_visible:
        expected_visible=sigmoid(np.dot(init_hidden.T,weight_matrix).T+bias_visible)
        rand_vec=np.random.uniform(0,1,num_visible).reshape(num_visible,1)
        visible=rand_vec<expected_visible
        visible=visible.astype(int)
        return visible
        
    return -1
