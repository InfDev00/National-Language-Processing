#sigmoid method
def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1/(1+np.exp(-x))
    ### END YOUR CODE

    return s

#naive_softmax method
#softmax method에 log를 통해 구현
def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the part 1)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the part 1)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the part 1)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the part 1)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~5-8 Lines)
    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    y_pred = softmax(outsideVectors.dot(centerWordVec[:, np.newaxis]).reshape(-1))
    loss = -np.log(y_pred[outsideWordIdx])
    gradCenterVec = -outsideVectors[outsideWordIdx] + np.sum(y_pred[:, np.newaxis]*outsideVectors, axis=0)
    gradOutsideVecs = y_pred[:, np.newaxis].dot(centerWordVec[np.newaxis, :])
    gradOutsideVecs[outsideWordIdx] -= centerWordVec

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


#negSampling method
def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)
    ### Please use your implementation of sigmoid in here.

    y_pred = np.zeros((len(indices), ))
    y_pred[0] = sigmoid(outsideVectors[outsideWordIdx][np.newaxis, :].dot(centerWordVec[:, np.newaxis]))
    y_pred[1:] = sigmoid(-outsideVectors[negSampleWordIndices].dot(centerWordVec[:, np.newaxis]).reshape(-1))
    loss = -np.log(y_pred[0]) - np.sum(np.log(y_pred[1:]))
    gradCenterVec = -(1-y_pred[0])*outsideVectors[outsideWordIdx] + np.sum((1-y_pred[1:])[:, np.newaxis]*outsideVectors[negSampleWordIndices], axis=0)
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    gradOutsideVecs[outsideWordIdx] = -(1-y_pred[0])*centerWordVec
    for i, Idx in enumerate(negSampleWordIndices):
        gradOutsideVecs[Idx] += (1-y_pred[i+1]) * centerWordVec

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs



#skipgram method
def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in part 1)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in part 1)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in part 1)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in part 1)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVecs = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    currentCenterWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[currentCenterWordIdx]

    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        single_loss, gradCenterVec, gradOutsideVec = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += single_loss
        gradCenterVecs[currentCenterWordIdx] += gradCenterVec
        gradOutsideVecs += gradOutsideVec
    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVecs