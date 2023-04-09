
#Write a method to work out the distinct words (word types) that occur in the corpus.
#corpus에서 나타난 word를 분류하는 method
def distinct_words(corpus): 
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    corpus_set = set()
    for word in corpus:
      corpus_set.update(word)

    corpus_words = sorted(list(corpus_set))
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words



#Write a method that constructs a co-occurrence matrix for a certain window-size  n
#window_size n 내부에서 동시 발생한 word를 co-occurrence matrix로 정리하는 method
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2ind = {}
    
    # ------------------
    # Write your implementation here.
    word2ind = dict(zip(words, range(num_words)))
    M = np.zeros((num_words, num_words))


    for cur_corpus in corpus:
      for idx, word in enumerate(cur_corpus):
        for i in range(1, window_size+1):
          left_idx = idx-i
          right_idx = idx+i


          if left_idx >= 0:
            left_word = cur_corpus[left_idx]
            M[word2ind[word], word2ind[left_word]]+=1

          if right_idx < len(cur_corpus):
            right_word = cur_corpus[right_idx]
            M[word2ind[word], word2ind[right_word]]+=1


    # ------------------

    return M, word2ind

#Construct a method that performs dimensionality reduction on the matrix to produce k-dimensional embeddings.
#Use SVD to take the top k components and produce a new matrix of k-dimensional embeddings.
#k-dimesional로 압축하는 method
#truncatedSVD를 사용해서 가장 자주 발생하는 K개를 추출
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced = svd.fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced



#a function to plot a set of 2D vectors in 2D space. For graphs, we will use Matplotlib (plt).
#2D vector를 2D space에 띄우는 함수
def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    for word in words:
      idx = word2ind[word]
      x, y = M_reduced[idx]

      plt.scatter(x, y, marker='x', color='red')
      plt.text(x, y, word, fontsize=9)

    plt.show()

    # ------------------


#모든 method를 활용하여 counted_based_word_vector 구현

# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
reuters_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']

plot_embeddings(M_normalized, word2ind_co_occurrence, words)