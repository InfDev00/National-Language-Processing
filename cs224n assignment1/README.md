Part 1: Count-based word vectors

You Shall know a word by the company it keeps (Firth, J. R. 1957:11)
Many word vector implementations are driven by the idea that similar words, i.e., (near) synonyms, will be used in similar contexts. As a result, similar words will often be spoken or written along with a shared subset of words, i.e., contexts. By examining these contexts, we can try to develop embeddings for our words. With this intuition in mind, many "old school" approaches to constructing word vectors relied on word counts. Here we elaborate upon one of those strategies, co-occurrence matrices (for more information, see here).

Co-occurence

A co-occurrence matrix counts how often things co-occur in some environment. Given some word  wi  occurring in the document, we consider the context window surrounding  wi  . Supposing our fixed window size is  n  , then this is the  n  preceding and  n  subsequent words in that document, i.e. words  wi−n…wi−1  and  wi+1…wi+n  . We build a co-occurrence matrix  M  , which is a symmetric word-by-word matrix in which  Mij  is the number of times  wj  appears inside  wi  's window among all documents.


Note: In NLP, we often add <START> and <END> tokens to represent the beginning and end of sentences, paragraphs or documents. In thise case we imagine <START> and <END> tokens encapsulating each document, e.g., "<START> All that glitters is not gold <END>", and include these tokens in our co-occurrence counts.

The rows (or columns) of this matrix provide one type of word vectors (those based on word-word co-occurrence), but the vectors will be large in general (linear in the number of distinct words in a corpus). Thus, our next step is to run dimensionality reduction. In particular, we will run SVD (Singular Value Decomposition), which is a kind of generalized PCA (Principal Components Analysis) to select the top  k  principal components. Here's a visualization of dimensionality reduction with SVD. In this picture our co-occurrence matrix is  A  with  n  rows corresponding to  n  words. We obtain a full matrix decomposition, with the singular values ordered in the diagonal  S  matrix, and our new, shorter length-  k  word vectors in  Uk .

This reduced-dimensionality co-occurrence representation preserves semantic relationships between words, e.g. doctor and hospital will be closer than doctor and dog.

If you are not friendly to SVD: you can refer to this comprehensive tutorial or these lecture notes (1, 2, 3) of Stanford CS168. While these materials give a great introduction to PCA/SVD, you do not have to understand all of these materials for this assignment or this class.
