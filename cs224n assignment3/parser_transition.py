
#transition mechanics your parser will use.
#parser가 사용할 method 및 init 구현
class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do NOT alter it in your code.
        self.sentence = sentence

        ### YOUR CODE HERE (3 Lines)
        self.stack = ["ROOT"]
        self.buffer = self.sentence.copy()
        self.dependencies = list()
        ### Your code should initialize the following fields:
        ###     self.stack: The current stack represented as a list with the top of the stack as the
        ###                 last element of the list.
        ###     self.buffer: The current buffer represented as a list with the first item on the
        ###                  buffer as the first item of the list
        ###     self.dependencies: The list of dependencies produced so far. Represented as a list of
        ###             tuples where each tuple is of the form (head, dependent).
        ###             Order for this list doesn't matter.
        ###
        ### Note: The root token should be represented with the string "ROOT"
        ### Note: If you need to use the sentence object to initialize anything, make sure to not directly 
        ###       reference the sentence object.  That is, remember to NOT modify the sentence object.

        ### END YOUR CODE


    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """
        ### YOUR CODE HERE (~7-12 Lines)
        
        if transition == "S":
          self.stack.append(self.buffer.pop(0))
        elif transition == "LA":
          first_word = self.stack[-1]
          second_word = self.stack.pop(-2)
          self.dependencies.append((first_word, second_word))
        elif transition == "RA":
          second_word = self.stack[-2]
          first_word = self.stack.pop(-1)

          self.dependencies.append((second_word, first_word))
          
        ### TODO:
        ###     Implement a single parsing step, i.e. the logic for the following as
        ###     described above:
        ###         1. Shift
        ###         2. Left Arc
        ###         3. Right Arc

        ### END YOUR CODE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


#minibatch_parse method
#batch_size 만큼씩만 parse
def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []

    ### YOUR CODE HERE (~8-10 Lines)

    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]

    while unfinished_parses:
      minibatches = unfinished_parses[:batch_size]
      transitions = model.predict(minibatches)
      for minibatch, transition in zip(minibatches, transitions):
        minibatch.parse_step(transition)
        if len(minibatch.buffer)==0 and len(minibatch.stack) == 1:
          unfinished_parses.remove(minibatch)

    dependencies = [partial_parse.dependencies for partial_parse in partial_parses]


    ### TODO:
    ###     Implement the minibatch parse algorithm.  Note that the pseudocode for this algorithm is given above.
    ###
    ###     Note: A shallow copy can be made with the "=" sign in python, e.g.
    ###                 unfinished_parses = partial_parses[:].
    ###             Here `unfinished_parses` is a shallow copy of `partial_parses`.
    ###             In Python, a shallow copied list like `unfinished_parses` does not contain new instances
    ###             of the object stored in `partial_parses`. Rather both lists refer to the same objects.
    ###             In our case, `partial_parses` contains a list of partial parses. `unfinished_parses`
    ###             contains references to the same objects. Thus, you should NOT use the `del` operator
    ###             to remove objects from the `unfinished_parses` list. This will free the underlying memory that
    ###             is being accessed by `partial_parses` and may cause your code to crash.


    ### END YOUR CODE

    return dependencies