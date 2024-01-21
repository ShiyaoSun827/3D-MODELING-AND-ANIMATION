###########################
# DO NOT MODIFY THIS PART #
# BUT YOU SHOULD READ IT  #
###########################
class Node:
    """
    Huffman tree node definition.
    """
    def __init__(self, symbol=None, count=0, left=None, right=None):
        """
        initialization
          symbol   : symbol to be coded
          count    : count of symbol
          left     : left child node
          right    : right child node
        """
        self.__left = left
        self.__right = right
        self.__symbol = symbol
        self.__count = count
        self.__code_word = ''

    def setLeft(self, l):
        """
        sets the left child of current node
        """
        self.__left = l
    
    def setRight(self, r):
        """
        sets the right child of current node
        """
        self.__right = r
    
    def getLeft(self):
        """
        returns the left child of current node
        """
        return self.__left
    
    def getRight(self):
        """
        returns the right child of current node
        """
        return self.__right

    def setSymbol(self, symbol):
        """
        sets coding symbol of current node
        """
        self.__symbol = symbol

    def getSymbol(self):
        """
        returns coding symbol of current node
        """
        return self.__symbol

    def setCount(self, count):
        """
        sets count of the symbol
        """
        self.__count = count

    def getCount(self):
        """
        returns count of the symbol
        """
        return self.__count
    
    def setCodeWord(self, code_word):
        """
        sets code-word of the symbol
        """
        self.__code_word = code_word

    def getCodeWord(self):
        """
        returns code-word of the symbol
        """
        return self.__code_word

    def __lt__(self, node):
        return self.__count < node.getCount()

    def __repr__(self):
        return "symbol: {}, count: {}, code-word: {}".format(self.__symbol, self.__count, self.__code_word)

###########################
# DO NOT MODIFY THIS PART #
# BUT YOU SHOULD READ IT  #
###########################

#############################
# YOUR OWN HELPER FUNCTIONS #
#############################

#############################
# YOUR OWN HELPER FUNCTIONS #
#############################
def return_leaves(root):
    leaves = []
    stack = [root]
    while len(stack) > 0:
        node = stack.pop()
        if not node.getLeft() and not node.getRight():
            leaves.append(node)
        if node.getRight():
            stack.append(node.getRight())
        if node.getLeft():
            stack.append(node.getLeft())
    return leaves

##############################
# FINISH THE BELOW FUNCTIONS #
##############################
def buildDictionary(message):
    """
    In this function, you need to count the occurrence of every symbol in the message and 
    return it in a python dictionary. The keys of the dictionary are the symbols, the values of 
    the dictionary is their corresponding occurrences.  
    counts the occurrence of every symbol in the message and store it in a python dictionary
      parameter:
        message: input message string
      return:
        python dictionary, key = symbol, value = occurrence
    """
    out_put = {}
    for i in message:
        if i not in out_put.keys():
            out_put[i] = 1
        else:
            out_put[i] += 1
    return out_put
        

def buildHuffmanTree(word_dict):
    """
    uses the word dictionary to generate a huffman tree using a min heap
      parameter:
        word_dict  : word dictionary generated by buildDictionary()
      return:
        root node of the huffman tree
    """
    #create a dict of nodes(use frequency as values)
    nodes = {}
    for i in word_dict.keys():
        temp = Node(symbol=i,count = word_dict[i])
        nodes[temp] = word_dict[i]
    #if len(nodes) == 1:
        #for i,j in nodes.items():
            #i.setCodeWord('0')
        #return i
   
  
  
    #sort dict
    sorted_items = sorted(nodes.items(),key = lambda item:item[1],reverse = True)
    sorted_nodes = {key: value for key, value in sorted_items}
    #sorted_nodes = dict(sorted_items)
  

    #find the two smallest value
    while len(sorted_nodes) > 1:
        minimum_left_node,value = sorted_nodes.popitem()
        minimum_right_node,value = sorted_nodes.popitem()
        internal_node = Node(symbol= None,
                             count=int(minimum_left_node.getCount() + minimum_right_node.getCount()))
        internal_node.setLeft(minimum_left_node)
        internal_node.setRight(minimum_right_node)
        sorted_nodes[internal_node] = internal_node.getCount()
        sorted_items = sorted(sorted_nodes.items(),key = lambda item:item[1],reverse = True)
        sorted_nodes = {key: value for key, value in sorted_items}
    root,value = sorted_nodes.popitem()
    return root

def assignCodeWord(root, code_word=''):
    """
    recursively assigns code-word to the nodes in the huffman tree
      parameter:
        root       : root node of the huffman tree
        code_word  : code-word for the root node
      return:
        no return
    """
    #recursive
    root.setCodeWord(code_word)
    if root.getSymbol() is None:
        left_code = code_word + '0'
        assignCodeWord(root.getLeft(),left_code)
        right_code = code_word + '1'
        assignCodeWord(root.getRight(),right_code)


def huffmanEncode(message):
    """
    converts the input message into huffman code
      parameter:
        message    : input message string
      return:
        a tuple, the first element is the huffman code string for the input message,
        the second element is the huffman tree root node
    """
    encode = ''
    word_dict = buildDictionary(message)
    root = buildHuffmanTree(word_dict)
    #if root.getLeft() is not None or root.getRight() is not None:

    assignCodeWord(root,'')
    leaves = return_leaves(root)
    
    for i in message:
        for leaf in leaves:
            if i == leaf.getSymbol():
                code = leaf.getCodeWord()
                encode += code
                break
    return encode,root



def huffmanDecode(message, huffman_tree):
    """
    decode the message
      parameter:
        message      : input huffman code string
        huffman_tree : huffman tree root node
      return:
        decoded message
    """
    decode = ''
    temp = ''
    code = ''
    if message =='':
        code = huffman_tree.getSymbol()
        decode = code*huffman_tree.getCount()
        return decode
    leaves = return_leaves(huffman_tree)
    for i in message:
        temp += i
        for leaf in leaves:
            if temp == leaf.getCodeWord():
                code = leaf.getSymbol()
        
                decode += code
                temp = ''
    return decode
        

    

def main():
    """
    main process goes here
    """
    message = input("Enter a message: ")
    encoded, rootNode = huffmanEncode(message)
    decoded = huffmanDecode(encoded, rootNode)
    print("Encode the message, and the huffman code is: ", encoded)
    print("Huffman code's length is: ", len(encoded))
    print("Decode the huffman code, and the decoded message is: ", decoded)

##############################
# FINISH THE ABOVE FUNCTIONS #
##############################


###########################
# DO NOT MODIFY THIS PART #
###########################
if __name__ == "__main__":
    main()
###########################
# DO NOT MODIFY THIS PART #
###########################