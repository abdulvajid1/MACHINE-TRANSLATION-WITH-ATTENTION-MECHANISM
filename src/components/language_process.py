import torch

class LangPorcess:
    def __init__(self):

        self.__word2int = {}
        self.__int2word = {}
        self.__vocabsize = 2
        self.__word2int['<pad>'] = 0
        self.__int2word[0] = '<pad>'
        self.__int2word[1] = '<eos>'
        self.__word2int['<eos>'] = 1
        self.__word_freq = {}
        self.__integer_encoded =[]

    def fit(self,X):
        X_split_words = [text.split(' ') for text in X]
        self.read_sentences(X_split_words)
        self.__integer_encoding(X_split_words)
        
        
    def read_sentences(self,X):
        # Loop thourgh each sentence and read words
        for sentence in X:
            for word in sentence:
                if word not in self.__word2int.keys():
                    self.__word_freq[word] = 1
                    self.read_words(word)
                else:
                    self.__word_freq[word] += 1


    
    def read_words(self,word):
            self.__word2int[word] = self.__vocabsize
            self.__int2word[self.__vocabsize] = word
            self.__vocabsize+=1

    def get_word_frequency(self):
        return dict(sorted(self.__word_freq.items(),
                           key=lambda x:x[1],
                           reverse=True))
    
    def get_word2index(self):
        return self.__int2word
    
    def get_index2word(self):
        return self.__word2int
    
    def __integer_encoding(self,X):  
        for word in X:
            torch_tensor = torch.tensor(list(map(lambda word : self.__word2int[word],word)))
            self.__integer_encoded.append(torch_tensor)

    def get_integer_encoding(self,padding=True,max_len=None):
        if padding:
            return self.__pad_sequence(self.__integer_encoded,max_len)
        else:
            return self.__integer_encoded
        
    def __pad_sequence(self,X,max_len):
            maxlen_padded = torch.nn.utils.rnn.pad_sequence(self.__integer_encoded,padding_value=0,batch_first=True)
            if max_len:
                return maxlen_padded[:,:max_len]
            else:
                return maxlen_padded
    
    def get_vocabsize(self):
        return self.__vocabsize
       
        