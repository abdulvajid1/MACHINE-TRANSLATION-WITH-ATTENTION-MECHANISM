import re
import string

# Remove puctuations
def remove_puctuations(word):
    word = re.sub('[{}]'.format(string.punctuation),repl='',string=word)
    return word.lower()