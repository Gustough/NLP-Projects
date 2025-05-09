import numpy as np
from scipy.sparse import csr_array
import sys
import gzip
from collections import defaultdict
import os
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from itertools import islice

class MultiLing:
    def __init__(self, input_files: str, n_files: int):
        '''
        A constructor that takes a directory of files containing parallel language data and creates
        word embeddings for the different languages. It also selects the sentences and words that are going to
        be used from the original files. Only sentences that exist in the files for all languages are selected,
        and only words that occur at least 10 times are encoded. Self sentences and chosen words contain these
        selections. Additionally, the matrix dimensions for the sparse matrix used later are stored in the attribute
        objects self.rows and self.columns.
        
        -Arg: 
        input files:
            the directory containing the files that will be turned into WordVector objects
        n_files: 
            the number of files from the directory that are going to be used from the directory
        '''
        word_counts = defaultdict(int)
        self.chosen_words = dict()
        sentence_check = defaultdict(int)
        self.sentences = set()
        self.folder_path = Path(input_files)
        self.number_files = int(n_files)
        self.lcs = []
        
        for file in islice(self.folder_path.iterdir(), self.number_files):
            with gzip.open(file, "rt") as f:
                for line in f:
                    sentence_check[line.split()[0]] += 1

        for key, value in sentence_check.items():
            if value == self.number_files:
                self.sentences.add(key)
                
        for file in islice(self.folder_path.iterdir(), self.number_files):
            with gzip.open(file, "rt") as f:
                language_code = os.path.basename(file).split('.')[0] + "/"
                self.lcs.append(os.path.basename(file).split('.')[0])
                for line in f:
                    if line.split()[0] in self.sentences:
                        for word in line.split()[1:]:
                            word_counts[language_code + word.lower()] += 1        

        for k, v in word_counts.items():
            if v >= 10: 
                self.chosen_words[k] = v
        self.rows = len(self.chosen_words)
        self.columns = len(self.sentences)
        
    def get_lists(self):
        '''
        Constructs the lists needed for the sparse matrix creation. As all languages are 
        originally contained in the same matrix, this function gets the ranges where one language
        ends and another begins and stores them in self.language_ranges. 
        
        -Returns:
            the lists used in the sparse matrix creation: row, col and data
        '''
        self.vocab = {word: i for i, word in enumerate(self.chosen_words.keys())}
        sentence_index = {sentence_id: i for i, sentence_id in enumerate(self.sentences)}
        self.language_ranges = defaultdict(list)


        for word, index in self.vocab.items():
            lang_tag = word.split('/')[0]  # Extract language code
            self.language_ranges[lang_tag].append(index)

        for lang, indices in self.language_ranges.items():
            self.language_ranges[lang] = (min(indices), max(indices)) #range for the different languages
        
        row, col, data = [], [], []
        
        for file in islice(self.folder_path.iterdir(), self.number_files):
            with gzip.open(file, "rt") as f:
                language_code = os.path.basename(file).split('.')[0] + "/"
                
                for line in f:  # Read file line by line
                    parts = line.split()
                                        
                    sentence_id = parts[0]  # First token is sentence ID
                    if sentence_id not in sentence_index:
                        continue  # Skip sentences not in the list
                    
                    words = parts[1:]  # Skip first token (sentence ID)
                    wc = defaultdict(int)  # Dictionary for word count in sentence
                    
                    for word in words:
                        lc_word = language_code + word.lower()
                        if lc_word in self.vocab:
                            word_idx = self.vocab[lc_word]  # Column index
                            wc[word_idx] += 1

                    # Add sentence-word relationships to lists
                    for word_idx, count in wc.items():
                        row.append(word_idx)  
                        col.append(sentence_index[sentence_id])
                        data.append(count)
        
        return (np.array(data), np.array(row), np.array(col))  
    
    def make_matrix(self):
        '''
        Constructs the sparse matrix with the list obtained from the get_list function using 
        csr_array and truncatedsvd.
        
        -Returns:
            the matrix containing the word vectors 
        '''
        data, row, col = self.get_lists()
        sparse_matrix = csr_array((data, (row, col)), shape=(self.rows, self.columns), dtype=int)
        
        svd = TruncatedSVD(n_components=100)
        svd.fit(sparse_matrix)
        word_vectors = svd.transform(sparse_matrix)
        
        return word_vectors
    
    def make_word_vec(self, output_dir):
        '''
        Turn the sparse matrix into separate word vector files and prints them to the output directory.
        
        -Args:
            output_dir:
                the directory where the word vector files are going to be stored
        
        -Returns:
            Confirmation that the files have been written
        '''
        wv = self.make_matrix()
        language_vecs = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for l in self.lcs:
            start, end  = self.language_ranges[l]
            language_vecs[l] = wv[start:end + 1]

        for language, embedding in language_vecs.items():
            output_path = os.path.join(output_dir, f"{language}.vec.gz")
            with gzip.open(output_path, "wt") as file:

                file.write(f"{embedding.shape[0]} {embedding.shape[1]}\n")
                
                filtered_vocab = [(word, index) for word, index in self.vocab.items() if word.startswith(language)] 

                # print(filtered_vocab[0], filtered_vocab[-1])
                for word, index in filtered_vocab:
                    vector_str = " ".join(map(str, wv[index]))  # Convert vector to space-separated string
                    file.write(f"{word} {vector_str}\n")
        return "Done writing the embedding files!"

def main(): #Takes 3 inputs: [1] input path containing the parallel corpora, [2] output path to which the files are written,
            #number of files that are supposed to be processed in the input directory
    ip = sys.argv[1] # Input path
    op = sys.argv[2] # Output path
    number_files = sys.argv[3] #integer
    res = MultiLing(ip, number_files).make_word_vec(op)
    print(res)

if __name__ == "__main__":
    main()