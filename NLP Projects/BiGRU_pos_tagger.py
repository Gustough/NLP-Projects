import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim.lr_scheduler as lr_scheduler
import pyconll
from sklearn.model_selection import train_test_split
# Libraries: torch, pyconll, scikit-learn
#Run time for swiss german about 30 seconds, for english about 3 minutes

device = torch.device("cpu")

class BiGRU(nn.Module):
    """
    Bidirectional GRU model for sequence tagging.

    Args:
        vocab_size (int): Number of unique words in the vocabulary.
        hidden_size (int): Number of hidden units in the GRU layer.
        num_classes (int): Number of unique output classes (POS tags).
        embedding_dim (int, optional): Dimensionality of word embeddings. Defaults to 100.
        num_layers (int, optional): Number of GRU layers. Defaults to 1.
    """  
    def __init__(self, vocab_size, hidden_size, num_classes, embedding_dim=100, num_layers=1):
        super(BiGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, 
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # If Bidirectional -> *2

    def forward(self, x, lengths):
        """
        Forward pass of the BiGRU model.

        Args:
            x (Tensor): Input tensor containing word indices.
            lengths (Tensor): Lengths of each sequence in the batch.

        Returns:
            Tensor: Output tensor containing class scores.
        """
        x = self.embedding(x)

        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_out, _ = self.gru(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.fc(out) 


class POSDataset(Dataset):
    """
    Dataset class for Part-of-Speech tagging.

    Args:
        sentences (list): List of tokenized sentences.
        tags (list): Corresponding list of POS tags.
        word_to_idx (dict): Mapping from words to unique indices.
        tag_to_idx (dict): Mapping from POS tags to unique indices.
    """
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
        # Unseen words/tags are mapped to index of 'UNKNOWN' in word_to_idx or tag_to_idx
        self.sentences = [torch.tensor([word_to_idx.get(w, word_to_idx['UNKNOWN']) for w in s], dtype=torch.long) for s in sentences] 
        self.tags = [torch.tensor([tag_to_idx.get(t, tag_to_idx['UNKNOWN']) for t in t_seq], dtype=torch.long) for t_seq in tags] 

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

def collate_fn(batch):
    """
    Function for DataLoader to handle padding.

    Args:
        batch (list): List of (sentence, tag) tuples.

    Returns:
        tuple: Padded sentences, padded tags, and lengths of original sequences.
    """
    sentences, tags = zip(*batch)
    lengths = [len(seq) for seq in sentences]

    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1) 

    return sentences_padded.to(device), tags_padded.to(device), torch.tensor(lengths, dtype=torch.long).to(device)

class POSTagger:
    """
    POS Tagger class for training and evaluation.

    Args:
        data_path (str): Path to the training dataset file.
    """
    def __init__(self, data_path):
        self.file = pyconll.load_from_file(data_path)
        self.sentences = []
        
        for sentence in self.file:
            words = [word.form for word in sentence]
            tags = [word.upos for word in sentence]
            self.sentences.append((words, tags))

        # Split into train & test
        self.train_data, self.test_data = train_test_split(self.sentences, test_size=0.2, random_state=1)
        self.train_words, self.train_tags = zip(*self.train_data)
        self.test_words, self.test_tags = zip(*self.test_data)

        self.build_mappings()

    def build_mappings(self):
        """
        Build mappings from words/tags to unique indices.
        """
        flat_words = [word for sentence in self.train_words for word in sentence]
        self.word_to_idx = {word: idx for idx, word in enumerate(set(flat_words))}
        self.word_to_idx['UNKNOWN'] = len(self.word_to_idx)

        flat_tags = [tag for sentence in self.train_tags for tag in sentence]
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(set(flat_tags))}
        self.tag_to_idx['UNKNOWN'] = len(self.tag_to_idx)
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        self.num_classes = len(self.tag_to_idx)

    def train(self, learning_rate=0.01, epochs=16, batch_size=26, hidden_size=64):
        """
        Train the BiGRU model on the dataset.

        Returns:
            BiGRU: Trained model.
        """
        train_dataset = POSDataset(self.train_words, self.train_tags, self.word_to_idx, self.tag_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        test_dataset = POSDataset(self.test_words, self.test_tags, self.word_to_idx, self.tag_to_idx)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        model = BiGRU(self.vocab_size, hidden_size, self.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0

            for sentences, tags, lengths in train_loader:
                optimizer.zero_grad()
                
                outputs = model(sentences, lengths)  # (batch, seq_len, num_classes)

                # Flatten for loss calculation
                outputs = outputs.view(-1, self.num_classes)
                tags = tags.view(-1)

                loss = loss_function(outputs, tags)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                
                # **Evaluate on validation set after every epoch**
                model.eval()
                total_val_loss = 0
                correct, total = 0, 0

                with torch.no_grad():
                    for sentences, tags, lengths in test_loader:
                        outputs = model(sentences, lengths)
                        outputs = outputs.view(-1, self.num_classes)
                        tags = tags.view(-1)

                        val_loss = loss_function(outputs, tags)
                        total_val_loss += val_loss.item()

                        predictions = outputs.argmax(dim=-1)
                        mask = tags != -1
                        correct += (predictions[mask] == tags[mask]).sum().item()
                        total += mask.sum().item()

                train_loss = total_train_loss
                val_loss = total_val_loss 
                val_acc = correct / total if total > 0 else 0
                
            scheduler.step()

            # current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}")

        return model

    def evaluate(self, model):
        """
        Evaluate the trained model on the test dataset.
        """
        test_dataset = POSDataset(self.test_words, self.test_tags, self.word_to_idx, self.tag_to_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for idx, (sentences, tags, lengths) in enumerate(test_loader):
                outputs = model(sentences, lengths)
                predictions = outputs.argmax(dim=-1)  # (batch, seq_len)

                # Get the actual words for this sentence from self.test_words
                words = self.test_words[idx]  # Extract words for the current test sample

                # Initialize lists to store the sentence and the corresponding tags
                sentence = []
                true_tags = []
                predicted_tags = []

                for i in range(lengths.item()):
                    if tags[0, i] != -1:  # Skip padding
                        word = words[i]  # Get the word at position i
                        true_tag = self.idx_to_tag[tags[0, i].item()]
                        predicted_tag = self.idx_to_tag[predictions[0, i].item()]

                        sentence.append(word)
                        true_tags.append(true_tag)
                        predicted_tags.append(predicted_tag)

                        # Count correct predictions
                        total += 1
                        if predictions[0, i] == tags[0, i]:
                            correct += 1
                
                
                # UNCOMMENT THIS TO SEE PREDICTED TAGS
                
                # # After processing the sentence, print the results
                # print("\nSentence: " + " ".join(sentence))
                # print("True Tags: " + " ".join(true_tags))
                # print("Predicted Tags: " + " ".join(predicted_tags))

                # # Highlight correctly and incorrectly tagged words
                # for word, true_tag, predicted_tag in zip(sentence, true_tags, predicted_tags):
                #     if true_tag == predicted_tag:
                #         print(f"{word} - Correct: {predicted_tag}")
                #     else:
                #         print(f"{word} - Incorrect: {predicted_tag} (True: {true_tag})")

        accuracy = correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    data_path = sys.argv[1]
    tagger = POSTagger(data_path)
    model = tagger.train(learning_rate=0.01, epochs=16, batch_size=26)
    tagger.evaluate(model)

if __name__ == '__main__':
    main()

#BEST PERFORMANCE: learning_rate=0.01, epochs=16, batch_size=26, embeddings=100, dropout=0.3, num_layers=1, bidirectional=True, hidden_size=64
# Datasets used: gsw_uzh-ud-test.conllu & en_gentle-ud-test.conllu