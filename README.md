# Quora question pairs kaggle competition

### Text Preprocessing
• Remove repeated words.

• Remove repeated emoji.

• Remove links.

• Remove punctuations.

• Remove stop words.

• Replace every emoji with its text. 

• Change back abbreviated words like (I'm --> I am).

• Lemmatization: transform words to its root.
 
### Embedding
• Replace each word by its corresponding vector in [Glove & Word2Vec] word embeddings.

### Model
• Using an encoding strategy to encode the two sentences into one vector by LSTM.

• concatenate and feed the two vectors to one node.

• calculate cosine distance and manhattan distance between the two vectors.

• concatenate the 3 nodes and feed them in the final layer with sigmoid activation.

### Results:

• loss: 0.3663 - acc: 0.8568 - val_loss: 0.5125 - val_acc: 0.7950

