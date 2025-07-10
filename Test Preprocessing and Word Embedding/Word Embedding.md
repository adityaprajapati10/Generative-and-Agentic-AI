# Word Embedding

After preprocessing, the next step is to **convert text (tokens or sentences) into numerical vectors** so that ML models can work with them. This process is called **embedding**.

---
## Common embedding methods:

- **One-Hot Encoding**
- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Word2Vec**
- (Others include GloVe, FastText, BERT etc.)

---
## **One-Hot Encoding

**One-Hot Encoding** is a basic technique to represent words as vectors.

Each **unique word** in the corpus is assigned an **index**, and the word is represented as a vector of zeros with a **1** at its assigned index.

#### How It Works
1. Combine all sentences into a single text.
2. Split into individual words.
3. Create a **vocabulary** (set of unique words).
4. Assign an index to each word.
5. For each word in a sentence, create a vector of length equal to the vocabulary size, placing `1` at the word’s index.

##### Example

```python
corpus = ["I love programming", "Python is great", "I love Python and programming"]

# Step 1: Create unique words
unique_words = list(set(" ".join(corpus).split()))
# Example Output: ['is', 'love', 'programming', 'and', 'I', 'great', 'Python']

# Step 2: Assign indices
word_to_index = {word: i for i, word in enumerate(unique_words)}
# {'is': 0, 'love': 1, 'programming': 2, 'and': 3, 'I': 4, 'great': 5, 'Python': 6}

# Step 3: Create One-Hot Vectors
one_hot_vectors = []
for sentence in corpus:
    sentence_vector = []
    for word in sentence.split():
        vector = [0] * len(unique_words)
        vector[word_to_index[word]] = 1
        sentence_vector.append(vector)
    one_hot_vectors.append(sentence_vector)
```

one_hot_vectors =
[
  [ [0,1,0,0,1,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0] ],
  [ [0,0,0,0,0,0,1], [1,0,0,0,0,0,0], [0,0,0,0,0,1,0] ],
  [ [0,0,0,0,1,0,0], [0,1,0,0,0,0,0], [0,0,0,0,0,0,1], [0,0,0,1,0,0,0], [0,0,1,0,0,0,0] ]
]

#### Limitations of One-Hot Encoding

###### 1. **High Dimensionality**
- The size of the one-hot vector equals the number of **unique words (vocabulary size)**.
- If the vocabulary has 10,000 words, each vector will be of length 10,000 — mostly filled with zeros.
- ➤ **Leads to sparse vectors**, wasting memory and computational resources.
###### 2. **No Semantic Meaning**
- All words are treated as **completely unrelated**.
- For example:
  - "king" and "queen" → No relationship.
  - "India" and "Delhi" → Treated as completely different.
- ➤ One-hot encoding cannot capture **context or similarity**.
###### 3. **No Context Awareness**
- Each word is encoded **independently of its position or surrounding words**.
- Example: "bank" in “river bank” vs. “money bank” → one-hot cannot distinguish meaning.
###### 4. **Fixed Vocabulary**
- One-hot encoding requires a **fixed vocabulary**.
- Any **new or unseen words** cannot be encoded without retraining the vocabulary.
----

## **Bag of Words (BoW)

**Bag of Words (BoW)** is a **frequency-based encoding technique** used in NLP to convert text data into numerical vectors.

- It represents text as the **frequency of words** in a document.
- The position or order of words is **not** considered.
- The result is a **vector** where each dimension corresponds to a **word** in the vocabulary, and the value is the number of times it appears.

### How It Works

1. Create a **vocabulary** of all unique words from the corpus.
2. Each sentence or document is converted into a vector of word counts.
3. The final output is a **matrix** where:
   - Rows = sentences/documents
   - Columns = word counts for each word in the vocabulary

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love programming", "Python is great", "I love Python and programming"]

# Create the vocabulary manually
unique_words = list(set(" ".join(corpus).lower().split()))

# Initialize CountVectorizer with the custom vocabulary
vectorizer = CountVectorizer(
    stop_words=None,
    lowercase=False,
    vocabulary=unique_words
)

# Fit and transform the corpus
x = vectorizer.fit_transform(corpus)

# Convert to array
x.toarray()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words=None ,lowercase=False, vocabulary=unique_words)
corpus = ["I love programming", "Python is great", "I love Python and programming"]
x = vectorizer.fit_transform(corpus)
x.toarray()
array([[0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 1]])
vectorizer.get_feature_names_out()
array(['and', 'great', 'is', 'love', 'programming', 'python'], dtype=object)

```

By default the counter vectorizer that you have created over here it holds so many different different parameter.

### CountVectorizer – Important Parameters

`CountVectorizer()` offers several useful parameters:

- `stop_words='english'` → Automatically removes English stop words.
    
- `lowercase=True` → Converts all characters to lowercase before tokenizing.
    
- `max_features=5000` → Limits vocabulary size to top N frequent words.
    
- `ngram_range=(1,2)` → Includes unigrams and bigrams.
    
- `vocabulary=` → Allows you to pass a custom vocabulary.
    
- `binary=True` → Uses 1/0 instead of actual counts (like One-Hot Encoding).
    

### Limitations of Bag of Words

- Ignores word order and context.
    
- Results in **sparse** and **high-dimensional vectors**.
    
- Cannot distinguish between **synonyms** or **semantically similar words**.
    
- Affects performance on large text corpora due to memory consumption.
	
- No Context Awareness



---

## **TF-IDF (Term Frequency-Inverse Document Frequency)

**TF-IDF** is a numerical statistic that reflects how important a word is to a document in a collection (corpus).

It is one of the most widely used techniques in **text vectorization**, especially in **text classification** and **information retrieval**.

TF = Number of times word appeared in document / Total number of word in document.
IDF = log(Total number of document / Number of document containing word)

TF-IDF = TF * IDF

#### Example :-

**Sentence**: _"My name is Ram", "I teach AI."_

Let’s say the total number of words in the document is 4(My name is Ram).

- **TF(my)** = 1 / 4  
- **IDF(my)** = log(2 / 1) → appears in only 1 out of 2 documents  
- So, **TF-IDF(my)** = (1/4) × log(2/1)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love nlp",
    "I teach gen ai",
    "I am working with euron"
]

# Initialize vectorizer
vect_tf_idf = TfidfVectorizer()

# Fit and transform
X = vect_tf_idf.fit_transform(corpus)

# Convert to array
X.toarray()
```

```
Sample Output

array([
 [0. , 0. , 0. , 0. , 0.707, 0.707, 0. , 0. , 0. ],
 [0.577, 0. , 0. , 0.577, 0. , 0. , 0.577, 0. , 0. ],
 [0. , 0.5 , 0.5 , 0. , 0. , 0. , 0. , 0.5 , 0.5 ]
])

```

### Why TF-IDF Is Useful

- If a word appears in **every document**, it becomes less useful — TF-IDF **reduces its weight**.
    
- If a word appears **rarely but importantly**, it gets **higher weight**.
    
- Helps reduce the impact of **common words** (similar to stop word removal).
    
- So even **without explicitly removing stop words**, TF-IDF can down-weight them effectively.

### Limitations of TF-IDF

- **No Context Awareness**: Doesn’t understand word meaning or relationships.
    
- **Word Order is Ignored**: “AI teaches me” vs. “I teach AI” → same vector space.
    
- **No Semantic Similarity**: “car” and “vehicle” treated as unrelated.
    
- **Computationally Expensive**: Especially with a large vocabulary or corpus.


---

## **Word2Vec (Word to Vector) **


**Word2Vec** is a powerful word embedding technique developed by **Google** that learns **semantic relationships** between words.

It is based on a **neural network**, so technically, it's a **shallow two-layer neural network** trained on a large corpus of text.

Unlike BoW or TF-IDF, Word2Vec **preserves context and meaning**. It maps each word to a **dense vector in a continuous vector space**, where **similar words have similar vectors**.


There are **two training architectures**:

### 1. CBOW (Continuous Bag of Words)
	
- Predicts the **current word** based on surrounding context (window).
	
- **Faster** and works better with **smaller datasets**.
	
- Good when you're trying to **guess the missing word**.

**Example:**  
		Input: ["I", "__", "AI"]  
        Model learns: "love" fits best in the context  
        Output: "love"

### 2. Skip-Gram
	
- Given the **current word**, predicts **surrounding words**.
	
- **Slower** but performs better on **rare words** and **larger datasets**.
	
- Useful when trying to understand **influence of a word**.

**Example:** 
		Input: "teach"  
		Output: ["I", "AI"] ← predicted context words

### Example Corpus and Explanation

**Corpus**: "I love teaching AI"

Using a window size of 1:

| CBOW Input         | Target     |
|--------------------|------------|
| [I, teaching]      | love       |
| [love, AI]         | teaching   |

| Skip-Gram Input | Output       |
|----------------|--------------|
| love           | [I, teaching]|
| teaching       | [love, AI]   |

### Why Word2Vec?

- Embeddings capture **syntactic and semantic similarity**.
- "king" - "man" + "woman" ≈ "queen"
- Words with **similar meaning are close** in vector space.
- Useful for **search engines**, **recommendation systems**, **chatbots**, and **translation models**.
### Limitations

- Requires **large data** for meaningful embeddings.
- Doesn't handle **out-of-vocabulary (OOV)** words.
- Context is **fixed and symmetric**, unlike modern models like BERT.

---
