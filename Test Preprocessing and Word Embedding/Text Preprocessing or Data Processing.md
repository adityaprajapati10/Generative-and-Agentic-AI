Text processing wise we are going to use a library called as Natural Language Toolkit(NLTK). NLTK is not the only library in the market there is an alternative of library called as spacey, called as Stanford NLP. there are so different kinds of libraries which is available, which will help you out to do text processing.

---
### Why Data Processing?

In every NLP project or any machine learning model that uses textual input, data processing is a crucial step. Before feeding the text into a model, you need to clean and transform it into a suitable format. This involves several steps such as:

- **Tokenization**: Breaking down the text into smaller units like words or characters.
    
- **Stop Word Removal**: Eliminating common words (e.g., "is", "the", "and") that may not contribute much to the model's understanding.
    
- **Punctuation Removal**: Stripping out punctuation marks that are not useful for analysis.
    
- **Normalization**: Converting text to a consistent format, such as lowercase.
    
- **Stemming or Lemmatization**: Reducing words to their base or root form (e.g., "running" → "run").
    

These steps help in reducing noise, improving efficiency, and ensuring that the model focuses on meaningful patterns in the data.

-----
### Why Preprocessing is Important ?

In any NLP or machine learning project that uses text, **data preprocessing is the very first and most important step**.  
We deal with raw text, but models can’t understand plain text — they need **numbers**.

So we need to do a lot of **pre-processing** on the text data to **clean and structure it**, and then **convert it into numerical representation**, known as **embedding**.

As someone working in NLP or Generative AI, your entire journey starts with **text processing** — and that's what you'll spend a lot of time doing!

Before using text in any NLP or machine learning model, it must be **cleaned and converted into a numerical format**.  
One of the most common ways to represent text numerically is through **embeddings**.

But before reaching the embedding stage, several preprocessing steps are crucial. These steps reduce noise, extract meaning, and ensure that the model focuses on the right patterns in the data.

---

### Key Text Preprocessing Steps :

#### 1. Tokenization
- Breaks down text into smaller units (*tokens*).
- Tokens can be **words**, **characters**, or **sub words**.
- **Example**:  Input: "NLP is fun!"  |  Output: "NLP", "is", "fun", "!"
#### 2. Stop Word Removal
- Removes **commonly used words** (e.g., *the, is, in, and*) that do not add meaningful information.
- Helps models focus on more significant words.
#### 3. Stemming and Lemmatization

##### ➤ Stemming
- Reduces words to their **root** by removing suffixes.
- May produce non-dictionary words.
- **Tool**: `PorterStemmer` (from NLTK)
- **Example**: Input: "running", "runner"  |  Output: "run", "runner"

#### ➤ Lemmatization
- Reduces words to their **dictionary base form** (lemma).
- More accurate than stemming, considers part of speech.
- **Tool**: `WordNetLemmatizer` (from NLTK)
- **Example**: Input: "running"  |  Output: "run"

### 4. Part-of-Speech (POS) Tagging
- Assigns grammatical labels like noun, verb, adjective, etc.
- Useful for context understanding and improving lemmatization.
- **Example**: 
  Input: "He is playing football"  
  Output: [("He", PRP), ("is", VBZ), ("playing", VBG), ("football", NN)]

### 5. NER (Name Entity Recognition )
- Identifies and classifies **named entities** in text into predefined categories such as:
  - Person (PER)
  - Organization (ORG)
  - Location (LOC)
  - Date, Time, Money, Percent, etc.
  
Helps in extracting **real-world entities** from text, enabling tasks like:
  - Information retrieval
  - Question answering
  - Text summarization

**Example:**
Input: `"Barack Obama was the president of the United States."`  
Output: [
		("Barack Obama", PERSON),  
		("United States", LOCATION)  
       ]

##### Common Entity Types

| Entity Type | Description                | Example             |
|-------------|----------------------------|---------------------|
| PERSON      | Names of people            | Elon Musk, Ram      |
| ORG         | Organizations, companies   | Google, WHO         |
| LOC         | Locations, cities, countries| Delhi, India        |
| DATE        | Dates                      | 1st July, 2025      |
| TIME        | Time                       | 5 PM, 10:30 AM      |
| MONEY       | Monetary values            | $100, ₹500          |
| GPE         | Geo-political entities     | USA, China          |


---
