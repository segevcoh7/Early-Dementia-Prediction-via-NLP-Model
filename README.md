# Distinguishing Dementia Using Hebrew NLP and LLM Models

This project focuses on developing a machine learning pipeline to distinguish early-stage dementia among Hebrew-speaking individuals. Using Natural Language Processing (NLP) techniques and Large Language Models (LLMs), the goal is to analyze speech patterns and identify linguistic markers associated with dementia. The approach incorporates a distribution and variety of speech characteristics from diverse individuals.

---

## Key Objectives

1. **Data Collection**: Gather a diverse dataset of speech samples from Hebrew-speaking individuals, including those diagnosed with dementia and healthy controls.
2. **Feature Extraction**: Use NLP techniques to extract linguistic features such as syntax, semantics, fluency, and speech dynamics.
3. **Model Development**: Fine-tune Hebrew-compatible LLMs to classify dementia-related speech patterns.
4. **Evaluation**: Measure model accuracy and generate insights into linguistic differences between healthy individuals and those with dementia.

---

## Repository Structure


---

## Steps in the Process

1. **Data Collection**:
   - Collect speech samples from clinical datasets (dementia patients) and healthy individuals.
   - Convert audio files to text using Hebrew-compatible speech-to-text systems.

2. **Data Preprocessing**:
   - Clean and normalize Hebrew text for NLP tasks.
   - Tokenize, lemmatize, and tag parts of speech using Hebrew NLP tools.

3. **Feature Engineering**:
   - Extract linguistic features:
     - **Lexical Features**: Vocabulary richness, word frequency.
     - **Syntactic Features**: Grammar patterns and sentence structure.
     - **Semantic Features**: Contextual coherence and sentiment analysis.
     - **Prosodic Features**: Speech rate and intonation (if audio is used).

4. **Model Development**:
   - Fine-tune Hebrew-compatible LLMs:
     - **AlephBERT**, **HeBERT**, or multilingual models like **mBERT** and **XLM-RoBERTa**.
   - Train models to classify dementia-related speech patterns.

5. **Model Evaluation**:
   - Evaluate models using metrics like Accuracy, Precision, Recall, and F1-score.
   - Use t-SNE or PCA to visualize differences between dementia and non-dementia speech.

---

## Tools and Technologies

- **Programming**: Python
- **NLP Libraries**: `transformers`, `spaCy`, `nltk`
- **Speech-to-Text**: Google Speech API, Wav2Vec2 for Hebrew
- **Machine Learning**: TensorFlow, PyTorch, scikit-learn
- **Visualization**: Matplotlib, Seaborn

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Dementia-NLP-Hebrew.git
