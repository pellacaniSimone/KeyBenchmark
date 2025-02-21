# KeyBenchmark

Another benchmark for the keyword extraction algorithm; both parametric and non-parametric.

# README  

## Keyword Extraction Benchmark  

This project provides a benchmark for keyword extraction methods using various techniques, including:  

- **RAKE** (Rapid Automatic Keyword Extraction)  
- **spaCy** (Linguistic parsing and Named Entity Recognition)  
- **YAKE** (Yet Another Keyword Extractor)  
- **Mini LLM** (Extracting keywords using a lightweight language model via an external Ollama API)  

The benchmark evaluates how well these techniques extract keywords compared to a ground truth dataset.  

---

## Features  

- Loads a dataset (`.csv`) containing text samples and their corresponding ground truth keywords.  
- Extracts keywords using four different methods.  
- Computes similarity scores between extracted and expected keywords.  
- Visualizes the performance of different extraction techniques using plots.  

---

## Requirements  

### Python Dependencies  

Install dependencies using `pip`:  

```bash
pip install pandas spacy yake nltk rake_nltk matplotlib requests pydantic
```

### Additional Setup  

Download and install **spaCy** language models:  

```bash
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_md
```

---

## Usage  

1. Ensure your dataset is in a CSV format with at least the following columns:  
   - `Risposta` (The text from which keywords should be extracted)  
   - `keywords` (The expected keywords for evaluation)  

2. Run the benchmark script:  

```bash
python benchmark.py
```

The script will:  
- Process each record in the dataset  
- Extract keywords using the different methods  
- Compute similarity scores  
- Display results and generate a performance comparison plot  

---

## API Integration  

The script communicates with an **Ollama API** for mini LLM-based keyword extraction. Make sure to replace `http://yourOllamaServerHere.lan:11434` with the actual API endpoint.  

---

## Output  

The script prints the extracted keywords for each record along with similarity scores:  

```plaintext
Record 0:
Expected                 : {'keywords': ['nutrition', 'food well-being', 'balanced diet', 'essential nutrients', 'portions control', 'mindful eating', 'chronic diseases', 'food labels', 'hydration', 'whole foods', 'optimal health']}
RAKE                     (0.389): {'keywords': ['ensuring adequate fluid intake supports overall health', 'includes reading nutritional information', 'concept revolves around consuming', 'achieve optimal nutritional status', 'encompass making conscious choices', 'body receives adequate vitamins', 'group offers unique nutrients', 'adopting balanced dietary habits']}
spaCy                    (0.515): {'keywords': ['integral aspects', 'essential nutrients', 'bodily functions', 'overall health', 'proper nutrition', 'adequate vitamins', 'minerals', 'proteins']}
YAKE                     (0.584): {'keywords': ['food well-being', 'food', 'well-being', 'Nutrition', 'healthy lifestyle', 'health', 'integral aspects', 'diet']}
miniLLM qwen2.5:0.5b     (0.559): {'keywords': ['balanced diet', 'essential nutrients', 'immunization system', 'heart disease', 'diabetes', 'certain cancers', 'fruit and vegetables', 'healthy fats']}
```

A plot is also generated comparing the performance of different extraction methods.  

---

## Use case  

Developed for benchmarking keyword extraction techniques using both traditional NLP methods and modern lightweight LLMs.


## Result in 1 iteration
LLM result may vary
```plaintext
Overall Results:
RAKE    : 0.408
spaCy   : 0.546
YAKE    : 0.565
miniLLM : 0.622
````