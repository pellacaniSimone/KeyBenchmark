import pandas as pd
import json, requests, spacy, yake, ast
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from spacy.lang.it.stop_words import STOP_WORDS as STOP_WORDS_IT
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_EN
from math import log2
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from pydantic import BaseModel

class FormatLLM(BaseModel):
    keywords: list[str]

class KeywordNonParametricBenchmark:
    prompt = {
        'keywords_ita': r"""
            RITORNA SOLE PAROLE CHIAVE DAL SEGUENTE TESTO
            ------
            TESTO
            {text}
            -------
            LA LISTA DEVE CONTENERE AL MASSIMO {num_words} PAROLE CHIAVE
            """,
        'keywords_eng': r"""
            RETURN ONLY KEYWORDS FROM THE FOLLOWING TEXT
            ------
            TEXT
            {text}
            -------
            THE LIST MUST CONTAIN A MAXIMUM OF {num_words} KEYWORDS
            """
    }
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.nlp_it = spacy.load('it_core_news_sm')
        self.nlp_en = spacy.load('en_core_web_md')
        self.rake_scores = []
        self.spacy_scores = []
        self.yake_scores = []
        self.mini_llm_scores = []
        self.record_avg = []
        self.record_numbers = []
        self.model='qwen2.5:0.5b'

    def __calculate_num_words(self, text: str) -> int:
        tokens = word_tokenize(text)
        return max(3, int(log2(len(tokens))))

    def __make_ollama_request_new(self, prompt: str, request="text") -> str:
        model = self.model
        ollama_server = "http://yourOllamaServerHere.lan:11434"
        data_keyword = {"model": model, "prompt": prompt, "stream": False, "format": FormatLLM.model_json_schema()}
        data_text = {"model": model, "prompt": prompt, "stream": False}
        data = data_text if request == "text" else data_keyword
        response = requests.post(f"{ollama_server}/api/generate", json=data)
        return response.json().get("response", "No response") if response.status_code == 200 else f"Error: {response.status_code}"

    def extract_keywords_mini_LLM(self, text, language):
        num_words = self.__calculate_num_words(text)
        prompt = self.prompt['keywords_eng'] if language == "en" else self.prompt['keywords_ita']
        formatted_prompt = prompt.format(num_words=num_words, text=text)
        response = self.__make_ollama_request_new(formatted_prompt, request="keyword")
        keywords = self.parse_keywords(response)
        return {"keywords": keywords if keywords is not None else []}

    def extract_keywords_rake(self, text, language):
        rake_lang = "english" if language == "en" else "italian"
        default_params={}
        rake = Rake(language=rake_lang, **default_params )
        rake.extract_keywords_from_text(text)
        tokens = word_tokenize(text)
        num_kw = max(3, int(log2(len(tokens))))
        keywords = rake.get_ranked_phrases()
        return {"keywords": [kw.strip() for kw in keywords[:num_kw]]}
 
    def extract_keywords_spacy(self, text, language):
        nlp = self.nlp_en if language == "en" else self.nlp_it
        stop_words = STOP_WORDS_EN if language == "en" else STOP_WORDS_IT
        doc = nlp(text)
        keywords = []
        for chunk in doc.noun_chunks:
            if not any(token.is_stop for token in chunk):
                keywords.append(chunk.text.lower())
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop and token.lemma_.lower() not in stop_words:
                keywords.append(token.lemma_.lower())
                
        tokens = word_tokenize(text)
        num_kw = max(3, int(log2(len(tokens))))
        return {"keywords": list(dict.fromkeys(keywords))[:num_kw]}

    def extract_keywords_yake(self, text, language):
        kw_extractor = yake.KeywordExtractor(lan=language, n=2, dedupLim=0.7, top=10)
        keywords = kw_extractor.extract_keywords(text)
        tokens = word_tokenize(text)
        num_kw = max(3, int(log2(len(tokens))))
        keywords.sort(key=lambda x: x[1])
        return {"keywords": [kw for kw, _ in keywords[:num_kw]]}

    def calculate_similarity_score(self, expected_keywords, extracted_keywords):
        if not expected_keywords or not extracted_keywords:
            return 0.0
        expected_list = expected_keywords.get('keywords', []) if isinstance(expected_keywords, dict) else expected_keywords
        extracted_list = extracted_keywords.get('keywords', []) if isinstance(extracted_keywords, dict) else extracted_keywords
        if not expected_list or not extracted_list:
            return 0.0
        similarities = []
        for exp_kw in expected_list:
            max_sim = max(SequenceMatcher(None, exp_kw.lower(), ext_kw.lower()).ratio() 
                         for ext_kw in extracted_list)
            similarities.append(max_sim)
        return sum(similarities) / len(similarities)

    def parse_keywords(self, keywords_str):
        try:
            if isinstance(keywords_str, dict):
                return keywords_str.get('keywords', [])
            parsed = json.loads(keywords_str)
            return parsed.get('keywords', []) if isinstance(parsed, dict) else parsed
        except (json.JSONDecodeError, TypeError):
            try:
                cleaned_str = keywords_str.replace("'", '"')
                parsed = json.loads(cleaned_str)
                return parsed.get('keywords', []) if isinstance(parsed, dict) else parsed
            except:
                try:
                    parsed = ast.literal_eval(keywords_str)
                    return parsed.get('keywords', []) if isinstance(parsed, dict) else parsed
                except:
                    if isinstance(keywords_str, str):
                        cleaned = keywords_str.strip('[]" ')
                        if cleaned:
                            keywords = [k.strip().strip('"\'') for k in cleaned.split(',')]
                            return [k for k in keywords if k]
                    return []

    def run(self):
        processed_records = 0
        for idx, row in self.df.iterrows():
            language = "en" if idx >= 10 else "it"
            text = row['Risposta']
            expected_kw = {"keywords": self.parse_keywords(row['keywords'])}
            
            if not expected_kw['keywords']:
                print(f"Warning: Could not parse keywords for record {idx}")
                continue

            processed_records += 1
            
            rake_kw = self.extract_keywords_rake(text, language)
            spacy_kw = self.extract_keywords_spacy(text, language)
            yake_kw = self.extract_keywords_yake(text, language)
            mini_llm_kw = self.extract_keywords_mini_LLM(text, language)
            
            rake_sim = self.calculate_similarity_score(expected_kw, rake_kw)
            spacy_sim = self.calculate_similarity_score(expected_kw, spacy_kw)
            yake_sim = self.calculate_similarity_score(expected_kw, yake_kw)
            mini_llm_sim = self.calculate_similarity_score(expected_kw, mini_llm_kw)
            
            self.rake_scores.append(rake_sim)
            self.spacy_scores.append(spacy_sim)
            self.yake_scores.append(yake_sim)
            self.mini_llm_scores.append(mini_llm_sim)
            
            avg_point = (rake_sim + spacy_sim + yake_sim + mini_llm_sim) / 4
            self.record_avg.append(avg_point)
            self.record_numbers.append(idx)
            
            print(f"\nRecord {idx}:")
            print(f"{'Expected':<25}: {expected_kw}")
            print(f"{'RAKE':<25}({rake_sim:.3f}): {rake_kw}")
            print(f"{'spaCy':<25}({spacy_sim:.3f}): {spacy_kw}")
            print(f"{'YAKE':<25}({yake_sim:.3f}): {yake_kw}")
            print(f"{'miniLLM ' + self.model:<25}({mini_llm_sim:.3f}): {mini_llm_kw}")
            
        if processed_records > 0:
            self.plot_results()
            print("\nOverall Results:")
            print(f"RAKE    : {sum(self.rake_scores) / len(self.rake_scores):.3f}")
            print(f"spaCy   : {sum(self.spacy_scores) / len(self.spacy_scores):.3f}")
            print(f"YAKE    : {sum(self.yake_scores) / len(self.yake_scores):.3f}")
            print(f"miniLLM : {sum(self.mini_llm_scores) / len(self.mini_llm_scores):.3f}")
        else:
            print("\nNo records were successfully processed. Please check the input data format.")

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.record_numbers, self.record_avg, label='µ', marker='+', linestyle='dashdot', color='gray')
        plt.plot(self.record_numbers, self.rake_scores, label='RAKE', marker='o', linestyle='-', color='b')
        plt.plot(self.record_numbers, self.spacy_scores, label='spaCy', marker='s', linestyle='--', color='g')
        plt.plot(self.record_numbers, self.yake_scores, label='YAKE', marker='^', linestyle='-.', color='r')
        plt.plot(self.record_numbers, self.mini_llm_scores, label='miniLLM', marker='d', linestyle='-', color='purple')
        plt.title("Keyword Extraction Benchmark Results", fontsize=16)
        plt.xlabel("Record Number", fontsize=14)
        plt.ylabel("Similarity Score", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(False)
        plt.xticks(self.record_numbers)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    benchmark = KeywordNonParametricBenchmark("merged_truth_cleaned.csv")
    benchmark.run()