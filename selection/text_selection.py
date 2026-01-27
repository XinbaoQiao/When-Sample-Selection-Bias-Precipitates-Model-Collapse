"""
Text Data Selection Methods
"""
import torch
import numpy as np
from typing import List, Tuple, Optional

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. ROUGE-based methods will not work.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Perplexity-based methods will not work.")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not available. BERTScore-based methods will not work.")

import torch.nn.functional as F


class RougeFilter:
    """Selection method based on ROUGE score"""
    def __init__(self, reference_texts: List[str], rouge_type: str = "rouge1"):
        if not ROUGE_AVAILABLE:
            raise ImportError("RougeFilter requires rouge_score. Install with: pip install rouge-score")

        self.reference_texts = reference_texts
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    def compute_rouge_scores(self, candidate_texts: List[str]) -> np.ndarray:
        """Calculate ROUGE scores between candidate and reference texts"""
        scores = []
        
        for candidate in candidate_texts:
            # Calculate ROUGE with all reference texts, take maximum
            max_score = 0.0
            for ref in self.reference_texts:
                score = self.scorer.score(ref, candidate)[self.rouge_type].fmeasure
                max_score = max(max_score, score)
            scores.append(max_score)
        
        return np.array(scores)
    
    def select(self, texts: List[str], selection_ratio: float = 0.3) -> Tuple[List[str], torch.Tensor]:
        """Select samples with highest ROUGE scores"""
        scores = self.compute_rouge_scores(texts)
        
        num_select = int(len(texts) * selection_ratio)
        indices = np.argsort(-scores)[:num_select]  # Negative for descending sort
        
        selected_texts = [texts[i] for i in indices]
        selected_indices = torch.tensor(indices)
        
        return selected_texts, selected_indices


class PerplexityFilter:
    """Selection method based on perplexity"""
    def __init__(self, model, tokenizer, device="cuda"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("PerplexityFilter requires transformers. Install with: pip install transformers")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Calculate perplexity of texts"""
        perplexities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Encode texts
            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                
                # Perplexity = exp(loss)
                perplexity = torch.exp(loss).cpu().item()
                perplexities.extend([perplexity] * len(batch_texts))
        
        return np.array(perplexities)
    
    def select(self, texts: List[str], selection_ratio: float = 0.3) -> Tuple[List[str], torch.Tensor]:
        """Select samples with lowest perplexity"""
        perplexities = self.compute_perplexity(texts)
        
        num_select = int(len(texts) * selection_ratio)
        indices = np.argsort(perplexities)[:num_select]  # Select lowest perplexity
        
        selected_texts = [texts[i] for i in indices]
        selected_indices = torch.tensor(indices)
        
        return selected_texts, selected_indices


class BERTScoreFilter:
    """Selection method based on BERTScore"""
    def __init__(self, reference_texts: List[str], lang: str = "en"):
        if not BERTSCORE_AVAILABLE:
            raise ImportError("BERTScoreFilter requires bert-score. Install with: pip install bert-score")

        self.reference_texts = reference_texts
        self.lang = lang

    def compute_bertscore(self, candidate_texts: List[str]) -> np.ndarray:
        """Calculate BERTScore between candidate and reference texts"""
        # For each candidate, calculate BERTScore with all references, take maximum
        scores = []

        for candidate in candidate_texts:
            max_score = 0.0
            for ref in self.reference_texts:
                P, R, F1 = bert_score(
                    [candidate],
                    [ref],
                    lang=self.lang,
                    verbose=False
                )
                max_score = max(max_score, F1.item())
            scores.append(max_score)

        return np.array(scores)

    def select(self, texts: List[str], selection_ratio: float = 0.3) -> Tuple[List[str], torch.Tensor]:
        """Select samples with highest BERTScore"""
        scores = self.compute_bertscore(texts)

        num_select = int(len(texts) * selection_ratio)
        indices = np.argsort(-scores)[:num_select]

        selected_texts = [texts[i] for i in indices]
        selected_indices = torch.tensor(indices)

        return selected_texts, selected_indices


class TextMatchingFilter:
    """Selection method based on text matching (Lin et al., 2023)"""
    def __init__(self, text_feature: np.ndarray):
        """
        Args:
            text_feature: Class text embedding vector [feature_dim]
        """
        self.text_feature = text_feature / np.linalg.norm(text_feature)

    def select(self, texts: List[str], text_features: np.ndarray,
               selection_ratio: float = 0.3, used=None) -> Tuple[List[str], torch.Tensor]:
        """
        Select samples closest to class text embedding
        Implementation reference: Text_matching function
        """
        # Normalize generated features
        gen_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        # Calculate cosine similarity (as negative distance)
        similarities = np.dot(gen_features, self.text_feature)

        # Handle used samples
        if used is not None:
            similarities[used] = -np.inf

        # Select samples with highest similarity
        num_select = int(len(texts) * selection_ratio)
        indices = np.argsort(-similarities)[:num_select]  # Negative for descending sort

        selected_texts = [texts[i] for i in indices]
        selected_indices = torch.tensor(indices)

        return selected_texts, selected_indices


class TextSamplingFilter:
    """Selection method based on text sampling (Lin et al., 2023)"""
    def __init__(self, text_feature: np.ndarray):
        """
        Args:
            text_feature: Class text embedding vector [feature_dim]
        """
        self.text_feature = text_feature / np.linalg.norm(text_feature)

    def select(self, texts: List[str], text_features: np.ndarray,
               selection_ratio: float = 0.3, used=None) -> Tuple[List[str], torch.Tensor]:
        """
        Sample with cosine similarity as probability
        Implementation reference: Text_sampling function
        """
        # Normalize generated features
        gen_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarities = np.dot(gen_features, self.text_feature)

        # Ensure similarity is positive
        similarities = np.clip(similarities, a_min=1e-8, a_max=None)

        # Handle used samples
        if used is not None:
            similarities[used] = 0

        # Convert to probability distribution
        probabilities = similarities / np.sum(similarities)

        # Sample
        num_select = int(len(texts) * selection_ratio)
        indices = np.random.choice(len(texts), size=num_select, replace=False, p=probabilities)

        selected_texts = [texts[i] for i in indices]
        selected_indices = torch.tensor(indices)

        return selected_texts, selected_indices

