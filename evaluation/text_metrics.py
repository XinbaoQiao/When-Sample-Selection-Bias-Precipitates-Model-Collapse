"""
Text evaluation metrics
"""
import torch
import numpy as np
from typing import List, Dict, Tuple

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. ROUGE metrics will not work.")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not available. BERTScore metrics will not work.")

try:
    import mauve
    MAUVE_AVAILABLE = True
except ImportError:
    MAUVE_AVAILABLE = False
    print("Warning: mauve not available. MAUVE metrics will not work.")

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def compute_rouge(
    generated_texts: List[str],
    reference_texts: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
) -> Dict[str, float]:
    if not ROUGE_AVAILABLE:
        print("Warning: rouge_score not available, returning dummy ROUGE scores")
        return {rouge_type: 0.0 for rouge_type in rouge_types}
    """
    Compute ROUGE score
    
    Args:
        generated_texts: Generated text list
        reference_texts: Reference text list
        rouge_types: ROUGE type list
    
    Returns:
        ROUGE score dictionary
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = {rouge_type: [] for rouge_type in rouge_types}
    
    for gen_text, ref_text in zip(generated_texts, reference_texts):
        score = scorer.score(ref_text, gen_text)
        for rouge_type in rouge_types:
            scores[rouge_type].append(score[rouge_type].fmeasure)
    
    # Compute mean
    result = {rouge_type: np.mean(scores[rouge_type]) for rouge_type in rouge_types}
    
    return result


def compute_bertscore(
    generated_texts: List[str],
    reference_texts: List[str],
    lang: str = "en"
) -> Dict[str, float]:
    if not BERTSCORE_AVAILABLE:
        print("Warning: bert_score not available, returning dummy BERTScore")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    """
    Compute BERTScore
    
    Args:
        generated_texts: Generated text list
        reference_texts: Reference text list
        lang: Language code
    
    Returns:
        BERTScore dictionary (P, R, F1)
    """
    P, R, F1 = bert_score(generated_texts, reference_texts, lang=lang, verbose=False)
    
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item())
    }


def compute_mauve(
    generated_texts: List[str],
    reference_texts: List[str],
    max_text_length: int = 512
) -> Dict[str, float]:
    if not MAUVE_AVAILABLE:
        print("Warning: mauve not available, returning dummy MAUVE score")
        return {"mauve": 0.0}
    """
    Compute MAUVE score
    
    MAUVE measures divergence between generated and human text distributions
    
    Args:
        generated_texts: Generated text list
        reference_texts: Reference text list
        max_text_length: Max text length
    
    Returns:
        MAUVE score dictionary
    """
    # Truncate text
    gen_truncated = [text[:max_text_length] for text in generated_texts]
    ref_truncated = [text[:max_text_length] for text in reference_texts]
    
    # Compute MAUVE
    out = mauve.compute_mauve(
        p_text=gen_truncated,
        q_text=ref_truncated,
        device_id=0,
        verbose=False
    )
    
    return {
        "mauve": float(out.mauve),
        "frontier_integral": float(out.frontier_integral)
    }


def compute_entropy(
    texts: List[str],
    n: int = 1
) -> Dict[str, float]:
    """
    Compute n-gram entropy
    
    Args:
        texts: Text list
        n: n for n-gram (1=unigram, 2=bigram, etc.)
    
    Returns:
        Entropy dictionary
    """
    # Extract n-grams
    ngrams = []
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
    
    # Compute frequency
    counter = Counter(ngrams)
    total = len(ngrams)
    
    # Compute entropy
    probs = [count / total for count in counter.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    
    # Normalized entropy (divide by max possible entropy)
    max_entropy = np.log(len(counter))
    normalized_entropy = entropy / (max_entropy + 1e-10) if max_entropy > 0 else 0.0
    
    return {
        f"{n}gram_entropy": float(entropy),
        f"{n}gram_normalized_entropy": float(normalized_entropy),
        f"unique_{n}grams": len(counter),
        f"total_{n}grams": total
    }


def compute_topic_shift(
    generated_texts: List[str],
    reference_texts: List[str],
    num_topics: int = 10
) -> Dict[str, float]:
    """
    Compute Topic Shift
    
    Use LDA to detect topic distribution differences between generated and reference texts
    
    Args:
        generated_texts: Generated text list
        reference_texts: Reference text list
        num_topics: Number of topics
    
    Returns:
        Topic shift metrics
    """
    all_texts = generated_texts + reference_texts
    labels = [0] * len(generated_texts) + [1] * len(reference_texts)
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(all_texts)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_distributions = lda.fit_transform(X)
    
    gen_topics = topic_distributions[:len(generated_texts)]
    ref_topics = topic_distributions[len(generated_texts):]
    
    gen_avg_dist = gen_topics.mean(axis=0)
    ref_avg_dist = ref_topics.mean(axis=0)
    
    kl_div = np.sum(ref_avg_dist * np.log((ref_avg_dist + 1e-10) / (gen_avg_dist + 1e-10)))
    
    # Compute JS divergence (symmetric KL)
    m = (gen_avg_dist + ref_avg_dist) / 2
    js_div = 0.5 * (
        np.sum(gen_avg_dist * np.log((gen_avg_dist + 1e-10) / (m + 1e-10))) +
        np.sum(ref_avg_dist * np.log((ref_avg_dist + 1e-10) / (m + 1e-10)))
    )
    
    return {
        "kl_divergence": float(kl_div),
        "js_divergence": float(js_div),
        "gen_topic_dist": gen_avg_dist.tolist(),
        "ref_topic_dist": ref_avg_dist.tolist()
    }

