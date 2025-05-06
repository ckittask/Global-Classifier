from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import torch

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

def extract_user_queries(conversation: str) -> List[str]:

    user_patterns = [
        r'\*\*Kasutaja\*\*:(.*?)(?:\*\*Robot\*\*:|$)',
        r'Kasutaja:(.*?)(?:Robot:|$)',                  
        r'User:(.*?)(?:Assistant:|$)',                  
        r'Human:(.*?)(?:AI:|$)'                         
    ]
    
    queries = []
    for pattern in user_patterns:
        matches = re.findall(pattern, conversation, re.DOTALL | re.IGNORECASE)
        queries.extend([clean_text(m) for m in matches if m.strip()])
    
    if not queries:
        sentences = sent_tokenize(conversation) if conversation else []
        queries = [s.strip() for s in sentences if s.strip().endswith('?')]
    
    return queries

def split_into_segments(text: str, max_length: int = 200) -> List[str]:
    """Split text into meaningful segments for more granular comparison."""
    try:
        sentences = sent_tokenize(text)
        segments = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) <= max_length:
                current += " " + sentence if current else sentence
            else:
                if current:
                    segments.append(clean_text(current))
                current = sentence
                
        if current:
            segments.append(clean_text(current))
            
        return segments
    except:
        return [clean_text(s) for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]

def extract_key_terms(text: str, n: int = 10) -> List[str]:
    """Extract important terms from text using TF-IDF."""
    if not text.strip():
        return []
        
    try:
        
        with open("data/estonian-stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.9, 
            ngram_range=(1, 2),
            stop_words=stopwords
        )
        
        tfidf_matrix = vectorizer.fit_transform([text, "dummy text"])
        
        feature_names = vectorizer.get_feature_names_out()
        
        scores = zip(feature_names, tfidf_matrix[0].toarray()[0])
        
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [term for term, score in sorted_scores[:n]]
    except:
        words = re.findall(r'\b\w{3,}\b', text)
        return list(set(words))[:n]

def compute_weighted_segment_relevance(
    conversation: str, 
    topic_docs: List[str],
    segment_weight: float = 0.6,
    query_weight: float = 0.3,
    term_weight: float = 0.1
) -> Dict:

    topic_combined = " ".join(topic_docs)
    topic_combined = clean_text(topic_combined)
    conversation = clean_text(conversation)
    
    if not topic_combined or not conversation:
        return {
            "relevance_score": 0.0,
            "segment_score": 0.0,
            "query_score": 0.0, 
            "term_score": 0.0,
            "segments_analyzed": 0,
            "queries_analyzed": 0,
            "top_segments": [],
            "top_terms": []
        }
    
    topic_segments = split_into_segments(topic_combined)
    conv_segments = split_into_segments(conversation)
    
    segment_scores = []
    top_segments = []
    
    if topic_segments and conv_segments:
        topic_embeddings = model.encode(topic_segments, convert_to_tensor=True)
        conv_embeddings = model.encode(conv_segments, convert_to_tensor=True)
        
        similarity_matrix = util.pytorch_cos_sim(conv_embeddings, topic_embeddings)
        
        max_similarities = torch.max(similarity_matrix, dim=1).values
        segment_scores = max_similarities.tolist()
        
        segment_score = torch.mean(max_similarities).item()
        
        for i, score in enumerate(segment_scores):
            if score > 0.6:  
                top_segments.append({
                    "segment": conv_segments[i],
                    "score": score
                })
    else:
        segment_score = 0.0
    
    queries = extract_user_queries(conversation)
    query_score = 0.0
    
    if queries and topic_combined:
        query_embeddings = model.encode(queries, convert_to_tensor=True)
        topic_embedding = model.encode(topic_combined, convert_to_tensor=True)
        
        query_similarities = [
            util.pytorch_cos_sim(q_emb, topic_embedding).item() 
            for q_emb in query_embeddings
        ]
        
        if len(query_similarities) > 1:
            weights = [1.0] + [0.7] * (len(query_similarities) - 1)
            weighted_sims = [s * w for s, w in zip(query_similarities, weights)]
            query_score = sum(weighted_sims) / sum(weights)
        else:
            query_score = query_similarities[0] if query_similarities else 0.0
    
    topic_terms = extract_key_terms(topic_combined)
    conv_terms = extract_key_terms(conversation)
    
    if topic_terms and conv_terms:
        topic_terms_set = set(topic_terms)
        conv_terms_set = set(conv_terms)
        
        intersection = len(topic_terms_set.intersection(conv_terms_set))
        union = len(topic_terms_set.union(conv_terms_set))
        
        term_score = intersection / union if union > 0 else 0.0
    else:
        term_score = 0.0
    
    combined_score = (
        segment_weight * segment_score +
        query_weight * query_score +
        term_weight * term_score
    )
    
    final_score = max(0.0, min(combined_score, 1.0))
    
    return {
        "relevance_score": float(final_score),
        "segment_score": float(segment_score),
        "query_score": float(query_score),
        "term_score": float(term_score),
        "segments_analyzed": len(conv_segments),
        "queries_analyzed": len(queries),
        "top_segments": sorted(top_segments, key=lambda x: x["score"], reverse=True)[:3],
        "top_terms": list(set(topic_terms).intersection(set(conv_terms)))
    }

def compute_relevance_score(conversation: str, topic_docs: List[str]) -> float:
    """
    Simple interface that returns just the relevance score.
    Maintains backward compatibility with the original function.
    """
    results = compute_weighted_segment_relevance(conversation, topic_docs)
    return results["relevance_score"]

def analyze_conversation_set(
    conversations: List[str], 
    topic_docs: List[str],
    threshold_good: float = 0.7,
    threshold_acceptable: float = 0.5
) -> Dict:
    """
    Analyze relevance scores for a set of conversations.
    
    Args:
        conversations: List of conversation texts
        topic_docs: List of topic document texts
        threshold_good: Minimum score for "good" relevance
        threshold_acceptable: Minimum score for "acceptable" relevance
        
    Returns:
        Dictionary with analysis results
    """
    scores = []
    detailed_results = []
    
    for i, conv in enumerate(conversations):
        detailed = compute_weighted_segment_relevance(conv, topic_docs)
        scores.append(detailed["relevance_score"])
        
        detailed["conversation_index"] = i
        detailed_results.append(detailed)
    
    sorted_results = sorted(detailed_results, key=lambda x: x["relevance_score"], reverse=True)
    
    good_count = sum(1 for s in scores if s >= threshold_good)
    acceptable_count = sum(1 for s in scores if threshold_acceptable <= s < threshold_good)
    poor_count = sum(1 for s in scores if s < threshold_acceptable)
    
    return {
        "mean_score": np.mean(scores) if scores else 0.0,
        "median_score": np.median(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "good_count": good_count,
        "acceptable_count": acceptable_count,
        "poor_count": poor_count,
        "total_count": len(scores),
        "good_percentage": (good_count / len(scores) * 100) if scores else 0,
        "detailed_results": sorted_results
    }

if __name__ == "__main__":
    # Example usage for a single conversation
    from utils import read_file
    
    # Read conversation and topic
    conversation = read_file("data/ID.ee/autentimine_riiklikes_e-teenustes/conversation_1.txt")
    topic_docs = read_file("../data/output_ID.ee/Autentimine_riiklikes_e-teenustes_-_ID.ee.txt").split("\n")
    
    # Get detailed relevance metrics
    results = compute_weighted_segment_relevance(conversation, topic_docs)
    
    print(f"Overall Relevance Score: {results['relevance_score']:.4f}")
    print(f"Segment Score: {results['segment_score']:.4f}")
    print(f"Query Score: {results['query_score']:.4f}")
    print(f"Term Score: {results['term_score']:.4f}")
    
    print("\nTop Matching Segments:")
    for segment in results["top_segments"]:
        print(f"- {segment['segment'][:50]}... ({segment['score']:.4f})")
    
    print("\nShared Key Terms:")
    for term in results["top_terms"]:
        print(f"- {term}")
    
    # Example for multiple conversations
    import glob
    
    conversation_files = glob.glob("data/ID.ee/autentimine_riiklikes_e-teenustes/conversation_*.txt")
    conversations = [read_file(f) for f in conversation_files]
    
    analysis = analyze_conversation_set(conversations, topic_docs)
    
    print("\n=== Conversation Set Analysis ===")
    print(f"Mean Relevance: {analysis['mean_score']:.4f}")
    print(f"Good Conversations: {analysis['good_count']}/{analysis['total_count']} ({analysis['good_percentage']:.1f}%)")
    print(f"Range: {analysis['min_score']:.4f} - {analysis['max_score']:.4f}")
    
    print("\nTop 3 Most Relevant Conversations:")
    for i, result in enumerate(analysis["detailed_results"][:3]):
        print(f"{i+1}. Conversation {result['conversation_index']}: {result['relevance_score']:.4f}")