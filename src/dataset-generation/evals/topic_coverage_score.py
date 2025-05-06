from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import  DBSCAN
import re
from collections import Counter, defaultdict
from loguru import logger
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class TopicCoverageAnalyzer:
    """
    A class for analyzing whether all important topics in a source document 
    are covered by the generated conversations.
    """
    
    def __init__(self, 
                embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
                min_segment_length: int = 100,
                max_segment_length: int = 300,
                min_topic_size: int = 3,
                clustering_threshold: float = 0.25):
        """
        Initialize the topic coverage analyzer.
        
        Args:
            embedding_model: Name of sentence transformer model to use
            min_segment_length: Minimum character length for document segments
            max_segment_length: Maximum character length for document segments
            min_topic_size: Minimum number of segments to form a topic
            clustering_threshold: DBSCAN clustering threshold
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.min_topic_size = min_topic_size
        self.clustering_threshold = clustering_threshold
        self.stopwords = self._load_estonian_stopwords()
        
        # Will store results
        self.document_segments = []
        self.document_topics = []
        self.conversation_topics = []
        self.uncovered_topics = []
        self.coverage_scores = {}
        
    def _load_estonian_stopwords(self) -> List[str]:
        """Load Estonian stopwords."""
        try:
            with open("data/estonian-stopwords.txt", "r", encoding="utf-8") as f:
                stopwords = [line.strip() for line in f if line.strip()]
            return stopwords
        except:
            logger.warning("Could not load stopwords from file Using basic set.")
            return ['ja', 'ning', 'et', 'on', 'ei', 'ka', 'kui', 'aga', 'see', 'mis']
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess Estonian text."""
        text = text.lower()
        
        text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        estonian_chars = 'äöüõÄÖÜÕ'
        text = re.sub(f'[^a-zA-Z0-9{estonian_chars} ]', ' ', text)
        
        return text
    
    def _segment_document(self, document: str) -> List[str]:
        """
        Split document into meaningful segments for topic identification.
        Uses paragraph and sentence boundaries to create logical segments.
        """
        document = re.sub(r'\s+', ' ', document).strip()
        
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', document)
        
        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) <= self.max_segment_length:
                segments.append(para)
                continue
                
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_segment = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_segment) + len(sentence) > self.max_segment_length:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += " " + sentence if current_segment else sentence
            
            if current_segment:
                segments.append(current_segment)
        
        segments = [seg for seg in segments if len(seg) >= self.min_segment_length]
        
        segments = [self._preprocess_text(seg) for seg in segments]
        
        return segments
    
    def _extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract important keywords from text using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                stop_words=self.stopwords,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            
            tfidf_matrix = vectorizer.fit_transform([text, "dummy text"])
            feature_names = vectorizer.get_feature_names_out()
            
            scores = zip(feature_names, tfidf_matrix[0].toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            return [word for word, score in sorted_scores[:n]]
            
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            words = text.lower().split()
            word_counts = Counter(words)
            word_counts = {w: c for w, c in word_counts.items() if w not in self.stopwords and len(w) > 2}
            return [w for w, c in word_counts.most_common(n)]
    
    def _identify_document_topics(self, segments: List[str]) -> List[Dict]:
        """
        Identify distinct topics within the document by clustering segments.
        
        Returns:
            List of topics, each containing details about the segments and keywords
        """
        if not segments:
            return []
            
        try:
            embeddings = self.embedding_model.encode(segments)
            
            clustering = DBSCAN(
                eps=self.clustering_threshold,
                min_samples=self.min_topic_size,
                metric='cosine'
            ).fit(embeddings)
            
            labels = clustering.labels_
            
            topics = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  
                    topics[label].append({
                        "segment": segments[i],
                        "segment_id": i,
                        "embedding": embeddings[i]
                    })
            
            topic_details = []
            for topic_id, topic_segments in topics.items():
                combined_text = " ".join([s["segment"] for s in topic_segments])
                
                keywords = self._extract_keywords(combined_text)
                
                centroid = np.mean([s["embedding"] for s in topic_segments], axis=0)
                
                topic_details.append({
                    "topic_id": topic_id,
                    "segments": topic_segments,
                    "segment_count": len(topic_segments),
                    "keywords": keywords,
                    "centroid": centroid,
                })
            
            return topic_details
            
        except Exception as e:
            logger.error(f"Error identifying document topics: {e}")
            return []
    
    def _evaluate_conversation_topic_coverage(self, 
                                             conversations: List[str], 
                                             document_topics: List[Dict]) -> Dict:
        """
        Evaluate how well conversations cover the document topics.
        
        Returns:
            Dictionary with coverage analysis results
        """
        if not conversations or not document_topics:
            return {
                "total_topics": 0,
                "covered_topics": 0,
                "coverage_percentage": 0.0,
                "uncovered_topics": [],
                "topic_coverage_scores": {},
                "conversation_coverage": []
            }
        
        preprocessed_conversations = [self._preprocess_text(conv) for conv in conversations]
        
        conversation_embeddings = self.embedding_model.encode(preprocessed_conversations)
        
        topic_coverage = {}
        
        for topic in document_topics:
            topic_id = topic["topic_id"]
            topic_centroid = topic["centroid"]
            
            similarities = cosine_similarity([topic_centroid], conversation_embeddings)[0]
            
            best_match_idx = np.argmax(similarities)
            best_match_score = similarities[best_match_idx]
            
            is_covered = best_match_score >= 0.5
            
            topic_coverage[topic_id] = {
                "topic_id": topic_id,
                "keywords": topic["keywords"],
                "segment_count": topic["segment_count"],
                "is_covered": is_covered,
                "best_match_score": float(best_match_score),
                "best_match_conversation_idx": int(best_match_idx)
            }
        
        uncovered_topics = [
            {
                "topic_id": t["topic_id"],
                "keywords": t["keywords"],
                "segment_count": t["segment_count"]
            }
            for t in document_topics
            if not topic_coverage[t["topic_id"]]["is_covered"]
        ]
        
        total_topics = len(document_topics)
        covered_topics = sum(1 for t in topic_coverage.values() if t["is_covered"])
        coverage_percentage = (covered_topics / total_topics) * 100 if total_topics > 0 else 0
        
        conversation_coverage = []
        for i, (conv, embedding) in enumerate(zip(conversations, conversation_embeddings)):
            covered_topics = []
            for topic in document_topics:
                topic_id = topic["topic_id"]
                similarity = cosine_similarity([topic["centroid"]], [embedding])[0][0]
                if similarity >= 0.5:
                    covered_topics.append({
                        "topic_id": topic_id,
                        "similarity": float(similarity),
                        "keywords": topic["keywords"]
                    })
            
            keywords = self._extract_keywords(conv)
            
            conversation_coverage.append({
                "conversation_idx": i,
                "covered_topics": covered_topics,
                "topic_count": len(covered_topics),
                "keywords": keywords
            })
        
        return {
            "total_topics": total_topics,
            "covered_topics": covered_topics,
            "coverage_percentage": coverage_percentage,
            "uncovered_topics": uncovered_topics,
            "topic_coverage_scores": topic_coverage,
            "conversation_coverage": conversation_coverage
        }
    
    def analyze_topic_coverage(self, 
                              document: str, 
                              conversations: List[str]) -> Dict:
        """
        Analyze how well the conversations cover topics in the document.
        

        """
        self.document_segments = self._segment_document(document)
        
        if not self.document_segments:
            logger.error("Could not extract meaningful segments from document")
            return {"error": "Could not extract meaningful segments from document"}
        
        self.document_topics = self._identify_document_topics(self.document_segments)
        
        if not self.document_topics:
            logger.error("Could not identify distinct topics in document")
            return {"error": "Could not identify distinct topics in document"}
        
        coverage_results = self._evaluate_conversation_topic_coverage(
            conversations, 
            self.document_topics
        )
        
        self.uncovered_topics = coverage_results["uncovered_topics"]
        self.coverage_scores = coverage_results["topic_coverage_scores"]
        
        return coverage_results
    
    def generate_coverage_report(self, 
                               document: str, 
                               conversations: List[str],
                               conversation_files: List[str] = None,
                               output_dir: str = None) -> Dict:
        """
        Generate comprehensive topic coverage report.
        
        Args:
            document: Source document text
            conversations: List of conversation texts
            conversation_files: Optional list of conversation filenames
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with coverage analysis and report file paths
        """
        coverage_results = self.analyze_topic_coverage(document, conversations)
        
        if "error" in coverage_results:
            return coverage_results
        
        if not conversation_files:
            conversation_files = [f"conversation_{i+1}.txt" for i in range(len(conversations))]
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, coverage in enumerate(coverage_results["conversation_coverage"]):
            if i < len(conversation_files):
                coverage["filename"] = conversation_files[i]
        
        output_files = {}
        if output_dir:
            md_report_path = os.path.join(output_dir, "topic_coverage_report.md")
            self._generate_markdown_report(coverage_results, md_report_path)
            output_files["markdown_report"] = md_report_path
            
            heatmap_path = os.path.join(output_dir, "topic_coverage_heatmap.png")
            self._generate_coverage_heatmap(coverage_results, heatmap_path)
            output_files["heatmap"] = heatmap_path
            
            csv_path = os.path.join(output_dir, "topic_coverage_data.csv")
            self._generate_coverage_csv(coverage_results, csv_path)
            output_files["csv_data"] = csv_path
        
        coverage_results["output_files"] = output_files
        
        return coverage_results
    
    def _generate_markdown_report(self, 
                                coverage_results: Dict, 
                                output_file: str) -> None:
        """Generate detailed markdown report of topic coverage."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Topic Coverage Analysis Report\n\n")
                
                f.write("## Summary\n\n")
                f.write(f"- **Total topics identified:** {coverage_results['total_topics']}\n")
                f.write(f"- **Topics covered by conversations:** {coverage_results['covered_topics']}\n")
                f.write(f"- **Coverage percentage:** {coverage_results['coverage_percentage']:.1f}%\n")
                f.write(f"- **Uncovered topics:** {len(coverage_results['uncovered_topics'])}\n\n")
                
                f.write("## Uncovered Topics\n\n")
                
                if coverage_results['uncovered_topics']:
                    f.write("The following topics from the source document are not adequately covered by the conversations:\n\n")
                    f.write("| Topic ID | Key Terms | Segment Count |\n")
                    f.write("|---------|-----------|---------------|\n")
                    
                    for topic in coverage_results['uncovered_topics']:
                        topic_id = topic['topic_id']
                        keywords = ", ".join(topic['keywords'][:5])  
                        segment_count = topic['segment_count']
                        f.write(f"| {topic_id} | {keywords} | {segment_count} |\n")
                else:
                    f.write("All topics are adequately covered by the conversations. Great job! 👍\n")
                
                f.write("\n")
                
                f.write("## Conversation Coverage Analysis\n\n")
                
                for conv_coverage in coverage_results['conversation_coverage']:
                    filename = conv_coverage.get('filename', f"Conversation {conv_coverage['conversation_idx']}")
                    topic_count = conv_coverage['topic_count']
                    f.write(f"### {filename}\n\n")
                    f.write(f"- **Topics covered:** {topic_count}\n")
                    
                    if conv_coverage['covered_topics']:
                        f.write("- **Covered topics:**\n")
                        for topic in conv_coverage['covered_topics']:
                            keywords = ", ".join(topic['keywords'][:3])
                            f.write(f"  - Topic {topic['topic_id']}: {keywords} (similarity: {topic['similarity']:.2f})\n")
                    else:
                        f.write("- This conversation doesn't strongly cover any specific topic.\n")
                    
                    f.write("\n")
                
                f.write("## Analysis Details\n\n")
                f.write(f"- **Analysis performed:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Embedding model:** {self.embedding_model_name}\n")
                f.write(f"- **Coverage threshold:** 0.5 (cosine similarity)\n")
                f.write(f"- **Document segments analyzed:** {len(self.document_segments)}\n")
                
                f.write("\n")
                f.write("---\n")
                f.write("*This report was automatically generated by the Topic Coverage Analysis tool.*")
                
            logger.info(f"Markdown report written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
    
    def _generate_coverage_heatmap(self, 
                                 coverage_results: Dict, 
                                 output_file: str) -> None:
        """Generate heatmap visualization of topic-conversation coverage."""
        try:
            topic_ids = sorted([t["topic_id"] for t in self.document_topics])
            conversation_ids = [i for i in range(len(coverage_results["conversation_coverage"]))]
            
            coverage_matrix = np.zeros((len(conversation_ids), len(topic_ids)))
            
            for conv_idx, conv_coverage in enumerate(coverage_results["conversation_coverage"]):
                for topic in conv_coverage["covered_topics"]:
                    topic_idx = topic_ids.index(topic["topic_id"])
                    coverage_matrix[conv_idx, topic_idx] = topic["similarity"]
            
            plt.figure(figsize=(max(8, len(topic_ids)*0.5), max(6, len(conversation_ids)*0.4)))
            
            conv_labels = []
            for i, conv in enumerate(coverage_results["conversation_coverage"]):
                if "filename" in conv:
                    # Shorten filename if needed
                    name = os.path.basename(conv["filename"])
                    if len(name) > 20:
                        name = name[:17] + "..."
                    conv_labels.append(name)
                else:
                    conv_labels.append(f"Conv {i+1}")
            
            topic_labels = []
            for topic_id in topic_ids:
                topic = next((t for t in self.document_topics if t["topic_id"] == topic_id), None)
                if topic:
                    keywords = " ".join(topic["keywords"][:2])  # Just top 2 keywords
                    topic_labels.append(f"Topic {topic_id}\n({keywords})")
                else:
                    topic_labels.append(f"Topic {topic_id}")
            
            ax = sns.heatmap(
                coverage_matrix,
                annot=True,
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                xticklabels=topic_labels,
                yticklabels=conv_labels,
                linewidths=0.5,
                fmt=".2f"
            )
            
            plt.title("Topic Coverage by Conversation")
            plt.ylabel("Conversations")
            plt.xlabel("Topics (with key terms)")
            plt.tight_layout()
            
            plt.xticks(rotation=45, ha="right")
            
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Coverage heatmap written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating coverage heatmap: {e}")
    
    def _generate_coverage_csv(self, 
                             coverage_results: Dict, 
                             output_file: str) -> None:
        """Generate CSV data file with coverage information."""
        try:
            topic_rows = []
            for topic in self.document_topics:
                topic_id = topic["topic_id"]
                coverage = coverage_results["topic_coverage_scores"].get(topic_id, {})
                
                topic_rows.append({
                    "topic_id": topic_id,
                    "keywords": ", ".join(topic["keywords"]),
                    "segment_count": topic["segment_count"],
                    "is_covered": coverage.get("is_covered", False),
                    "best_match_score": coverage.get("best_match_score", 0.0),
                    "best_match_conversation": coverage.get("best_match_conversation_idx", -1)
                })
            
            topics_df = pd.DataFrame(topic_rows)
            
            conv_rows = []
            for conv in coverage_results["conversation_coverage"]:
                filename = conv.get("filename", f"conversation_{conv['conversation_idx']}.txt")
                
                conv_rows.append({
                    "conversation_id": conv["conversation_idx"],
                    "filename": filename,
                    "topics_covered": conv["topic_count"],
                    "keywords": ", ".join(conv["keywords"]),
                    "covered_topic_ids": ", ".join([str(t["topic_id"]) for t in conv["covered_topics"]])
                })
            
            conversations_df = pd.DataFrame(conv_rows)
            
            with pd.ExcelWriter(output_file) as writer:
                topics_df.to_excel(writer, sheet_name="Topics", index=False)
                conversations_df.to_excel(writer, sheet_name="Conversations", index=False)
                
                summary_data = {
                    "Metric": [
                        "Total Topics", 
                        "Topics Covered", 
                        "Coverage Percentage", 
                        "Uncovered Topics",
                        "Total Conversations"
                    ],
                    "Value": [
                        coverage_results["total_topics"],
                        coverage_results["covered_topics"],
                        f"{coverage_results['coverage_percentage']:.1f}%",
                        len(coverage_results["uncovered_topics"]),
                        len(coverage_results["conversation_coverage"])
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            logger.info(f"Coverage data written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating coverage CSV: {e}")


def read_file(file_path: str) -> str:
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def read_topic_documents(topic_docs_path: str) -> str:
    """Read topic document content."""
    if os.path.isfile(topic_docs_path):
        return read_file(topic_docs_path)
    elif os.path.isdir(topic_docs_path):
        files = glob.glob(os.path.join(topic_docs_path, "*.txt"))
        content = []
        for f in files:
            content.append(read_file(f))
        return "\n\n".join(content)
    else:
        logger.error(f"Invalid topic document path: {topic_docs_path}")
        return ""

def read_conversations_from_directory(directory: str, pattern: str = "conversation_*.txt") -> Tuple[List[str], List[str]]:
    """
    Read all conversation files from a directory matching a pattern.
    
    Returns:
        Tuple of (conversation_contents, filenames)
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    contents = []
    filenames = []
    
    for f in files:
        content = read_file(f)
        if content.strip():  # Skip empty files
            contents.append(content)
            filenames.append(os.path.basename(f))
    
    return contents, filenames

def analyze_topic_coverage_from_files(
    conversation_path: str,
    topic_docs_path: str,
    output_dir: str = None
) -> Dict:
    """
    Analyze topic coverage using conversation files and topic document.

    """
    document = read_topic_documents(topic_docs_path)
    
    if not document:
        logger.error(f"Could not read topic document from {topic_docs_path}")
        return {"error": "Could not read topic document"}
    
    if os.path.isdir(conversation_path):
        conversations, filenames = read_conversations_from_directory(conversation_path)
    else:
        content = read_file(conversation_path)
        if content:
            conversations = [content]
            filenames = [os.path.basename(conversation_path)]
        else:
            conversations = []
            filenames = []
    
    if not conversations:
        logger.error(f"No conversations found at {conversation_path}")
        return {"error": "No conversations found"}
    
    analyzer = TopicCoverageAnalyzer()
    
    return analyzer.generate_coverage_report(
        document,
        conversations,
        filenames,
        output_dir
    )

if __name__ == "__main__":
    


    conversation_path = "data/ID.ee/autentimine_riiklikes_e-teenustes"
    topic_docs_path = "../data/output_ID.ee/Autentimine_riiklikes_e-teenustes_-_ID.ee.txt"
    output_dir = "data/ID.ee/autentimine_riiklikes_e-teenustes/topic_coverage_analysis"

    
    # Analyze topic coverage
    results = analyze_topic_coverage_from_files(
        conversation_path,
        topic_docs_path,
        output_dir
    )
    
    # Print summary results
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("Topic Coverage Analysis:")
        print(f"- Total topics identified: {results['total_topics']}")
        print(f"- Topics covered by conversations: {results['covered_topics']}")
        print(f"- Coverage percentage: {results['coverage_percentage']:.1f}%")
        print(f"- Uncovered topics: {len(results['uncovered_topics'])}")
        
        if results['uncovered_topics']:
            print("\nUncovered topics:")
            for topic in results['uncovered_topics'][:5]:  # Show up to 5
                keywords = ", ".join(topic['keywords'][:3])
                print(f"- Topic {topic['topic_id']}: {keywords}")
            
            if len(results['uncovered_topics']) > 5:
                print(f"  ...and {len(results['uncovered_topics']) - 5} more")
        
        if output_dir:
            print("\nOutput files:")
            for file_type, file_path in results['output_files'].items():
                print(f"- {file_type}: {file_path}")