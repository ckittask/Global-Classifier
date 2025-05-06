from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter, defaultdict
from loguru import logger
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# remove the default stderr handler
logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class AgencyConfusionAnalyzer:
    """
    A class for analyzing potential confusion between conversations from different agencies.
    Focused on identifying conversations that might be misclassified due to similarity.
    """
    
    def __init__(self, 
                embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
                confusion_threshold: float = 0.85,
                top_confusion_samples: int = 10):
        
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.confusion_threshold = confusion_threshold
        self.top_confusion_samples = top_confusion_samples
        self.stopwords = self._load_estonian_stopwords()
        
    def _load_estonian_stopwords(self) -> List[str]:
        try:
            with open("data/estonian-stopwords.txt", "r", encoding="utf-8") as f:
                est_stopwords = [line.strip() for line in f if line.strip()]
            return est_stopwords
        except:
            logger.warning("Estonian stopwords file not found. Using default stopwords.")
            return ['ja', 'ning', 'et', 'on', 'ei', 'ka', 'kui', 'aga', 'see', 'mis']
    
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)        
        text = re.sub(r'\s+', ' ', text).strip()
        
        special_characters = 'äöüõÄÖÜÕ'
        text = re.sub(f'[^a-zA-Z0-9{special_characters} ]', ' ', text)
        
        return text

    def extract_user_queries(self, conversation: str, first_only: bool = False) -> List[str]:
        """
        Extract user queries/questions from a conversation.

        """
        user_patterns = [
            r'\*\*Kasutaja\*\*:(.*?)(?:\*\*Robot\*\*:|$)', 
            r'Kasutaja:(.*?)(?:Robot:|$)',                   
            r'User:(.*?)(?:Assistant:|$)',                   
            r'Human:(.*?)(?:AI:|$)'                        
        ]
        
        queries = []
        for pattern in user_patterns:
            matches = re.findall(pattern, conversation, re.DOTALL | re.IGNORECASE)
            if matches:
                cleaned_matches = [self._preprocess_text(m) for m in matches if m.strip()]
                if first_only and cleaned_matches:
                    return [cleaned_matches[0]]
                queries.extend(cleaned_matches)
        
        if not queries:
            sentences = re.split(r'(?<=[.!?])\s+', conversation)
            question_sentences = [s.strip() for s in sentences if s.strip().endswith('?')]
            if question_sentences:
                cleaned_sentences = [self._preprocess_text(s) for s in question_sentences]
                if first_only and cleaned_sentences:
                    return [cleaned_sentences[0]]
                queries.extend(cleaned_sentences)
        
        if not queries and len(conversation.strip()) < 200:
            queries = [self._preprocess_text(conversation)]
        
        return queries

    def analyze_cross_agency_confusion(self, 
                                      agency_topics_conversations: Dict[str, Dict[str, List[str]]]) -> Dict:
        """
        Analyze potential confusion between different agencies' conversations.

        """
        if not isinstance(agency_topics_conversations, dict) or len(agency_topics_conversations) < 2:
            logger.error("Need at least two agencies to analyze confusion")
            return {"error": "Need at least two agencies to analyze confusion"}
        
        agency_topic_queries = {}
        agency_topic_query_indices = {}  
        agency_query_counts = {}
        
        for agency, topic_conversations in agency_topics_conversations.items():
            agency_topic_queries[agency] = {}
            agency_topic_query_indices[agency] = {}
            total_agency_queries = 0
            
            for topic, conversations in topic_conversations.items():
                queries = []
                indices = []
                
                for i, conv in enumerate(conversations):
                    first_query = self.extract_user_queries(conv, first_only=True)
                    if first_query:
                        queries.append(first_query[0])
                        indices.append(i)
                
                agency_topic_queries[agency][topic] = queries
                agency_topic_query_indices[agency][topic] = indices
                total_agency_queries += len(queries)
            
            agency_query_counts[agency] = total_agency_queries
        
        all_agencies_have_queries = all(count > 0 for count in agency_query_counts.values())
        if not all_agencies_have_queries:
            empty_agencies = [agency for agency, count in agency_query_counts.items() if count == 0]
            logger.error(f"No queries found for agencies: {', '.join(empty_agencies)}")
            return {"error": f"No queries found for agencies: {', '.join(empty_agencies)}"}
        
        agency_queries = {}
        agency_query_topics = {}
        agency_query_indices = {}
        
        for agency, topic_queries in agency_topic_queries.items():
            queries = []
            topics = []
            indices = []
            
            for topic, topic_query_list in topic_queries.items():
                for i, query in enumerate(topic_query_list):
                    queries.append(query)
                    topics.append(topic)
                    
                    conv_idx = agency_topic_query_indices[agency][topic][i]
                    indices.append((topic, conv_idx))
            
            agency_queries[agency] = queries
            agency_query_topics[agency] = topics
            agency_query_indices[agency] = indices
        
        agency_embeddings = {}
        for agency, queries in agency_queries.items():
            agency_embeddings[agency] = self.embedding_model.encode(queries)
        
        confusion_results = {}
        
        agency_pairs = []
        for i, agency1 in enumerate(agency_queries.keys()):
            for j, agency2 in enumerate(agency_queries.keys()):
                if i < j:  
                    agency_pairs.append((agency1, agency2))
        
        for agency1, agency2 in agency_pairs:
            queries1 = agency_queries[agency1]
            queries2 = agency_queries[agency2]
            topics1 = agency_query_topics[agency1]
            topics2 = agency_query_topics[agency2]
            embeddings1 = agency_embeddings[agency1]
            embeddings2 = agency_embeddings[agency2]
            indices1 = agency_query_indices[agency1]
            indices2 = agency_query_indices[agency2]
            
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            
            confusion_pairs = []
            topic_pair_confusion = defaultdict(list)
            
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    sim = similarity_matrix[i, j]
                    if sim >= self.confusion_threshold:
                        topic1 = topics1[i]
                        topic2 = topics2[j]
                        topic_pair = (topic1, topic2)
                        
                        confusion_pair = {
                            "agency1_query": queries1[i],
                            "agency2_query": queries2[j],
                            "agency1_topic": topic1,
                            "agency2_topic": topic2,
                            "similarity": float(sim),
                            "agency1_conv_idx": indices1[i],
                            "agency2_conv_idx": indices2[j]
                        }
                        
                        confusion_pairs.append(confusion_pair)
                        topic_pair_confusion[topic_pair].append(confusion_pair)
            
            confusion_pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
            confusion_rate = len(confusion_pairs) / (len(queries1) * len(queries2))
            avg_similarity = np.mean(similarity_matrix)
            max_similarity = np.max(similarity_matrix)
            
            topic_pair_metrics = {}
            for (topic1, topic2), pairs in topic_pair_confusion.items():
                topic1_count = topics1.count(topic1)
                topic2_count = topics2.count(topic2)
                possible_pairs = topic1_count * topic2_count
                
                topic_pair_rate = len(pairs) / possible_pairs
                
                sorted_pairs = sorted(pairs, key=lambda x: x["similarity"], reverse=True)
                
                topic_pair_metrics[(topic1, topic2)] = {
                    "confusion_pairs": sorted_pairs[:self.top_confusion_samples],
                    "confusion_count": len(pairs),
                    "total_possible_pairs": possible_pairs,
                    "confusion_rate": topic_pair_rate,
                    "topic1_query_count": topic1_count,
                    "topic2_query_count": topic2_count
                }
            
            if topic_pair_metrics:
                most_confusable_topic_pair = max(
                    topic_pair_metrics.items(), 
                    key=lambda x: x[1]["confusion_rate"]
                )
            else:
                most_confusable_topic_pair = None
            
            confusion_results[(agency1, agency2)] = {
                "confusion_pairs": confusion_pairs[:self.top_confusion_samples],
                "confusion_count": len(confusion_pairs),
                "total_possible_pairs": len(queries1) * len(queries2),
                "confusion_rate": confusion_rate,
                "average_similarity": float(avg_similarity),
                "max_similarity": float(max_similarity),
                "agency1_query_count": len(queries1),
                "agency2_query_count": len(queries2),
                "topic_pair_metrics": topic_pair_metrics,
                "most_confusable_topic_pair": most_confusable_topic_pair
            }
        
        all_confusion_rates = [r["confusion_rate"] for r in confusion_results.values()]
        all_avg_similarities = [r["average_similarity"] for r in confusion_results.values()]
        all_confusion_counts = [r["confusion_count"] for r in confusion_results.values()]
        
        overall_results = {
            "overall_confusion_rate": np.mean(all_confusion_rates),
            "overall_similarity": np.mean(all_avg_similarities),
            "total_confusion_pairs": sum(all_confusion_counts),
            "agency_pair_results": confusion_results,
            "agency_topic_counts": {
                agency: {topic: len(queries) for topic, queries in topics.items()}
                for agency, topics in agency_topic_queries.items()
            },
            "most_confusable_pair": max(confusion_results.items(), 
                                      key=lambda x: x[1]["confusion_rate"]) if confusion_results else None
        }
        
        return overall_results

    def analyze_topic_confusion_within_agency(self, 
                                             topic_conversations: Dict[str, List[str]],
                                             agency_name: str = None) -> Dict:
        """
        Analyze potential confusion between different topics within the same agency.
        
        Args:
            topic_conversations: Dictionary mapping topic names to lists of conversations
            agency_name: Optional agency name for reporting
            
        Returns:
            Dictionary with confusion analysis results
        """
        if not isinstance(topic_conversations, dict) or len(topic_conversations) < 2:
            logger.error("Need at least two topics to analyze confusion")
            return {"error": "Need at least two topics to analyze confusion"}
        
        topic_queries = {}
        topic_query_indices = {}
        
        for topic, conversations in topic_conversations.items():
            queries = []
            indices = []
            
            for i, conv in enumerate(conversations):
                first_query = self.extract_user_queries(conv, first_only=True)
                if first_query:
                    queries.append(first_query[0])
                    indices.append(i)
            
            topic_queries[topic] = queries
            topic_query_indices[topic] = indices
        
        all_topics_have_queries = all(len(queries) > 0 for queries in topic_queries.values())
        if not all_topics_have_queries:
            empty_topics = [topic for topic, queries in topic_queries.items() if not queries]
            logger.error(f"No queries found for topics: {', '.join(empty_topics)}")
            return {"error": f"No queries found for topics: {', '.join(empty_topics)}"}
        
        topic_embeddings = {}
        for topic, queries in topic_queries.items():
            topic_embeddings[topic] = self.embedding_model.encode(queries)
        
        confusion_results = {}
        
        topic_pairs = []
        for i, topic1 in enumerate(topic_queries.keys()):
            for j, topic2 in enumerate(topic_queries.keys()):
                if i < j:  
                    topic_pairs.append((topic1, topic2))
        
        for topic1, topic2 in topic_pairs:
            queries1 = topic_queries[topic1]
            queries2 = topic_queries[topic2]
            embeddings1 = topic_embeddings[topic1]
            embeddings2 = topic_embeddings[topic2]
            indices1 = topic_query_indices[topic1]
            indices2 = topic_query_indices[topic2]
            
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            
            confusion_pairs = []
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    sim = similarity_matrix[i, j]
                    if sim >= self.confusion_threshold:
                        confusion_pairs.append({
                            "topic1_query": queries1[i],
                            "topic2_query": queries2[j],
                            "similarity": float(sim),
                            "topic1_conv_idx": indices1[i],
                            "topic2_conv_idx": indices2[j]
                        })
            
            confusion_pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
            confusion_rate = len(confusion_pairs) / (len(queries1) * len(queries2))
            avg_similarity = np.mean(similarity_matrix)
            max_similarity = np.max(similarity_matrix)
            
            confusion_results[(topic1, topic2)] = {
                "confusion_pairs": confusion_pairs[:self.top_confusion_samples],
                "confusion_count": len(confusion_pairs),
                "total_possible_pairs": len(queries1) * len(queries2),
                "confusion_rate": confusion_rate,
                "average_similarity": float(avg_similarity),
                "max_similarity": float(max_similarity),
                "topic1_query_count": len(queries1),
                "topic2_query_count": len(queries2)
            }
        
        all_confusion_rates = [r["confusion_rate"] for r in confusion_results.values()]
        all_avg_similarities = [r["average_similarity"] for r in confusion_results.values()]
        all_confusion_counts = [r["confusion_count"] for r in confusion_results.values()]
        
        overall_results = {
            "agency_name": agency_name,
            "overall_confusion_rate": np.mean(all_confusion_rates) if all_confusion_rates else 0.0,
            "overall_similarity": np.mean(all_avg_similarities) if all_avg_similarities else 0.0,
            "total_confusion_pairs": sum(all_confusion_counts),
            "topic_pair_results": confusion_results,
            "most_confusable_pair": max(confusion_results.items(), 
                                      key=lambda x: x[1]["confusion_rate"]) if confusion_results else None
        }
        
        return overall_results

    def generate_confusion_matrix_visualization(self, 
                                              confusion_results: Dict,
                                              output_file: str = None,
                                              is_agency_level: bool = True) -> None:
        """
        Generate a heatmap visualization of the confusion between agencies or topics.
        
        Args:
            confusion_results: Results from analyze_cross_agency_confusion or analyze_topic_confusion
            output_file: Path to save the visualization
            is_agency_level: Whether this is agency-level (True) or topic-level (False) confusion
        """
        if is_agency_level:
            pair_results = confusion_results.get("agency_pair_results", {})
            entity_name = "Agency"
        else:
            pair_results = confusion_results.get("topic_pair_results", {})
            entity_name = "Topic"
        
        if not pair_results:
            logger.error("No confusion results to visualize")
            return
        
        entities = set()
        for pair in pair_results.keys():
            entities.add(pair[0])
            entities.add(pair[1])
        
        entities = sorted(list(entities))
        n_entities = len(entities)
        
        confusion_matrix = np.zeros((n_entities, n_entities))
        similarity_matrix = np.zeros((n_entities, n_entities))
        
        for (entity1, entity2), result in pair_results.items():
            i = entities.index(entity1)
            j = entities.index(entity2)
            
            confusion_rate = result["confusion_rate"]
            avg_similarity = result["average_similarity"]
            
            confusion_matrix[i, j] = confusion_rate
            confusion_matrix[j, i] = confusion_rate
            
            similarity_matrix[i, j] = avg_similarity
            similarity_matrix[j, i] = avg_similarity
        
        np.fill_diagonal(confusion_matrix, 0)
        np.fill_diagonal(similarity_matrix, 0)
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(confusion_matrix, annot=True, cmap="YlOrRd", 
                   xticklabels=entities, yticklabels=entities, 
                   vmin=0, vmax=min(1.0, confusion_matrix.max() * 1.2), fmt=".3f")
        plt.title(f"{entity_name} Confusion Rate")
        plt.tight_layout()
        
        plt.subplot(1, 2, 2)
        sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", 
                   xticklabels=entities, yticklabels=entities, 
                   vmin=0, vmax=1.0, fmt=".3f")
        plt.title(f"{entity_name} Query Similarity")
        plt.tight_layout()
        
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Confusion matrix visualization saved to {output_file}")
        else:
            plt.show()

    def generate_topic_pair_heatmap(self,
                                  confusion_results: Dict,
                                  output_file: str = None) -> None:
        """
        Generate a heatmap showing confusion between specific topic pairs across agencies.

        """
        agency_pair_results = confusion_results.get("agency_pair_results", {})
        
        if not agency_pair_results:
            logger.error("No agency pair results to visualize")
            return
        all_topic_pairs = set()
        topic_confusion_data = []
        
        for (agency1, agency2), pair_result in agency_pair_results.items():
            for (topic1, topic2), topic_metrics in pair_result.get("topic_pair_metrics", {}).items():
                topic_pair_id = f"{topic1} ({agency1}) - {topic2} ({agency2})"
                all_topic_pairs.add(topic_pair_id)
                
                topic_confusion_data.append({
                    "topic_pair": topic_pair_id,
                    "agency_pair": f"{agency1}-{agency2}",
                    "topic1": topic1,
                    "topic2": topic2,
                    "agency1": agency1,
                    "agency2": agency2,
                    "confusion_rate": topic_metrics["confusion_rate"],
                    "confusion_count": topic_metrics["confusion_count"],
                    "total_possible": topic_metrics["total_possible_pairs"]
                })
        
        if len(topic_confusion_data) > 20:
            topic_confusion_data.sort(key=lambda x: x["confusion_rate"], reverse=True)
            topic_confusion_data = topic_confusion_data[:20]
        
        topic_confusion_df = pd.DataFrame(topic_confusion_data)
        
        if len(topic_confusion_df) == 0:
            logger.warning("No topic confusion data to visualize")
            return
        
        plt.figure(figsize=(12, 10))
        
        topic_confusion_df = topic_confusion_df.sort_values("confusion_rate")
        ax = sns.barplot(
            x="confusion_rate",
            y="topic_pair",
            data=topic_confusion_df,
            palette="YlOrRd"
        )
        
        for i, row in enumerate(topic_confusion_df.itertuples()):
            ax.text(
                row.confusion_rate + 0.01, i,
                f"{row.confusion_count}/{row.total_possible}",
                va='center'
            )
        
        plt.title("Most Confusable Topic Pairs Across Agencies")
        plt.xlabel("Confusion Rate")
        plt.ylabel("Topic Pair")
        plt.tight_layout()
        
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Topic pair confusion visualization saved to {output_file}")
        else:
            plt.show()



    def generate_confusion_report(self, 
                                confusion_results: Dict,
                                output_file: str = None,
                                is_agency_level: bool = True) -> None:
        """
        Generate a detailed report from confusion analysis results.
 
        """
        if is_agency_level:
            pair_results = confusion_results.get("agency_pair_results", {})
            entity_name = "Agency"
            entity_label = "agency"
        else:
            pair_results = confusion_results.get("topic_pair_results", {})
            entity_name = "Topic"
            entity_label = "topic"
            agency_name = confusion_results.get("agency_name", "Unknown Agency")
        
        if not pair_results:
            logger.error("No confusion results to report")
            return
        
        report_lines = []
        
        if is_agency_level:
            report_lines.append("# Cross-Agency Confusion Analysis\n")
        else:
            report_lines.append(f"# Topic Confusion Analysis for {agency_name}\n")
        
        report_lines.append("## Overview\n")
        report_lines.append(f"- **Overall Confusion Rate**: {confusion_results['overall_confusion_rate']:.4f}")
        report_lines.append(f"- **Average Query Similarity**: {confusion_results['overall_similarity']:.4f}")
        report_lines.append(f"- **Total Confusable Query Pairs**: {confusion_results['total_confusion_pairs']}")
        
        if confusion_results['most_confusable_pair']:
            most_confusable = confusion_results['most_confusable_pair']
            pair_names = most_confusable[0]
            pair_results_data = most_confusable[1]
            
            report_lines.append(f"- **Most Confusable {entity_name} Pair**: {pair_names[0]} and {pair_names[1]}")
            report_lines.append(f"  - Confusion Rate: {pair_results_data['confusion_rate']:.4f}")
            report_lines.append(f"  - Confusable Pairs: {pair_results_data['confusion_count']} out of {pair_results_data['total_possible_pairs']} possible\n")
        

        has_topic_metrics = False
        if is_agency_level:
            for result_val in pair_results.values():
                if isinstance(result_val, dict) and "topic_pair_metrics" in result_val:
                    if result_val["topic_pair_metrics"]:  
                        has_topic_metrics = True
                        break
            
        if is_agency_level and has_topic_metrics:
            report_lines.append("## Most Confusable Topic Pairs\n")
            
            topic_pair_data = []
            
            for (agency1, agency2), pair_result in pair_results.items():
                if not isinstance(pair_result, dict):
                    continue
                    
                topic_pair_metrics = pair_result.get("topic_pair_metrics", {})
                if not isinstance(topic_pair_metrics, dict):
                    continue
                    
                for (topic1, topic2), topic_metrics in topic_pair_metrics.items():
                    if not isinstance(topic_metrics, dict):
                        continue
                        
                    confusion_rate = topic_metrics.get("confusion_rate", 0)
                    if confusion_rate > 0:
                        topic_pair_data.append({
                            "agency1": agency1,
                            "agency2": agency2,
                            "topic1": topic1,
                            "topic2": topic2,
                            "confusion_rate": confusion_rate,
                            "confusion_count": topic_metrics.get("confusion_count", 0),
                            "total_possible": topic_metrics.get("total_possible_pairs", 0)
                        })
            
            topic_pair_data.sort(key=lambda x: x["confusion_rate"], reverse=True)
            
            for i, data in enumerate(topic_pair_data[:10]):
                report_lines.append(f"{i+1}. **{data['topic1']} ({data['agency1']})** and **{data['topic2']} ({data['agency2']})**")
                report_lines.append(f"   - Confusion Rate: {data['confusion_rate']:.4f}")
                report_lines.append(f"   - Confusable Pairs: {data['confusion_count']} out of {data['total_possible']} possible\n")
        
        
        report_lines.append("## Interpretation\n")
        
        overall_rate = confusion_results['overall_confusion_rate']
        if overall_rate < 0.01:
            interpretation = "The overall confusion is **very low**. The classifier should have no significant issues distinguishing between these entities."
        elif overall_rate < 0.05:
            interpretation = "The overall confusion is **low**. The classifier may occasionally confuse some queries, but this should be minimal."
        elif overall_rate < 0.10:
            interpretation = "The overall confusion is **moderate**. Some specific query types may be difficult to classify correctly."
        elif overall_rate < 0.20:
            interpretation = "The overall confusion is **high**. There are significant areas of overlap that may lead to classification errors."
        else:
            interpretation = "The overall confusion is **very high**. The classifier may struggle to reliably distinguish between these entities."
        
        report_lines.append(f"- **Overall Confusion**: {interpretation}")
        
        report_lines.append("\n")
        
        report_lines.append(f"## Detailed {entity_name} Pair Analysis\n")
        

        valid_pairs = []
        for pair, result in pair_results.items():
            if isinstance(result, dict) and "confusion_rate" in result:
                valid_pairs.append((pair, result))
        
        sorted_pairs = sorted(valid_pairs, 
                            key=lambda x: x[1]["confusion_rate"], 
                            reverse=True)
        
        for pair, result in sorted_pairs:
            entity1, entity2 = pair
            report_lines.append(f"### {entity1} vs {entity2}\n")
            report_lines.append(f"- **Confusion Rate**: {result['confusion_rate']:.4f}")
            report_lines.append(f"- **Average Similarity**: {result['average_similarity']:.4f}")
            report_lines.append(f"- **Maximum Similarity**: {result['max_similarity']:.4f}")
            report_lines.append(f"- **Confusable Pairs**: {result['confusion_count']} out of {result['total_possible_pairs']} possible\n")
            
            if 'confusion_pairs' in result and result['confusion_pairs']:
                report_lines.append("#### Example Confusable Queries\n")
                report_lines.append("| Similarity | " + f"{entity1} Query | {entity2} Query |")
                report_lines.append("|------------|" + "-" * (len(entity1) + 8) + "|" + "-" * (len(entity2) + 8) + "|")
                
                for pair_item in result['confusion_pairs']:
                    if not isinstance(pair_item, dict):
                        continue
                        
                    query1_key = f"{entity_label}1_query"
                    query2_key = f"{entity_label}2_query"
                    
                    if query1_key not in pair_item or query2_key not in pair_item:
                        continue
                        
                    query1 = pair_item[query1_key][:50] + ("..." if len(pair_item[query1_key]) > 50 else "")
                    query2 = pair_item[query2_key][:50] + ("..." if len(pair_item[query2_key]) > 50 else "")
                    similarity = pair_item.get('similarity', 0.0)
                    
                    report_lines.append(f"| {similarity:.4f} | {query1} | {query2} |")
                
                report_lines.append("\n")
            

            if is_agency_level and isinstance(result, dict) and "topic_pair_metrics" in result:
                topic_pair_metrics = result["topic_pair_metrics"]
                
                if isinstance(topic_pair_metrics, dict) and topic_pair_metrics:
                    report_lines.append("#### Topic-Level Confusion\n")
                    
   
                    valid_topic_pairs = []
                    for tp_key, tp_value in topic_pair_metrics.items():
                        if isinstance(tp_value, dict) and "confusion_rate" in tp_value:
                            valid_topic_pairs.append((tp_key, tp_value))
                    
                    sorted_topic_pairs = sorted(
                        valid_topic_pairs,
                        key=lambda x: x[1]["confusion_rate"],
                        reverse=True
                    )
                    
                    for (topic1, topic2), topic_metrics in sorted_topic_pairs[:5]:
                        confusion_rate = topic_metrics.get("confusion_rate", 0)
                        if confusion_rate > 0:
                            report_lines.append(f"- **{topic1}** vs **{topic2}**")
                            report_lines.append(f"  - Confusion Rate: {confusion_rate:.4f}")
                            report_lines.append(f"  - Confusable Pairs: {topic_metrics.get('confusion_count', 0)} out of {topic_metrics.get('total_possible_pairs', 0)} possible\n")
        
        report_lines.append("## Recommendations\n")
        
        if overall_rate < 0.05:
            report_lines.append("- No significant changes needed; the confusion level is acceptable.")
        else:
            report_lines.append("Recommendations to reduce confusion:")
            
            most_confusable = confusion_results['most_confusable_pair']
            if most_confusable and isinstance(most_confusable[1], dict) and most_confusable[1].get('confusion_rate', 0) > 0.1:
                pair_names = most_confusable[0]
                report_lines.append(f"- Focus on differentiating {pair_names[0]} and {pair_names[1]} queries with more distinct examples.")
                
            report_lines.append("- Review and rewrite the most similar queries to make them more clearly distinct.")
            report_lines.append("- Consider adding more domain-specific terminology to queries to help the classifier distinguish between entities.")
            report_lines.append("- For queries that are inherently ambiguous, consider creating specific examples for the classifier to learn from.")
        
        full_report = "\n".join(report_lines)
        
        if output_file:
            try:
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                
                logger.info(f"Confusion report written to {output_file}")
            except Exception as e:
                logger.error(f"Error writing confusion report: {e}")
        else:
            logger.info(full_report)
        
        return full_report


def read_file(file_path: str) -> str:
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def read_conversations_from_directory(directory: str, pattern: str = "conversation_*.txt") -> Tuple[List[str], List[str]]:
    """
    Read all conversation files from a directory matching a pattern.

    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    contents = []
    filenames = []
    
    for f in files:
        content = read_file(f)
        if content.strip():  
            contents.append(content)
            filenames.append(os.path.basename(f))
    
    return contents, filenames

def read_agency_topic_conversations(agency_dir: str) -> Dict[str, List[str]]:
    """
    Read all conversations from topic directories within an agency directory.
    
    Args:
        agency_dir: Path to the agency directory containing topic subdirectories
        
    Returns:
        Dictionary mapping topic names to lists of conversations
    """
    topic_dirs = [d for d in glob.glob(os.path.join(agency_dir, "*")) if os.path.isdir(d)]
    
    topic_conversations = {}
    
    for topic_dir in topic_dirs:
        topic_name = os.path.basename(topic_dir)
        conversations, _ = read_conversations_from_directory(topic_dir)
        
        if conversations:
            topic_conversations[topic_name] = conversations
            logger.info(f"Read {len(conversations)} conversations for topic {topic_name}")
        else:
            logger.warning(f"No conversations found for topic {topic_name} in {topic_dir}")
    
    return topic_conversations

def analyze_cross_agency_confusion(agency_dirs: Dict[str, str], 
                                 output_dir: str = None,
                                 confusion_threshold: float = 0.85) -> Dict:
    """
    Analyze confusion between different agencies' conversations.
    
    """
    agency_topic_conversations = {}
    
    for agency_name, agency_dir in agency_dirs.items():
        topic_conversations = read_agency_topic_conversations(agency_dir)
        
        if topic_conversations:
            agency_topic_conversations[agency_name] = topic_conversations
            total_conversations = sum(len(convs) for convs in topic_conversations.values())
            total_topics = len(topic_conversations)
            logger.info(f"Read {total_conversations} conversations across {total_topics} topics for {agency_name}")
        else:
            logger.warning(f"No topic conversations found for {agency_name} in {agency_dir}")
    
    if len(agency_topic_conversations) < 2:
        logger.error("Need at least two agencies with conversations to analyze confusion")
        return {"error": "Need at least two agencies with conversations"}
    
    analyzer = AgencyConfusionAnalyzer(confusion_threshold=confusion_threshold)
    
    # Analyze cross-agency confusion
    results = analyzer.analyze_cross_agency_confusion(agency_topic_conversations)
    
    # Generate outputs if directory specified
    if output_dir and "error" not in results:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report
        report_file = os.path.join(output_dir, "cross_agency_confusion_report.md")
        analyzer.generate_confusion_report(results, report_file, is_agency_level=True)
        
        # Generate visualization
        viz_file = os.path.join(output_dir, "cross_agency_confusion_matrix.png")
        analyzer.generate_confusion_matrix_visualization(results, viz_file, is_agency_level=True)
        
        # Generate topic-level visualization
        topic_viz_file = os.path.join(output_dir, "cross_agency_topic_confusion.png")
        analyzer.generate_topic_pair_heatmap(results, topic_viz_file)
        
        # Add output files to results
        results["output_files"] = {
            "report": report_file,
            "agency_visualization": viz_file,
            "topic_visualization": topic_viz_file
        }
    
    return results

def analyze_within_agency_topic_confusion(agency_dir: str,
                                        agency_name: str = None,
                                        output_dir: str = None,
                                        confusion_threshold: float = 0.85) -> Dict:
    """
    Analyze confusion between different topics within the same agency.
"""
    if not agency_name:
        agency_name = os.path.basename(agency_dir)
    
    topic_conversations = read_agency_topic_conversations(agency_dir)
    
    if len(topic_conversations) < 2:
        logger.error(f"Need at least two topics with conversations to analyze confusion within {agency_name}")
        return {"error": f"Need at least two topics with conversations within {agency_name}"}
    
    analyzer = AgencyConfusionAnalyzer(confusion_threshold=confusion_threshold)
    
    results = analyzer.analyze_topic_confusion_within_agency(topic_conversations, agency_name)
    
    if output_dir and "error" not in results:
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, f"{agency_name}_topic_confusion_report.md")
        analyzer.generate_confusion_report(results, report_file, is_agency_level=False)
        
        viz_file = os.path.join(output_dir, f"{agency_name}_topic_confusion_matrix.png")
        analyzer.generate_confusion_matrix_visualization(results, viz_file, is_agency_level=False)
        
        results["output_files"] = {
            "report": report_file,
            "visualization": viz_file
        }
    
    return results


if __name__ == "__main__":


    agency1_dir = "data/ID.ee"
    agency2_dir = "data/Politsei-_ja_Piirivalveamet"
    agency1_name = "ID.ee"
    agency2_name = "Politsei"
    output_dir = "data/confusion_analysis"
    confusion_threshold = 0.85
    within_agency=False
    if within_agency:
        results = analyze_within_agency_topic_confusion(
            agency_dir=agency1_dir,
            agency_name=agency1_name,
            output_dir=output_dir,
            confusion_threshold=confusion_threshold
        )
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Topic Confusion Analysis for {agency1_name}:")
            print(f"Overall confusion rate: {results['overall_confusion_rate']:.4f}")
            print(f"Total confusable pairs: {results['total_confusion_pairs']}")
            
            if results['most_confusable_pair']:
                pair = results['most_confusable_pair'][0]
                metrics = results['most_confusable_pair'][1]
                print(f"Most confusable topics: {pair[0]} and {pair[1]} (confusion rate: {metrics['confusion_rate']:.4f})")
            
            if "output_files" in results:
                print(f"\nReport saved to: {results['output_files']['report']}")
                print(f"Visualization saved to: {results['output_files']['visualization']}")
    else:
        agency_dirs = {
            agency1_name: agency1_dir,
            agency2_name: agency2_dir
        }
        
        results = analyze_cross_agency_confusion(
            agency_dirs=agency_dirs,
            output_dir=output_dir,
            confusion_threshold=confusion_threshold
        )
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("Cross-Agency Confusion Analysis:")
            print(f"Overall confusion rate: {results['overall_confusion_rate']:.4f}")
            print(f"Overall similarity: {results['overall_similarity']:.4f}")
            print(f"Total confusable pairs: {results['total_confusion_pairs']}")
            
            if results['most_confusable_pair']:
                pair = results['most_confusable_pair'][0]
                metrics = results['most_confusable_pair'][1]
                print(f"Most confusable agencies: {pair[0]} and {pair[1]} (confusion rate: {metrics['confusion_rate']:.4f})")
            
            if "output_files" in results:
                print(f"\nReport saved to: {results['output_files']['report']}")
                print(f"Agency visualization saved to: {results['output_files']['agency_visualization']}")
                print(f"Topic visualization saved to: {results['output_files']['topic_visualization']}")