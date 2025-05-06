# Evaluation Metrics 

## Overview

The Evaluation Metrics is a comprehensive framework designed to assess the quality of generated conversations. This pipeline combines multiple complementary metrics that evaluate different aspects of conversation quality and dataset completeness, enabling thorough assessment of training data for a robust classification system.

## Pipeline Architecture

The evaluation pipeline consists of nine specialized metrics that collectively cover all critical aspects of conversation and dataset quality:

| Metric | Focus | Scope | What It Measures |
|--------|-------|-------|------------------|
| Information Coverage | Content | Single Conversation | How well a conversation covers key information from source material |
| Redundancy Analysis | Uniqueness | Conversation Set | Repetition within conversations and similarity between conversations |
| Relevance Score | Alignment | Single Conversation | How well a conversation aligns with its intended topic |
| Length Appropriateness | Structure | Single Conversation | Whether a conversation has suitable length for its topic complexity |
| Topic Consistency | Coherence | Single Conversation | How well a conversation maintains focus on a single topic |
| Topic Coverage Gap | Completeness | Dataset | Whether all important topics are represented by conversations |
| Agency Confusion | Distinctiveness | Cross-Agency | Whether conversations from different agencies are sufficiently distinct |
| Query Diversity | Variation | Topic/Agency | Variety of ways user queries are formulated for the same topics |
| Qualitative Evaluation | Human-like | Single Conversation | Subjective quality aspects assessed by language model |

## Detailed Description of Metrics

### 1. Information Coverage Score

**Purpose:** Evaluates how well a conversation incorporates key information from reference documents.

**Implementation:**
- Uses semantic similarity to measure coverage of source document information
- Segments both conversation and reference documents into chunks
- Reports proportion of reference chunks covered in the conversation

**Interpretation:**
- 0.8-1.0: Excellent coverage
- 0.6-0.8: Good coverage
- 0.4-0.6: Moderate coverage
- 0.2-0.4: Limited coverage
- 0.0-0.2: Poor coverage

**Dependencies:**
- Requires source document for each topic
- Uses `paraphrase-multilingual-mpnet-base-v2` model

### 2. Redundancy Analysis

**Purpose:** Identifies repetitive content within and between conversations.

**Implementation:**
- Intra-conversation: Measures repetition within single conversations
- Inter-conversation: Measures similarity between different conversations
- Generates similarity heatmaps to visualize redundancy patterns

**Interpretation:**
- Intra-conversation: Score below 0.1 is excellent (minimal repetition)
- Inter-conversation: Score below 0.1 is excellent (high diversity)

**Dependencies:**
- Works on both individual conversations and conversation sets
- Uses `paraphrase-multilingual-mpnet-base-v2` model

### 3. Relevance Score

**Purpose:** Evaluates how well a conversation addresses its intended topic.

**Implementation:**
- Multi-faceted analysis using segment-level, query-focused, and key term approaches
- Weighted scoring system (60% segment relevance, 30% query relevance, 10% terminology)
- Prioritizes user queries as most important for classification

**Interpretation:**
- 0.8-1.0: Excellent relevance
- 0.7-0.8: Good relevance
- 0.5-0.7: Acceptable relevance
- 0.3-0.5: Poor relevance
- 0.0-0.3: Irrelevant

**Dependencies:**
- Requires topic documents
- Uses `paraphrase-multilingual-mpnet-base-v2` model

### 4. Length Appropriateness

**Purpose:** Ensures conversations have suitable length for their topic complexity.

**Implementation:**
- Analyzes topic complexity using vocabulary diversity and information density
- Determines optimal turn count based on complexity
- Intelligently detects conversation turns using multiple patterns

**Interpretation:**
- 0.9-1.0: Excellent length appropriateness
- 0.7-0.9: Good length appropriateness
- 0.5-0.7: Acceptable length
- 0.3-0.5: Poor length (too short or too long)
- 0.0-0.3: Inadequate length

**Dependencies:**
- Works on single conversations
- Requires topic documents for complexity assessment

### 5. Topic Consistency

**Purpose:** Evaluates how well a conversation maintains focus on a single topic.

**Implementation:**
- Combines coherence (internal consistency) and alignment (topic focus)
- Analyzes turn-to-turn semantic similarity
- Compares conversation to source document using multiple similarity measures

**Interpretation:**
- 0.8-1.0: Excellent consistency
- 0.6-0.8: Good consistency
- 0.4-0.6: Acceptable consistency
- 0.2-0.4: Poor consistency
- 0.0-0.2: Inadequate consistency

**Dependencies:**
- Requires topic documents
- Uses `paraphrase-multilingual-mpnet-base-v2` model

### 6. Topic Coverage Gap Analysis

**Purpose:** Identifies topics from source documents that lack corresponding conversations.

**Implementation:**
- Extracts distinct topics from source documents using clustering
- Maps conversations to topics using semantic similarity
- Identifies uncovered topics or underrepresented information areas

**Interpretation:**
- Coverage percentage should be at least 90%
- Uncovered topics list should ideally be empty
- All topics should have at least one conversation with similarity > 0.5

**Dependencies:**
- Requires complete source documents
- Uses `paraphrase-multilingual-mpnet-base-v2` model
- Utilizes DBSCAN clustering

### 7. Agency Confusion Analysis

**Purpose:** Identifies potential classification confusion between different agencies.

**Implementation:**
- Compares first user queries across agencies and topics
- Identifies highly similar queries that might cause misclassification
- Reports confusion rates and highlights most confusable agency and topic pairs

**Interpretation:**
- Confusion rate < 0.01: Very low confusion
- Confusion rate < 0.05: Low confusion
- Confusion rate < 0.10: Moderate confusion
- Confusion rate < 0.20: High confusion
- Confusion rate > 0.20: Very high confusion

**Dependencies:**
- Requires conversations from multiple agencies
- Uses `paraphrase-multilingual-mpnet-base-v2` model

### 8. Query Diversity

**Purpose:** Evaluates the variety of ways user queries are formulated for the same topics.

**Implementation:**
- Analyzes lexical diversity (vocabulary variation)
- Measures semantic diversity (intent variation)
- Identifies distinct query clusters within topics

**Interpretation:**
- Score > 0.8: Excellent diversity
- Score > 0.6: Good diversity
- Score > 0.4: Moderate diversity
- Score > 0.2: Limited diversity
- Score < 0.2: Poor diversity

**Dependencies:**
- Focuses specifically on user queries
- Uses `paraphrase-multilingual-mpnet-base-v2` model

### 9. Qualitative Evaluation

**Purpose:** Provides human-like assessment of subjective conversation quality aspects.

**Implementation:**
- Uses lightweight LLM (Gemma-2B) to evaluate multiple quality dimensions
- Generates numerical scores (1-5) with explanatory reasoning
- Identifies strengths, weaknesses, and improvement suggestions

**Interpretation:**
- 4.5-5.0: Excellent quality
- 4.0-4.4: Good quality
- 3.0-3.9: Acceptable quality
- 2.0-2.9: Poor quality
- 1.0-1.9: Inadequate quality

**Dependencies:**
- Requires GPU with 9GB VRAM or can run on CPU
- Uses Google's Gemma-2B-IT model

## Integration and Execution Flow

The evaluation pipeline follows a hierarchical structure that matches your directory organization:

1. **Conversation Level:**
   - Information Coverage, Relevance, Length Appropriateness, Topic Consistency
   - Qualitative Evaluation

2. **Topic Level:**
   - Redundancy Analysis, Query Diversity
   - Topic-specific aggregation of conversation-level metrics

3. **Agency Level:**
   - Topic Coverage Gap Analysis
   - Cross-topic metrics aggregation

4. **Cross-Agency Level:**
   - Agency Confusion Analysis
   - Comparative metrics across agencies

## Pipeline Completeness

The current pipeline provides comprehensive coverage of all critical aspects of conversation quality and dataset completeness:

| Aspect | Covered By | Completeness |
|--------|------------|--------------|
| Content Quality | Information Coverage, Relevance, Qualitative Evaluation | Complete |
| Structural Quality | Length Appropriateness, Topic Consistency | Complete |
| Diversity & Uniqueness | Redundancy Analysis, Query Diversity | Complete |
| Dataset Completeness | Topic Coverage Gap Analysis | Complete |
| Classification Robustness | Agency Confusion Analysis | Complete |
| Subjective Quality | Qualitative Evaluation | Complete |

## Usage Recommendations

For optimal evaluation results:

1. **Initial Dataset Assessment:**
   - Run Topic Coverage Gap Analysis first to ensure comprehensive topic coverage
   - Use Agency Confusion Analysis to identify potential classification issues

2. **Individual Conversation Quality:**
   - Apply Information Coverage, Relevance, Length Appropriateness, and Topic Consistency
   - Use Qualitative Evaluation for subjective assessment

3. **Dataset Diversity Checks:**
   - Employ Redundancy Analysis to identify overly similar conversations
   - Use Query Diversity to ensure varied user input formulations

4. **Prioritizing Improvements:**
   - Address gaps in topic coverage first
   - Then focus on reducing agency confusion
   - Finally, improve individual conversation quality


