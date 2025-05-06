import os
import json
from loguru import logger
import sys

import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")



# Set default model to use
DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"  # Small enough for 9GB VRAM


class TopicAgencyFitEvaluator:
    """
    A qualitative evaluator that assesses how well conversations fit their intended topics
    and agencies, using a lightweight model that can run on a GPU with 9GB VRAM.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_4bit: bool = True,
        cpu_only: bool = False,
        device: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """
        Initialize the evaluator with a lightweight model.
        """
        if device is None:
            if cpu_only:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing evaluator with model: {model_name} on {device}")

        # Configure quantization if using 4-bit precision
        if use_4bit and device == "cuda":
            logger.info("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure model
        model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Set generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        # Define evaluation criteria templates
        self.criteria_templates = self._define_evaluation_criteria()

    def _define_evaluation_criteria(self) -> Dict[str, str]:
        """Define the evaluation criteria templates."""
        return {
            "topic_fit": (
                "Evaluate how well this conversation aligns with the given topic on a scale of 1-5. "
                "Consider whether the conversation stays focused on the topic, addresses key aspects of the topic, "
                "and provides information that would be expected for this topic. "
                "Format your response as: 'Score: X\\nReasoning: Your detailed reasoning here'"
            ),
            "agency_fit": (
                "Evaluate how appropriate this conversation is for the specified government agency on a scale of 1-5. "
                "Consider whether the conversation addresses services, questions, or issues that would typically be "
                "handled by this agency. Does it match the agency's scope and responsibilities? "
                "Format your response as: 'Score: X\\nReasoning: Your detailed reasoning here'"
            ),
            "classification_confidence": (
                "How confidently could an automated system classify this conversation as belonging to the given "
                "topic and agency? Rate on a scale of 1-5, where 1 is 'Very difficult to classify correctly' and "
                "5 is 'Very easy to classify correctly'. Consider how distinctive the conversation is and whether "
                "it contains clear indicators of the topic and agency. "
                "Format your response as: 'Score: X\\nReasoning: Your detailed reasoning here'"
            ),
            "potential_confusion": (
                "Identify any other topics or agencies that this conversation might be confused with. "
                "Explain why confusion might occur and suggest ways to make the conversation more distinctly "
                "about its intended topic and agency."
            ),
            "key_indicators": (
                "List the specific words, phrases, or conversation elements that most strongly indicate "
                "this conversation belongs to the given topic and agency. What are the most distinctive aspects?"
            ),
        }

    def _extract_score(self, response: str) -> Tuple[Optional[float], str]:
        """
        Extract numerical score from a generated response.


        """
        try:
            if "Score:" in response and "\n" in response:
                score_part, reasoning = response.split("\n", 1)
                score_str = score_part.replace("Score:", "").strip()

                score = float(score_str)

                if 1 <= score <= 5:
                    return score, reasoning.strip()

            import re

            number_matches = re.findall(r"\b([1-5](?:\.\d)?)\b", response[:50])
            if number_matches:
                return float(number_matches[0]), response

            return None, response
        except:
            return None, response

    def _clean_conversation(self, conversation: str) -> str:
        """
        Clean and format the conversation text for evaluation.

        Args:
            conversation: The  conversation text

        Returns:
            Cleaned conversation text
        """
        # Basic cleaning
        conversation = conversation.strip()

        # Make sure user/assistant markers are clear
        conversation = conversation.replace("**Kasutaja**:", "User:")
        conversation = conversation.replace("**Robot**:", "Assistant:")
        conversation = conversation.replace("Kasutaja:", "User:")
        conversation = conversation.replace("Robot:", "Assistant:")

        # Ensure there are line breaks between turns
        conversation = conversation.replace("User:", "\nUser:")
        conversation = conversation.replace("Assistant:", "\nAssistant:")

        return conversation

    def read_topic_description(self, topic_file: str) -> str:
        """
        Read and process a topic description file.

        Args:
            topic_file: Path to the topic description file

        Returns:
            Processed topic description
        """
        try:
            with open(topic_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Basic processing to extract key information
            # Extract the first 1000 characters if very long
            if len(content) > 2000:
                content = content[:2000] + "... [content truncated]"

            return content
        except Exception as e:
            logger.error(f"Error reading topic file {topic_file}: {e}")
            return "No topic description available."

    def read_agency_description(self, agency_name: str) -> str:
        """
        Get a description of the agency based on its name.

        Args:
            agency_name: Name of the agency

        Returns:
            Agency description
        """
        # This could be expanded to read from actual files describing agencies
        agency_descriptions = {
            "ID.ee": (
                "ID.ee provides information and services related to Estonian digital identity, "
                "including ID cards, Mobile-ID, Smart-ID, and digital signatures. They handle "
                "questions about applying for, using, and troubleshooting Estonian digital "
                "identity services."
            ),
            "Politsei-_ja_Piirivalveamet": (
                "The Police and Border Guard Board (Politsei- ja Piirivalveamet) handles law enforcement, "
                "border control, citizenship and migration issues. They provide services related to "
                "passports, identity documents, residence permits, citizenship applications, and "
                "reporting crimes or incidents."
            ),
        }

        # Extract agency name from path if needed
        if "/" in agency_name or "\\" in agency_name:
            agency_name = os.path.basename(agency_name)

        # Try to match with known descriptions
        for known_agency, description in agency_descriptions.items():
            if (
                known_agency.lower() in agency_name.lower()
                or agency_name.lower() in known_agency.lower()
            ):
                return description

        # If no match, provide a generic description
        return f"This is an Estonian government agency named '{agency_name}'."


    def evaluate_topic_agency_fit(
        self,
        conversation: str,
        topic_name: str,
        topic_description: str,
        agency_name: str,
        agency_description: str,
        criteria: str = "topic_fit",
    ) -> Dict[str, Any]:
        """
        Evaluate how well a conversation fits a topic and agency.

        Args:
            conversation: The conversation text
            topic_name: Name of the topic
            topic_description: Description of the topic
            agency_name: Name of the agency
            agency_description: Description of the agency
            criteria: The evaluation criteria to use

        Returns:
            Dictionary with evaluation results
        """
        # Clean the conversation
        clean_conv = self._clean_conversation(conversation)

        # Get the evaluation prompt
        prompt_template = self.criteria_templates.get(criteria)
        if not prompt_template:
            logger.error(f"Unknown criteria: {criteria}")
            return {"error": f"Unknown criteria: {criteria}"}

        # Construct the full prompt
        prompt = (
            f"You are an expert conversation evaluator for Estonian government agencies.\n\n"
            f"AGENCY: {agency_name}\n"
            f"AGENCY DESCRIPTION: {agency_description}\n\n"
            f"TOPIC: {topic_name}\n"
            f"TOPIC DESCRIPTION: {topic_description}\n\n"
            f"CONVERSATION TO EVALUATE:\n{clean_conv}\n\n"
            f"EVALUATION TASK:\n{prompt_template}\n\n"
            f"YOUR EVALUATION:"
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate evaluation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
            )

        # Decode the response and extract only the generated part
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt) :].strip()

        # Extract score if applicable
        score, reasoning = self._extract_score(response)

        return {
            "criteria": criteria,
            "score": score,
            "response": response,
            "reasoning": reasoning if score is not None else response,
        }

    def evaluate_conversation_topic_agency_fit(
        self,
        conversation: str,
        topic_name: str,
        topic_description: str,
        agency_name: str,
        agency_description: str,
    ) -> Dict[str, Any]:
        """
        Perform a full evaluation of a conversation's fit to topic and agency.

        Args:
            conversation: The conversation text
            topic_name: Name of the topic
            topic_description: Description of the topic
            agency_name: Name of the agency
            agency_description: Description of the agency

        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {}
        numerical_scores = {}

        # Evaluate score-based criteria
        score_criteria = ["topic_fit", "agency_fit", "classification_confidence"]

        for criteria in tqdm(score_criteria, desc="Evaluating fit criteria"):
            eval_result = self.evaluate_topic_agency_fit(
                conversation,
                topic_name,
                topic_description,
                agency_name,
                agency_description,
                criteria,
            )
            results[criteria] = eval_result

            if eval_result["score"] is not None:
                numerical_scores[criteria] = eval_result["score"]

        # Evaluate non-score criteria
        results["potential_confusion"] = self.evaluate_topic_agency_fit(
            conversation,
            topic_name,
            topic_description,
            agency_name,
            agency_description,
            "potential_confusion",
        )

        results["key_indicators"] = self.evaluate_topic_agency_fit(
            conversation,
            topic_name,
            topic_description,
            agency_name,
            agency_description,
            "key_indicators",
        )

        # Calculate aggregate score
        if numerical_scores:
            # Weight classification confidence slightly higher
            weights = {
                "topic_fit": 0.35,
                "agency_fit": 0.35,
                "classification_confidence": 0.3,
            }

            weighted_sum = sum(
                numerical_scores.get(k, 0) * v
                for k, v in weights.items()
                if k in numerical_scores
            )
            total_weight = sum(v for k, v in weights.items() if k in numerical_scores)

            aggregate_score = weighted_sum / total_weight if total_weight > 0 else None
        else:
            aggregate_score = None

        return {
            "topic_name": topic_name,
            "agency_name": agency_name,
            "detailed_scores": results,
            "numerical_scores": numerical_scores,
            "aggregate_score": aggregate_score,
        }


def path_to_str(obj):
    """
    Convert any Path objects to strings for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with all Path objects converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [path_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(path_to_str(item) for item in obj)
    else:
        return obj


def find_topic_description_file(topic_dir: str) -> Optional[str]:
    """
    Find a suitable topic description file in the specified directory or its parent.

    Args:
        topic_dir: Path to the topic directory

    Returns:
        Path to the topic description file, or None if not found
    """
    # Check for common topic description file patterns in the topic directory
    topic_name = os.path.basename(topic_dir)
    parent_dir = os.path.dirname(topic_dir)

    # Possible patterns for topic description files
    potential_files = [
        # In the topic directory
        os.path.join(topic_dir, "description.txt"),
        os.path.join(topic_dir, f"{topic_name}.txt"),
        os.path.join(topic_dir, "topic.txt"),
        # In the parent directory
        os.path.join(parent_dir, f"{topic_name}.txt"),
        os.path.join(parent_dir, f"{topic_name}_description.txt"),
        os.path.join(
            parent_dir,
            "output_" + os.path.basename(parent_dir),
            f"{topic_name}_-_{os.path.basename(parent_dir)}.txt",
        ),
        os.path.join(
            parent_dir,
            "..",
            "data",
            "output_" + os.path.basename(parent_dir),
            f"{topic_name}_-_{os.path.basename(parent_dir)}.txt",
        ),
    ]

    # Try each potential file
    for file_path in potential_files:
        if os.path.isfile(file_path):
            logger.info(f"Found topic description file: {file_path}")
            return file_path

    logger.warning(f"No topic description file found for {topic_name}")
    return None


# Modify these functions in your code to handle explicit conversation paths


def evaluate_topic_directory(
    evaluator: TopicAgencyFitEvaluator,
    topic_dir: str,
    agency_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
    conversation_path: Optional[str] = None,  # Add this parameter
) -> Dict[str, Any]:
    """
    Evaluate all conversations in a topic directory for topic and agency fit.

    Args:
        evaluator: The evaluator instance
        topic_dir: Path to the topic directory
        agency_dir: Path to the agency directory
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files
        conversation_path: Optional explicit path to conversation files (overrides topic_dir)

    Returns:
        Dictionary with evaluation summary
    """
    # Find conversation files - use explicit path if provided
    if conversation_path:
        # Use the explicit conversation path
        conversation_files = sorted(list(Path(conversation_path).glob(pattern)))
        logger.info(f"Looking for conversations in explicit path: {conversation_path}")
    else:
        # Look in the topic directory (original behavior)
        conversation_files = sorted(list(Path(topic_dir).glob(pattern)))
        logger.info(f"Looking for conversations in topic directory: {topic_dir}")

    if not conversation_files:
        logger.warning(f"No conversation files matching pattern '{pattern}' found")
        return {"error": f"No conversation files found"}

    # Create output directory with topic name
    topic_name = os.path.basename(topic_dir)
    topic_output_dir = os.path.join(output_dir, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    # Evaluate each conversation
    all_results = []
    for file_path in conversation_files:
        try:
            result = evaluate_conversation_file(
                evaluator, str(file_path), topic_dir, agency_dir, topic_output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")

    # Calculate summary statistics
    topic_summary = {
        "topic_name": topic_name,
        "agency_name": os.path.basename(agency_dir),
        "total_conversations": len(all_results),
        "conversation_files": [r.get("file_name") for r in all_results],
    }

    # Calculate average scores
    scores = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        valid_scores = [
            r.get("numerical_scores", {}).get(criteria)
            for r in all_results
            if r.get("numerical_scores", {}).get(criteria) is not None
        ]
        if valid_scores:
            scores[criteria] = {
                "average": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
            }

    # Calculate aggregate score
    aggregate_scores = [
        r.get("aggregate_score")
        for r in all_results
        if r.get("aggregate_score") is not None
    ]
    if aggregate_scores:
        topic_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )
        topic_summary["min_aggregate_score"] = min(aggregate_scores)
        topic_summary["max_aggregate_score"] = max(aggregate_scores)

    topic_summary["scores"] = scores

    # Save topic summary
    with open(
        os.path.join(topic_output_dir, "topic_fit_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(path_to_str(topic_summary), f, ensure_ascii=False, indent=2)

    # Generate a readable report
    _generate_topic_fit_report(
        topic_summary,
        all_results,
        os.path.join(topic_output_dir, "topic_fit_report.md"),
    )

    return topic_summary


# Modified functions to work with flat directory structure


def find_topic_files(agency_dir: str) -> List[Tuple[str, str]]:
    """
    Find topic files in an agency directory with flat structure.

    Args:
        agency_dir: Path to the agency directory

    Returns:
        List of tuples: (topic_filename, full_path_to_topic_file)
    """
    # Look for .txt files directly in the agency directory
    topic_files = []

    # Check if directory exists
    if not os.path.isdir(agency_dir):
        logger.warning(f"Agency directory not found: {agency_dir}")
        return []

    # Find all .txt files in the agency directory
    for file_path in Path(agency_dir).glob("*.txt"):
        topic_name = file_path.stem  # Get filename without extension
        topic_files.append((topic_name, str(file_path)))

    logger.info(f"Found {len(topic_files)} topic files in {agency_dir}")
    return topic_files


def evaluate_topic_file(
    evaluator: TopicAgencyFitEvaluator,
    topic_name: str,
    topic_file_path: str,
    agency_dir: str,
    conversation_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
) -> Dict[str, Any]:
    """
    Evaluate conversations for a topic based on a topic file.

    Args:
        evaluator: The evaluator instance
        topic_name: Name of the topic (filename without extension)
        topic_file_path: Path to the topic file
        agency_dir: Path to the agency directory
        conversation_dir: Path to the conversation directory
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files

    Returns:
        Dictionary with evaluation summary
    """
    # Construct path to conversation directory for this topic
    topic_conversation_dir = os.path.join(
        conversation_dir, os.path.basename(agency_dir), topic_name
    )

    # Check if the conversation directory exists
    if not os.path.isdir(topic_conversation_dir):
        logger.warning(f"Conversation directory not found: {topic_conversation_dir}")
        return {
            "topic_name": topic_name,
            "error": f"Conversation directory not found: {topic_conversation_dir}",
        }

    # Find conversation files
    conversation_files = sorted(list(Path(topic_conversation_dir).glob(pattern)))

    if not conversation_files:
        logger.warning(
            f"No conversation files matching pattern '{pattern}' found in {topic_conversation_dir}"
        )
        return {
            "topic_name": topic_name,
            "error": f"No conversation files found in {topic_conversation_dir}",
        }

    # Create output directory with topic name
    agency_name = os.path.basename(agency_dir)
    topic_output_dir = os.path.join(output_dir, agency_name, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    # Read topic description
    topic_description = evaluator.read_topic_description(topic_file_path)
    agency_description = evaluator.read_agency_description(agency_name)

    # Evaluate each conversation
    all_results = []
    for file_path in conversation_files:
        try:
            # Read conversation
            with open(file_path, "r", encoding="utf-8") as f:
                conversation = f.read()

            # Evaluate conversation
            logger.info(f"Evaluating conversation: {file_path.name}")
            result = evaluator.evaluate_conversation_topic_agency_fit(
                conversation,
                topic_name,
                topic_description,
                agency_name,
                agency_description,
            )

            # Add file information
            result["file_name"] = file_path.name
            result["file_path"] = str(file_path)

            # Save individual result
            file_stem = file_path.stem
            output_file = os.path.join(
                topic_output_dir, f"{file_stem}_topic_agency_fit.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(path_to_str(result), f, ensure_ascii=False, indent=2)

            all_results.append(result)

        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")

    # Calculate summary statistics
    topic_summary = {
        "topic_name": topic_name,
        "agency_name": agency_name,
        "total_conversations": len(all_results),
        "conversation_files": [r.get("file_name") for r in all_results],
    }

    # Calculate average scores
    scores = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        valid_scores = [
            r.get("numerical_scores", {}).get(criteria)
            for r in all_results
            if r.get("numerical_scores", {}).get(criteria) is not None
        ]
        if valid_scores:
            scores[criteria] = {
                "average": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
            }

    # Calculate aggregate score
    aggregate_scores = [
        r.get("aggregate_score")
        for r in all_results
        if r.get("aggregate_score") is not None
    ]
    if aggregate_scores:
        topic_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )
        topic_summary["min_aggregate_score"] = min(aggregate_scores)
        topic_summary["max_aggregate_score"] = max(aggregate_scores)

    topic_summary["scores"] = scores

    # Save topic summary
    with open(
        os.path.join(topic_output_dir, "topic_fit_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(path_to_str(topic_summary), f, ensure_ascii=False, indent=2)

    # Generate a readable report
    _generate_topic_fit_report(
        topic_summary,
        all_results,
        os.path.join(topic_output_dir, "topic_fit_report.md"),
    )

    return topic_summary


def evaluate_agency_directory_flat(
    evaluator: TopicAgencyFitEvaluator,
    agency_dir: str,
    conversation_base_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
) -> Dict[str, Any]:
    """
    Evaluate all topics in an agency directory with flat structure.

    Args:
        evaluator: The evaluator instance
        agency_dir: Path to the agency directory with topic files
        conversation_base_dir: Base directory containing conversation files
        output_dir: Directory to save evaluation results
        pattern: File pattern to match conversation files

    Returns:
        Dictionary with agency evaluation summary
    """
    # Find topic files in the agency directory
    topic_files = find_topic_files(agency_dir)

    if not topic_files:
        logger.warning(f"No topic files found in {agency_dir}")
        return {"error": f"No topic files found in {agency_dir}"}

    # Create output directory with agency name
    agency_name = os.path.basename(agency_dir)
    agency_output_dir = os.path.join(output_dir, agency_name)
    os.makedirs(agency_output_dir, exist_ok=True)

    # Evaluate each topic
    topic_results = []
    for topic_name, topic_file_path in topic_files:
        logger.info(f"Evaluating topic: {topic_name}")
        result = evaluate_topic_file(
            evaluator,
            topic_name,
            topic_file_path,
            agency_dir,
            conversation_base_dir,
            output_dir,
            pattern,
        )
        topic_results.append(result)

    # Calculate agency summary
    agency_summary = {
        "agency_name": agency_name,
        "topics_evaluated": len(topic_results),
        "topic_results": {
            r.get("topic_name", ""): r for r in topic_results if "topic_name" in r
        },
    }

    # Calculate overall agency scores
    scores_by_criteria = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        all_scores = []
        for topic in topic_results:
            if "scores" in topic and criteria in topic["scores"]:
                avg_score = topic["scores"][criteria]["average"]
                all_scores.append(avg_score)

        if all_scores:
            scores_by_criteria[criteria] = sum(all_scores) / len(all_scores)

    agency_summary["average_scores"] = scores_by_criteria

    # Calculate overall aggregate score
    aggregate_scores = [
        topic.get("average_aggregate_score")
        for topic in topic_results
        if "average_aggregate_score" in topic
    ]
    if aggregate_scores:
        agency_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )

    # Save agency summary
    with open(
        os.path.join(agency_output_dir, "agency_fit_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(path_to_str(agency_summary), f, ensure_ascii=False, indent=2)

    # Generate agency report
    _generate_agency_fit_report(
        agency_summary, os.path.join(agency_output_dir, "agency_fit_report.md")
    )

    return agency_summary


def evaluate_conversation_file(
    evaluator: TopicAgencyFitEvaluator,
    conversation_file: str,
    topic_dir: str,
    agency_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate a single conversation file for topic and agency fit.

    Args:
        evaluator: The evaluator instance
        conversation_file: Path to the conversation file
        topic_dir: Path to the topic directory
        agency_dir: Path to the agency directory
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation results
    """
    # Read conversation
    try:
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation = f.read()
    except Exception as e:
        logger.error(f"Error reading conversation file {conversation_file}: {e}")
        return {"error": f"Error reading file: {e}"}

    # Get topic and agency names
    topic_name = os.path.basename(topic_dir)
    agency_name = os.path.basename(agency_dir)

    # Find topic description file
    topic_description_file = find_topic_description_file(topic_dir)
    if topic_description_file:
        topic_description = evaluator.read_topic_description(topic_description_file)
    else:
        topic_description = f"This topic is about {topic_name}."

    # Get agency description
    agency_description = evaluator.read_agency_description(agency_name)

    # Evaluate the conversation
    logger.info(f"Evaluating conversation: {os.path.basename(conversation_file)}")
    result = evaluator.evaluate_conversation_topic_agency_fit(
        conversation, topic_name, topic_description, agency_name, agency_description
    )

    # Add file information
    result["file_name"] = os.path.basename(conversation_file)
    result["file_path"] = conversation_file

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save evaluation result
    file_stem = os.path.splitext(os.path.basename(conversation_file))[0]
    output_file = os.path.join(output_dir, f"{file_stem}_topic_agency_fit.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(path_to_str(result), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation result to {output_file}")

    return result


def evaluate_topic_directory(
    evaluator: TopicAgencyFitEvaluator,
    topic_dir: str,
    agency_dir: str,
    output_dir: str,
    pattern: str = "conversation_*.txt",
) -> Dict[str, Any]:
    """
    Evaluate all conversations in a topic directory for topic and agency fit.


    Returns:
        Dictionary with evaluation summary
    """
    # Find conversation files
    conversation_files = sorted(list(Path(topic_dir).glob(pattern)))

    if not conversation_files:
        logger.warning(
            f"No conversation files matching pattern '{pattern}' found in {topic_dir}"
        )
        return {"error": f"No conversation files found in {topic_dir}"}

    # Create output directory with topic name
    topic_name = os.path.basename(topic_dir)
    topic_output_dir = os.path.join(output_dir, topic_name)
    os.makedirs(topic_output_dir, exist_ok=True)

    # Evaluate each conversation
    all_results = []
    for file_path in conversation_files:
        try:
            result = evaluate_conversation_file(
                evaluator, str(file_path), topic_dir, agency_dir, topic_output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")

    # Calculate summary statistics
    topic_summary = {
        "topic_name": topic_name,
        "agency_name": os.path.basename(agency_dir),
        "total_conversations": len(all_results),
        "conversation_files": [r.get("file_name") for r in all_results],
    }

    # Calculate average scores
    scores = {}
    for criteria in ["topic_fit", "agency_fit", "classification_confidence"]:
        valid_scores = [
            r.get("numerical_scores", {}).get(criteria)
            for r in all_results
            if r.get("numerical_scores", {}).get(criteria) is not None
        ]
        if valid_scores:
            scores[criteria] = {
                "average": sum(valid_scores) / len(valid_scores),
                "min": min(valid_scores),
                "max": max(valid_scores),
            }

    # Calculate aggregate score
    aggregate_scores = [
        r.get("aggregate_score")
        for r in all_results
        if r.get("aggregate_score") is not None
    ]
    if aggregate_scores:
        topic_summary["average_aggregate_score"] = sum(aggregate_scores) / len(
            aggregate_scores
        )
        topic_summary["min_aggregate_score"] = min(aggregate_scores)
        topic_summary["max_aggregate_score"] = max(aggregate_scores)

    topic_summary["scores"] = scores

    # Save topic summary
    with open(
        os.path.join(topic_output_dir, "topic_fit_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(path_to_str(topic_summary), f, ensure_ascii=False, indent=2)

    # Generate a readable report
    _generate_topic_fit_report(
        topic_summary,
        all_results,
        os.path.join(topic_output_dir, "topic_fit_report.md"),
    )

    return topic_summary


def _generate_topic_fit_report(
    topic_summary: Dict[str, Any], all_results: List[Dict[str, Any]], output_file: str
):
    """
    Generate a readable Markdown report for topic fit evaluation.


    """
    with open(output_file, "w", encoding="utf-8") as f:
        topic_name = topic_summary.get("topic_name", "Unknown Topic")
        agency_name = topic_summary.get("agency_name", "Unknown Agency")

        f.write(f"# Topic and Agency Fit Evaluation: {topic_name}\n\n")
        f.write(f"**Agency**: {agency_name}\n\n")

        # Summary statistics
        f.write("## Summary\n\n")
        f.write(
            f"- **Conversations Evaluated**: {topic_summary.get('total_conversations', 0)}\n"
        )

        # Average scores
        if "scores" in topic_summary:
            f.write("### Average Scores\n\n")
            f.write("| Criteria | Average | Min | Max |\n")
            f.write("|----------|---------|-----|-----|\n")

            for criteria, stats in topic_summary["scores"].items():
                criteria_name = criteria.replace("_", " ").title()
                f.write(
                    f"| {criteria_name} | {stats['average']:.2f} | {stats['min']:.1f} | {stats['max']:.1f} |\n"
                )

            f.write("\n")

        # Overall score
        if "average_aggregate_score" in topic_summary:
            score = topic_summary["average_aggregate_score"]
            f.write(f"**Overall Topic-Agency Fit Score**: {score:.2f}/5.0\n\n")

            # Interpretation
            if score >= 4.5:
                interpretation = "Excellent fit for agency and topic"
            elif score >= 4.0:
                interpretation = "Good fit for agency and topic"
            elif score >= 3.0:
                interpretation = "Acceptable fit, with some room for improvement"
            elif score >= 2.0:
                interpretation = "Poor fit, needs significant improvement"
            else:
                interpretation = "Very poor fit, major revisions needed"

            f.write(f"**Interpretation**: {interpretation}\n\n")

        f.write("## Individual Conversation Evaluations\n\n")

        sorted_results = sorted(
            all_results,
            key=lambda r: (
                r.get("aggregate_score", 0)
                if r.get("aggregate_score") is not None
                else 0
            ),
            reverse=True,
        )

        for result in sorted_results:
            file_name = result.get("file_name", "Unknown")

            f.write(f"### {file_name}\n\n")

            aggregate_score = result.get("aggregate_score")
            if aggregate_score is not None:
                f.write(f"**Overall Score**: {aggregate_score:.2f}/5.0\n\n")

            if "numerical_scores" in result:
                f.write("#### Scores\n\n")
                for criteria, score in result["numerical_scores"].items():
                    criteria_name = criteria.replace("_", " ").title()
                    f.write(f"- **{criteria_name}**: {score:.1f}/5.0\n")
                f.write("\n")

            if (
                "detailed_scores" in result
                and "key_indicators" in result["detailed_scores"]
            ):
                key_indicators = result["detailed_scores"]["key_indicators"].get(
                    "response", ""
                )
                if key_indicators:
                    f.write("#### Key Topic/Agency Indicators\n\n")
                    f.write(f"{key_indicators}\n\n")

            if (
                "detailed_scores" in result
                and "potential_confusion" in result["detailed_scores"]
            ):
                confusion = result["detailed_scores"]["potential_confusion"].get(
                    "response", ""
                )
                if confusion:
                    f.write("#### Potential Classification Confusion\n\n")
                    f.write(f"{confusion}\n\n")

            f.write("---\n\n")


def _generate_agency_fit_report(agency_summary: Dict[str, Any], output_file: str):
    """
    Generate a readable Markdown report for agency fit evaluation.


    """
    with open(output_file, "w", encoding="utf-8") as f:
        agency_name = agency_summary.get("agency_name", "Unknown Agency")

        f.write(f"# Agency Fit Evaluation: {agency_name}\n\n")

        f.write("## Summary\n\n")
        f.write(
            f"- **Topics Evaluated**: {agency_summary.get('topics_evaluated', 0)}\n"
        )

        if "average_scores" in agency_summary:
            f.write("### Agency-wide Average Scores\n\n")
            f.write("| Criteria | Score |\n")
            f.write("|----------|-------|\n")

            for criteria, score in agency_summary["average_scores"].items():
                criteria_name = criteria.replace("_", " ").title()
                f.write(f"| {criteria_name} | {score:.2f} |\n")

            f.write("\n")

        if "average_aggregate_score" in agency_summary:
            score = agency_summary["average_aggregate_score"]
            f.write(f"**Overall Agency Fit Score**: {score:.2f}/5.0\n\n")

            if score >= 4.5:
                interpretation = (
                    "Excellent - Conversations very clearly represent this agency"
                )
            elif score >= 4.0:
                interpretation = "Good - Conversations represent this agency well"
            elif score >= 3.0:
                interpretation = (
                    "Acceptable - Conversations adequately represent this agency"
                )
            elif score >= 2.0:
                interpretation = (
                    "Poor - Many conversations don't clearly represent this agency"
                )
            else:
                interpretation = "Very poor - Major issues with agency representation"

            f.write(f"**Interpretation**: {interpretation}\n\n")

        f.write("## Topic Scores\n\n")

        topic_scores = []
        for topic_name, topic_data in agency_summary.get("topic_results", {}).items():
            if "average_aggregate_score" in topic_data:
                topic_scores.append(
                    {
                        "name": topic_name,
                        "score": topic_data["average_aggregate_score"],
                        "conversations": topic_data.get("total_conversations", 0),
                    }
                )

        topic_scores.sort(key=lambda x: x["score"], reverse=True)

        if topic_scores:
            f.write("| Topic | Fit Score | Conversations |\n")
            f.write("|-------|-----------|---------------|\n")

            for topic in topic_scores:
                f.write(
                    f"| {topic['name']} | {topic['score']:.2f} | {topic['conversations']} |\n"
                )

            f.write("\n")

        f.write("## Recommendations\n\n")

        if topic_scores:
            problem_topics = [t for t in topic_scores if t["score"] < 3.5]
            if problem_topics:
                f.write("### Topics Needing Improvement\n\n")
                f.write(
                    "The following topics may need attention to better align with this agency:\n\n"
                )
                for topic in problem_topics:
                    f.write(f"- **{topic['name']}** ({topic['score']:.2f}/5.0)\n")
                f.write("\n")

            # Identify exemplary topics
            good_topics = [t for t in topic_scores if t["score"] >= 4.5]
            if good_topics:
                f.write("### Exemplary Topics\n\n")
                f.write(
                    "These topics demonstrate excellent alignment with this agency and can serve as examples:\n\n"
                )
                for topic in good_topics:
                    f.write(f"- **{topic['name']}** ({topic['score']:.2f}/5.0)\n")
                f.write("\n")

        f.write("### General Recommendations\n\n")
        f.write(
            "- Review conversations in lower-scoring topics to ensure they clearly relate to this agency\n"
        )
        f.write(
            "- Consider adding more agency-specific terminology and context to ambiguous conversations\n"
        )
        f.write(
            "- Ensure that each conversation contains clear indicators of the responsible agency\n"
        )
        f.write(
            "- Use the key indicators identified in individual reports to strengthen agency associations\n"
        )


def main():

    # Create evaluator
    evaluator = TopicAgencyFitEvaluator(
        model_name=DEFAULT_MODEL_NAME,
        use_4bit=True,
        cpu_only=False,
        temperature=0.1,
    )
    input_dir = "../data/output_ID.ee/"
    conversation_dir = "data/ID.ee/"
    # Evaluate the agency directory
    agency_dir = input_dir
    output_dir = "output"

    # Perform evaluation
    evaluate_agency_directory_flat(evaluator, agency_dir, conversation_dir, output_dir)


if __name__ == "__main__":
    main()
