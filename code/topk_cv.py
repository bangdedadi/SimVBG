import openai
import httpx
import json
import csv
import concurrent.futures
import re
from tqdm import tqdm
import random
import numpy as np
import time
import os
import argparse
from collections import defaultdict

# Initialize client
def init_client(model_name):
    if model_name in ["gpt-3.5-turbo", "text-davinci-003", "gpt-4o-mini", "claude-3-haiku-20240307", "deepseek-v3-250324"]:
        return openai.OpenAI(
            base_url="your_base_url_here",
            api_key="your_api_key_here",
            http_client=httpx.Client(base_url="your_base_url_here", follow_redirects=True),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Embedding service class
class EmbeddingService:
    def __init__(self, api_key, base_url="your_base_url_here"):
        self.client = httpx.Client(
            base_url=base_url,
            follow_redirects=True,
            timeout=30.0
        )
        self.api_key = api_key
        self.model = "text-embedding-ada-002"

    def get_embedding(self, text):
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self.client.post(
                "/embeddings",
                headers=headers,
                json={"model": self.model, "input": text.strip()},
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data["data"][0]["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 1536  # Default embedding vector

    def batch_get_embeddings(self, texts, batch_size=20):
        """Batch retrieval of embedding vectors to reduce API calls"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i + batch_size, len(texts))]
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = self.client.post(
                    "/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": [text.strip() for text in batch]},
                )
                response.raise_for_status()
                response_data = response.json()
                batch_embeddings = [data["embedding"] for data in response_data["data"]]
                all_embeddings.extend(batch_embeddings)
                
                # Add delay to avoid API rate limits
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error getting batch embeddings: {e}")
                # If batch processing fails, try processing one by one
                for text in batch:
                    all_embeddings.append(self.get_embedding(text))
                    time.sleep(0.5)
                
        return all_embeddings

# LLM API call function
def call_llm_api(messages, model):
    client = init_client(model)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        # Retry once
        time.sleep(2)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Retry failed: {e}")
            return "Error: Failed to get response from API"

# Helper function: Extract the first number from the response
def extract_first_number(response):
    match = re.search(r"\b\d+\b", response)
    if match:
        return int(match.group())
    else:
        return None

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Generate question text
def generate_question_text_with_options(question_key, questions_data):
    if question_key not in questions_data:
        return f"Question {question_key} not found in data."
    
    question_text = questions_data[question_key].get("question_text", "No question text available.")
    options = questions_data[question_key].get("options", {})

    if options:
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        return f"{question_text}\nOptions:\n{options_text}"
    else:
        return question_text

# Generate natural language profile text list
def generate_profile_text_list(row, profile_qns, nature_options_data):
    profile_info_list = []
    for q in profile_qns:
        if q in row and q in nature_options_data:
            option_value = row[q]
            try:
                option_value_int = int(option_value)  # Try to convert answer to integer
            except ValueError:
                continue
            
            # Get question options and template
            options = nature_options_data[q].get('options', {})
            template = nature_options_data[q].get('template', None)
            question_text = nature_options_data[q]['question_text']

            # Handle multiple choice questions
            if options:
                if option_value in options:
                    profile_info_list.append(f"{options[option_value]}")  # Use natural language option
                else:
                    print(f"Warning: Option value {option_value} not found in options for question {q}")
            
            # Handle fill-in-the-blank questions
            elif template:
                # Use template to generate natural language
                profile_info_list.append(template.format(value=option_value))
            
            # Default handling method
            else:
                # If no template, use default question+answer format
                profile_info_list.append(f"{question_text}: {option_value}")
    return profile_info_list

# Generate profile text list with original questions
def generate_profile_text_list_withquestions(row, profile_qns, questions):
    profile_info_list = []
    for q in profile_qns:
        if q in row and q in questions:
            option_value = row[q]
            try:
                option_value_int = int(option_value)  # Try to convert answer to integer
            except ValueError:
                continue
            
            # Get question text and options
            question_text = questions[q]['question_text']
            options = questions[q].get('options', {})

            # Handle multiple choice questions
            if options and option_value in options:
                options_text = "\n".join([v for k, v in options.items()])
                profile_info_list.append(f"{question_text}\n{options_text}\n{option_value}")
            # Handle fill-in-the-blank questions
            elif not options:
                profile_info_list.append(f"{question_text}\n{option_value}")
            else:
                print(f"Warning: Option value {option_value} not found in options for question {q}")
                
    return profile_info_list

# Read CSV data
def load_data(csv_file):
    data = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader, start=1):
            row['Row_Number'] = i
            data.append(row)
    return data

# Load JSON file
def load_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load split data from JSON file
def load_split_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract split content - compatible with new and old formats
    if "fold" in data:  # New cross-validation format
        questions = data.get("test_set", [])
        profile_qns_train = data.get("train_set", [])
        return [], questions, profile_qns_train, []
    else:  # Old format
        seed_information = data.get("seed_information", [])
        questions = data.get("test_set", [])
        profile_qns_train = data.get("train_set", [])
        profile_qns_val = data.get("validation_set", [])
        return seed_information, questions, profile_qns_train, profile_qns_val

# New: Create k-fold cross-validation splits
def create_cross_validation_splits(questions, num_folds=5, random_seed=42):
    """
    Create k-fold cross-validation splits
    
    Parameters:
    - questions: List of all questions
    - num_folds: Number of folds for cross-validation, default is 5
    - random_seed: Random seed for consistent splits
    
    Returns:
    - splits: List containing num_folds splits, each split is a dictionary with train and test sets
    """
    # Set random seed for consistency
    random.seed(random_seed)
    
    # Shuffle the questions list
    shuffled_questions = questions.copy()
    random.shuffle(shuffled_questions)
    
    # Calculate size of each fold
    fold_size = len(shuffled_questions) // num_folds
    
    # Create num_folds splits
    splits = []
    for i in range(num_folds):
        # Calculate start and end indices for current fold
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(shuffled_questions)
        
        # Current fold as test set
        test_set = shuffled_questions[start_idx:end_idx]
        
        # Remaining folds as training set
        train_set = shuffled_questions[:start_idx] + shuffled_questions[end_idx:]
        
        # Create current split
        split = {
            "fold": i + 1,
            "train_set": train_set,
            "test_set": test_set
        }
        
        splits.append(split)
    
    return splits

# New: Save cross-validation splits to JSON files
def save_cross_validation_splits(splits, output_dir):
    """
    Save cross-validation splits to JSON files
    
    Parameters:
    - splits: List of cross-validation splits
    - output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split to a separate file
    for split in splits:
        fold = split["fold"]
        filename = f"SPLIT_FOLD_{fold}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(splits)} cross-validation splits to {output_dir} directory")

# Phase 1: Generate embedding vectors
def generate_embeddings(
    sampled_data, 
    profile_qns, 
    question_qns, 
    nature_options_data, 
    questions_data, 
    embedding_service, 
    output_file,
    use_natural_options=True
):
    all_user_data = []
    
    # Preprocess all texts for batch embedding
    all_profile_texts = []
    all_question_texts = []
    user_profile_map = {}  # Mapping from user ID to profile text
    
    # 1. Collect all texts for embedding
    print("Collecting texts for embedding...")
    for row in tqdm(sampled_data):
        row_number = row['Row_Number']
        
        # Generate profile text
        if use_natural_options:
            profile_text_list = generate_profile_text_list(row, profile_qns, nature_options_data)
        else:
            profile_text_list = generate_profile_text_list_withquestions(row, profile_qns, questions_data)
        
        # Store profile text for user
        user_profile_map[row_number] = profile_text_list
        
        # Add to text list for embedding
        all_profile_texts.extend(profile_text_list)
    
    # Generate question text
    for q in question_qns:
        if q in questions_data:
            question_text = generate_question_text_with_options(q, questions_data)
            all_question_texts.append((q, question_text))
    
    # 2. Batch retrieve embedding vectors
    print("Generating profile embeddings...")
    profile_embeddings = embedding_service.batch_get_embeddings([text for text in all_profile_texts])
    
    print("Generating question embeddings...")
    question_embeddings = embedding_service.batch_get_embeddings([text for _, text in all_question_texts])
    
    # 3. Build user data structure
    profile_text_index = 0
    for row in tqdm(sampled_data, desc="Building user data structure"):
        row_number = row['Row_Number']
        profile_text_list = user_profile_map[row_number]
        
        # Get profile embeddings for this user
        profile_embs = profile_embeddings[profile_text_index:profile_text_index+len(profile_text_list)]
        profile_text_index += len(profile_text_list)
        
        # Build user data
        user_data = {
            "Row_Number": row_number,
            "profile_texts": profile_text_list,
            "profile_embeddings": profile_embs,
            "questions": [
                {
                    "question": q, 
                    "embedding": question_embeddings[i],
                    "question_text": all_question_texts[i][1]
                } 
                for i, (q, _) in enumerate(all_question_texts)
            ],
            "original_answers": {q: int(row[q]) if q in row and row[q].isdigit() else None for q in question_qns}
        }
        all_user_data.append(user_data)
    
    # 4. Save embedding data
    print(f"Saving embeddings to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_user_data, f, indent=4)
    
    return all_user_data

# Load previously generated embeddings
def load_embeddings(embedding_file):
    print(f"Loading embeddings from {embedding_file}...")
    try:
        with open(embedding_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading embeddings: {e}")
        return None

# Phase 2: Answer questions based on TopK retrieval
def answer_questions_with_topk(
    user_data, 
    questions_data, 
    output_file, 
    topk=3, 
    sim_threshold=0.3, 
    answer_model="gpt-3.5-turbo"
):
    results = defaultdict(list)
    
    # Process user questions in parallel
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        for question in user["questions"]:
            all_tasks.append((user, question, row_number, question["question"]))
    
    with tqdm(total=len(all_tasks), desc="Processing questions") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            futures = {}
            for user, question, row_number, question_id in all_tasks:
                future = executor.submit(
                    process_question_with_topk,
                    user, question, topk, sim_threshold, answer_model, questions_data
                )
                futures[future] = (row_number, question_id)
                time.sleep(0.05)  # Prevent API request concentration
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    answer, response, prompt, retrieved_items = future.result()
                    results[row_number].append({
                        "question_id": question_id,
                        "answer": answer,
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id),
                        "response": response,
                        "prompt": prompt,
                        "retrieved_items": retrieved_items,
                        "retrieved_count": len(retrieved_items)
                    })
                except Exception as e:
                    print(f"Error processing Row {row_number}, Question {question_id}: {e}")
                    results[row_number].append({
                        "question_id": question_id,
                        "answer": None,
                        "original_answer": None,
                        "response": f"Error: {str(e)}",
                        "prompt": None,
                        "retrieved_items": [],
                        "retrieved_count": 0
                    })
                pbar.update(1)
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(dict(results), f, indent=4)
    
    print(f"Results saved to {output_file}")
    return results

# Process individual question with TopK retrieval and answering
def process_question_with_topk(user, question, topk, sim_threshold, answer_model, questions_data):
    # Get question embedding and profile embeddings
    question_emb = question["embedding"]
    profile_texts = user["profile_texts"]
    profile_embs = user["profile_embeddings"]
    
    # Calculate similarities and filter
    similarities = [
        (text, cosine_similarity(np.array(emb), np.array(question_emb)))
        for text, emb in zip(profile_texts, profile_embs)
    ]
    
    # Filter based on threshold
    filtered_profiles = [(text, sim) for text, sim in similarities if sim >= sim_threshold]
    
    # Select TopK
    topk_profiles = sorted(filtered_profiles, key=lambda x: x[1], reverse=True)[:topk]
    
    # If no results meet the threshold, select the highest similarity ones
    if not topk_profiles and similarities:
        topk_profiles = sorted(similarities, key=lambda x: x[1], reverse=True)[:topk]
    
    # Construct retrieval result text
    profile_text = "\n".join([f"- {text} [similarity: {sim:.4f}]" for text, sim in topk_profiles])
    
    # Construct prompt
    question_text_with_options = question.get("question_text", 
        generate_question_text_with_options(question["question"], questions_data))
    
    prompt = (
        f"Question: {question_text_with_options}\n\n"
        f"Relevant user information:\n{profile_text}\n\n"
        f"Based ONLY on the relevant user information provided above, answer the question. "
        f"Consider both the question context and the user's background from the provided relevant information. "
        f"Aim for a balanced perspective that respects accuracy while reflecting the user's viewpoint.\n"
        f"Answer format: 'option you selected'"
    )
    
    # Call LLM API
    messages = [{"role": "user", "content": prompt}]
    response = call_llm_api(messages, answer_model)
    
    # Extract answer
    answer = extract_first_number(response)
    
    # Return results, including retrieved items
    retrieved_items = [{"text": text, "similarity": float(sim)} for text, sim in topk_profiles]
    
    return answer, response, prompt, retrieved_items

# Evaluate TopK results
def evaluate_topk_results(results_file, output_file=None):
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Calculate statistics
    total_questions = 0
    correct_answers = 0
    retrieved_counts = []
    no_retrieval = 0
    
    for user_id, questions in results.items():
        for q in questions:
            total_questions += 1
            
            if q["answer"] == q["original_answer"]:
                correct_answers += 1
                
            retrieved_counts.append(q["retrieved_count"])
            
            if q["retrieved_count"] == 0:
                no_retrieval += 1
    
    # Calculate metrics
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    avg_retrieved = sum(retrieved_counts) / len(retrieved_counts) if retrieved_counts else 0
    
    # Generate evaluation report
    report = {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "average_retrieved_items": avg_retrieved,
        "questions_with_no_retrieval": no_retrieval,
        "percentage_no_retrieval": no_retrieval / total_questions if total_questions > 0 else 0
    }
    
    # Print report
    print("\n--- TopK Evaluation Report ---")
    print(f"Total questions: {report['total_questions']}")
    print(f"Correct answers: {report['correct_answers']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Average retrieved items: {report['average_retrieved_items']:.2f}")
    print(f"Questions with no retrieval: {report['questions_with_no_retrieval']} ({report['percentage_no_retrieval']:.2%})")
    
    # Save report
    if output_file:
        with open(output_file, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Evaluation report saved to {output_file}")
    
    return report

# New: Evaluate all cross-validation fold results and summarize
def evaluate_cross_validation_results(result_files, output_file=None):
    """
    Evaluate all cross-validation fold results and calculate mean and standard deviation
    
    Parameters:
    - result_files: List of result file paths for all cross-validation folds
    - output_file: Output file path to save summary evaluation results
    
    Returns:
    - Summary evaluation report
    """
    # Store evaluation results for each fold
    fold_results = []
    
    # Evaluate results for each fold
    for i, result_file in enumerate(result_files, 1):
        print(f"Evaluating fold {i} results from {result_file}...")
        fold_report = evaluate_topk_results(result_file, None)  # Don't save individual fold evaluation reports
        fold_report["fold"] = i
        fold_results.append(fold_report)
    
    # Calculate mean and standard deviation
    avg_accuracy = np.mean([report["accuracy"] for report in fold_results])
    std_accuracy = np.std([report["accuracy"] for report in fold_results])
    
    avg_retrieved = np.mean([report["average_retrieved_items"] for report in fold_results])
    std_retrieved = np.std([report["average_retrieved_items"] for report in fold_results])
    
    avg_no_retrieval_pct = np.mean([report["percentage_no_retrieval"] for report in fold_results])
    std_no_retrieval_pct = np.std([report["percentage_no_retrieval"] for report in fold_results])
    
    # Generate summary report
    summary_report = {
        "fold_results": fold_results,
        "average_accuracy": avg_accuracy,
        "std_accuracy": std_accuracy,
        "average_retrieved_items": avg_retrieved,
        "std_retrieved_items": std_retrieved,
        "average_percentage_no_retrieval": avg_no_retrieval_pct,
        "std_percentage_no_retrieval": std_no_retrieval_pct
    }
    
    # Print summary results
    print("\n--- Cross-Validation Summary ---")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Retrieved Items: {avg_retrieved:.2f} ± {std_retrieved:.2f}")
    print(f"Average Percentage No Retrieval: {avg_no_retrieval_pct:.2%} ± {std_no_retrieval_pct:.2%}")
    
    # Save summary report
    if output_file:
        with open(output_file, "w") as f:
            json.dump(summary_report, f, indent=4)
        print(f"Cross-validation summary saved to {output_file}")
    
    return summary_report

# Modified main function to support k-fold cross-validation
def main():
    # Set default parameters
    default_config = {
        "data_file": '../WVS_dataset/WVS_Cross-National_Wave_7_csv_v6_0.csv',
        "questions_file": '../WVS_dataset/questions.json',
        "nature_options_file": '../WVS_dataset/nature_options.json',
        "split_dir": '../data_split/CROSS_VAL_5_FOLDS_SEED=42',
        "embedding_dir": '../embeddings/cv5_top3_sample=100_seed=42',
        "results_dir": '../results/cv5_top3_sample=100_model=gpt-3.5-turbo',
        "sample_size": 100,
        "topk": 3,
        "threshold": 0.3,
        "answer_model": 'gpt-3.5-turbo',
        "api_key": 'your_api_key_here',
        "random_seed": 42,
        "use_natural_options": False,
        "skip_embedding": True,
        "evaluate": True,
        "num_folds": 5,
        "create_splits": True,
        "all_questions": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q65", "Q66", "Q67", "Q68", "Q69", "Q70", "Q71", "Q72", "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87", "Q88", "Q89", "Q90", "Q91", "Q92", "Q93", "Q94", "Q95", "Q96", "Q97", "Q98", "Q99", "Q100", "Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q114", "Q115", "Q116", "Q117", "Q118", "Q119", "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129", "Q130", "Q131", "Q132", "Q133", "Q134", "Q135", "Q136", "Q137", "Q138", "Q139", "Q140", "Q141", "Q142", "Q143", "Q144", "Q145", "Q146", "Q147", "Q148", "Q149", "Q150", "Q151", "Q152", "Q153", "Q154", "Q155", "Q156", "Q157", "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q164", "Q165", "Q166", "Q167", "Q168", "Q169", "Q170", "Q171", "Q172", "Q173", "Q174", "Q175", "Q176", "Q177", "Q178", "Q179", "Q180", "Q181", "Q182", "Q183", "Q184", "Q185", "Q186", "Q187", "Q188", "Q189", "Q190", "Q191", "Q192", "Q193", "Q194", "Q195", "Q196", "Q197", "Q198", "Q199", "Q200", "Q201", "Q202", "Q203", "Q204", "Q205", "Q206", "Q207", "Q208", "Q209", "Q210", "Q211", "Q212", "Q213", "Q214", "Q215", "Q216", "Q217", "Q218", "Q219", "Q220", "Q221", "Q222", "Q223", "Q224", "Q225", "Q226", "Q227", "Q228", "Q229", "Q230", "Q231", "Q232", "Q233", "Q234", "Q235", "Q236", "Q237", "Q238", "Q239", "Q240", "Q241", "Q242", "Q243", "Q244", "Q245", "Q246", "Q247", "Q248", "Q249", "Q250", "Q251", "Q252", "Q253", "Q254", "Q255", "Q256", "Q257", "Q258", "Q259", "Q260", "Q261", "Q262", "Q263", "Q264", "Q265", "Q266", "Q267", "Q268", "Q269", "Q270", "Q271", "Q272", "Q273", "Q274", "Q275", "Q276", "Q277", "Q278", "Q279", "Q280", "Q281", "Q282", "Q283", "Q284", "Q285", "Q286", "Q287", "Q288", "Q289", "Q290"]
    }
    
    # Parse command line arguments, allow overriding defaults
    parser = argparse.ArgumentParser(description="Run TopK RAG baseline with cross-validation")
    parser.add_argument("--data_file", type=str, help="Path to the CSV data file")
    parser.add_argument("--questions_file", type=str, help="Path to the questions JSON file")
    parser.add_argument("--nature_options_file", type=str, help="Path to the nature options JSON file")
    parser.add_argument("--split_dir", type=str, help="Directory for split JSON files")
    parser.add_argument("--embedding_dir", type=str, help="Directory to save/load embeddings")
    parser.add_argument("--results_dir", type=str, help="Directory to save results")
    parser.add_argument("--sample_size", type=int, help="Number of samples to use")
    parser.add_argument("--topk", type=int, help="Number of top profile items to retrieve")
    parser.add_argument("--threshold", type=float, help="Similarity threshold")
    parser.add_argument("--answer_model", type=str, help="Model to use for answering")
    parser.add_argument("--api_key", type=str, help="API key for embedding service")
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--use_natural_options", action="store_true", help="Use natural language options")
    parser.add_argument("--no_natural_options", action="store_false", dest="use_natural_options", help="Don't use natural language options")
    parser.add_argument("--skip_embedding", action="store_true", help="Skip embedding generation and load from file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate results after generation")
    parser.add_argument("--num_folds", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--create_splits", action="store_true", help="Create new cross-validation splits")
    parser.add_argument("--no_create_splits", action="store_false", dest="create_splits", help="Use existing cross-validation splits")
    
    args = parser.parse_args()
    
    # Update configuration, only use parameters explicitly provided in command line
    config = default_config.copy()
    for arg, value in vars(args).items():
        if value is not None:  # Only update parameters provided in command line
            config[arg] = value
    
    # Create required directories
    os.makedirs(config["split_dir"], exist_ok=True)
    os.makedirs(config["embedding_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)
    
    # Set random seed
    random.seed(config["random_seed"])
    
    # Load data
    print(f"Loading data from {config['data_file']}...")
    data = load_data(config["data_file"])
    
    print(f"Loading questions from {config['questions_file']}...")
    questions_data = load_json_file(config["questions_file"])
    
    print(f"Loading nature options from {config['nature_options_file']}...")
    nature_options_data = load_json_file(config["nature_options_file"])
    
    # Random sample data
    sampled_data = random.sample(data, min(config["sample_size"], len(data)))
    
    # Initialize embedding service
    embedding_service = EmbeddingService(api_key=config["api_key"])
    
    # Create cross-validation splits
    if config["create_splits"]:
        print(f"Creating {config['num_folds']}-fold cross-validation splits...")
        cv_splits = create_cross_validation_splits(
            config["all_questions"], 
            num_folds=config["num_folds"], 
            random_seed=config["random_seed"]
        )
        save_cross_validation_splits(cv_splits, config["split_dir"])
    else:
        # Load existing cross-validation splits
        print(f"Using existing cross-validation splits from {config['split_dir']}...")
        cv_splits = []
        for i in range(1, config["num_folds"] + 1):
            split_file = os.path.join(config["split_dir"], f"SPLIT_FOLD_{i}.json")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    cv_splits.append(json.load(f))
            else:
                print(f"Warning: Split file {split_file} not found")
    
    # Store result file paths for each fold
    result_files = []
    
    # Process each cross-validation fold
    for fold_idx, split in enumerate(cv_splits, 1):
        print(f"\n=== Processing Fold {fold_idx}/{config['num_folds']} ===")
        if(fold_idx==1):
            continue
        # Get train and test sets for current fold
        profile_qns = split["train_set"]
        question_qns = split["test_set"]
        
        # Build file paths for current fold
        embedding_file = os.path.join(config["embedding_dir"], f"fold_{fold_idx}_embeddings.json")
        results_file = os.path.join(config["results_dir"], f"fold_{fold_idx}_results.json")
        evaluation_file = os.path.join(config["results_dir"], f"fold_{fold_idx}_evaluation.json")
        
        result_files.append(results_file)
        
        # Generate or load embeddings
        if not config["skip_embedding"]:
            print(f"Generating embeddings for fold {fold_idx}...")
            user_data = generate_embeddings(
                sampled_data, 
                profile_qns, 
                question_qns, 
                nature_options_data, 
                questions_data, 
                embedding_service, 
                embedding_file,
                use_natural_options=config["use_natural_options"]
            )
        else:
            print(f"Loading existing embeddings for fold {fold_idx}...")
            user_data = load_embeddings(embedding_file)
            if user_data is None:
                print(f"Failed to load embeddings for fold {fold_idx}. Generating new embeddings...")
                user_data = generate_embeddings(
                    sampled_data, 
                    profile_qns, 
                    question_qns, 
                    nature_options_data, 
                    questions_data, 
                    embedding_service, 
                    embedding_file,
                    use_natural_options=config["use_natural_options"]
                )
        
        # Answer questions with TopK retrieval
        print(f"Answering questions for fold {fold_idx} with TopK retrieval (k={config['topk']}, threshold={config['threshold']})...")
        results = answer_questions_with_topk(
            user_data, 
            questions_data, 
            results_file, 
            topk=config["topk"], 
            sim_threshold=config["threshold"], 
            answer_model=config["answer_model"]
        )
        
        # Evaluate current fold results
        if config["evaluate"]:
            print(f"Evaluating results for fold {fold_idx}...")
            evaluate_topk_results(results_file, evaluation_file)
    
    # Evaluate summary results for all cross-validation folds
    if config["evaluate"] and len(result_files) > 1:
        summary_file = os.path.join(config["results_dir"], "cross_validation_summary.json")
        evaluate_cross_validation_results(result_files, summary_file)

if __name__ == "__main__":
    main()