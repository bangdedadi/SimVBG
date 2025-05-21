import openai
import httpx
import json
import csv
import concurrent.futures
from collections import defaultdict
import re
from tqdm import tqdm
import random
import numpy as np
from typing import Dict, List
import threading
import time
import os
import math
import copy  # For deep copying data

ALL_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q65", "Q66", "Q67", "Q68", "Q69", "Q70", "Q71", "Q72", "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87", "Q88", "Q89", "Q90", "Q91", "Q92", "Q93", "Q94", "Q95", "Q96", "Q97", "Q98", "Q99", "Q100", "Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q114", "Q115", "Q116", "Q117", "Q118", "Q119", "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129", "Q130", "Q131", "Q132", "Q133", "Q134", "Q135", "Q136", "Q137", "Q138", "Q139", "Q140", "Q141", "Q142", "Q143", "Q144", "Q145", "Q146", "Q147", "Q148", "Q149", "Q150", "Q151", "Q152", "Q153", "Q154", "Q155", "Q156", "Q157", "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q164", "Q165", "Q166", "Q167", "Q168", "Q169", "Q170", "Q171", "Q172", "Q173", "Q174", "Q175", "Q176", "Q177", "Q178", "Q179", "Q180", "Q181", "Q182", "Q183", "Q184", "Q185", "Q186", "Q187", "Q188", "Q189", "Q190", "Q191", "Q192", "Q193", "Q194", "Q195", "Q196", "Q197", "Q198", "Q199", "Q200", "Q201", "Q202", "Q203", "Q204", "Q205", "Q206", "Q207", "Q208", "Q209", "Q210", "Q211", "Q212", "Q213", "Q214", "Q215", "Q216", "Q217", "Q218", "Q219", "Q220", "Q221", "Q222", "Q223", "Q224", "Q225", "Q226", "Q227", "Q228", "Q229", "Q230", "Q231", "Q232", "Q233", "Q234", "Q235", "Q236", "Q237", "Q238", "Q239", "Q240", "Q241", "Q242", "Q243", "Q244", "Q245", "Q246", "Q247", "Q248", "Q249", "Q250", "Q251", "Q252", "Q253", "Q254", "Q255", "Q256", "Q257", "Q258", "Q259", "Q260", "Q261", "Q262", "Q263", "Q264", "Q265", "Q266", "Q267", "Q268", "Q269", "Q270", "Q271", "Q272", "Q273", "Q274", "Q275", "Q276", "Q277", "Q278", "Q279", "Q280", "Q281", "Q282", "Q283", "Q284", "Q285", "Q286", "Q287", "Q288", "Q289", "Q290"]


def init_client(model_name):
    if model_name in ["gpt-3.5-turbo", "text-davinci-003", "gpt-4o-mini", "claude-3-haiku-20240307"]:
        return openai.OpenAI(
            base_url="your_base_url_here",
            api_key="your_api_key_here",
            http_client=httpx.Client(base_url="your_base_url_here", follow_redirects=True),
        )
    elif model_name in ["qwen-2.5-32b","qwen-2.5-7b","llama-3.1-8b"]:
        return openai.OpenAI(
            base_url="http://0.0.0.0:20022/v1",
            api_key="token-abc123",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def submit_with_delay(executor, func, *args, delay=0.1, **kwargs):
    future = executor.submit(func, *args, **kwargs)
    time.sleep(delay)  # Delay submitting the next task
    return future

def call_llm_api(messages, model):
    client = init_client(model)
    if model == "qwen-2.5-32b":
        model_here= "your_model_here"
    elif model == "qwen-2.5-7b":
        model_here= "your_model_here"
    elif model == "llama-3.1-8b":
        model_here= "your_model_here"
    else:
        model_here=model
    completion = client.chat.completions.create(
        model=model_here,
        messages=messages,
        temperature=0.0
    )
    return completion.choices[0].message.content

def extract_answer_and_analysis(response):
    """
    Extract answer and analysis from LLM response
    Format is 'Answer: [number]\nAnalysis: [analysis text]'
    """
    answer_match = re.search(r"Answer:\s*(\d+)", response)
    answer = int(answer_match.group(1)) if answer_match else None
    
    analysis_match = re.search(r"Analysis:(.*?)(?=Answer:|$)", response, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    
    # If Analysis tag can't be found but there's other text, use all content except the Answer line as analysis
    if not analysis and answer is not None:
        parts = response.split('\n', 1)
        if len(parts) > 1:
            analysis = parts[1].strip()
    
    return answer, analysis

def extract_first_number(response):
    match = re.search(r"\b\d+\b", response)
    if match:
        return int(match.group())
    else:
        return None

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

def format_options(options):
    if not options:
        return "No options available"
    return "\n".join([f"{key}: {value}" for key, value in options.items()])

def generate_full_profile_text(profile_texts):
    return "\n".join(profile_texts)

def combine_story_and_profile(story, profile_text):
    """
    Combine the generated story and original profile information in the specified format
    """
    combined_text = f"brief story:\n{story}\n\noriginal profile:\n{profile_text}"
    return combined_text

def generate_story_with_llm(profile_text, model):
    """
    Generate user story
    """
    # Read story prompt template from file or create default template
    try:
        # Create directory (if it doesn't exist)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Try to load story prompt template
        with open("Q_prompts/storymodule.txt", "r", encoding="utf-8") as f:
            story_template = f.read()
            
    except FileNotFoundError:
        # Create default template
        print("story1.txt not found, using default template.")
        default_template = (
            "Please generate a coherent and engaging story about a person with the following profile information:\n\n"
            "{profile_text}\n\n"
            "Create a narrative that captures this person's background, values, opinions, and life experiences. "
            "Include their thought processes, beliefs, emotional reactions, relationships, behavioral patterns, "
            "and how they typically act in various situations. "
            "The story should be in third person and between 400-600 words."
        )
        
        # Create directory (if it doesn't exist)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Save default template
        with open("Q_prompts/story.txt", "w", encoding="utf-8") as f:
            f.write(default_template)
        
        story_template = default_template

    # Replace placeholders
    prompt_content = story_template.format(profile_text=profile_text)
    messages = [{"role": "user", "content": prompt_content}]
    return call_llm_api(messages, model)

def get_perspective_profile(profile_qns, row, nature_options_data, questions_data, perspective, use_nature_options=True):
    """
    Filter relevant questions based on perspective and generate corresponding profile text
    """
    # Get the question set for the corresponding perspective
    if perspective == "cognitive":
        perspective_questions = ALL_QUESTIONS
    elif perspective == "affective":
        perspective_questions = ALL_QUESTIONS
    elif perspective == "behavioral":
        perspective_questions = ALL_QUESTIONS
    else:
        raise ValueError(f"Invalid perspective: {perspective}")
    
    # Get intersection to find profile questions for this perspective
    perspective_profile_qns = list(set(profile_qns).intersection(set(perspective_questions)))
    
    # Generate profile text for this perspective
    if use_nature_options:
        return generate_profile_text_list(row, perspective_profile_qns, nature_options_data)
    else:
        return generate_profile_text_list_withquestions(row, perspective_profile_qns, questions_data)

def generate_profile_text_list(row, profile_qns, nature_options_data):
    profile_info_list = []
    for q in profile_qns:
        if q in row and q in nature_options_data:
            option_value = row[q]
            try:
                option_value_int = int(option_value)  # Try to convert answer to integer
            except ValueError:
                print(f"Invalid value: {q} = {option_value}")
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
            
            # Handle fill-in questions
            elif template:
                # Use template to generate natural language
                profile_info_list.append(template.format(value=option_value))
            
            # Default handling method
            else:
                # If no template, use default question+answer format
                profile_info_list.append(f"{question_text}: {option_value}")
    return profile_info_list

def generate_profile_text_list_withquestions(row, profile_qns, questions):
    profile_info_list = []
    for q in profile_qns:
        if q in row and q in questions:
            option_value = row[q]
            try:
                option_value_int = int(option_value)  # Try to convert answer to integer
            except ValueError:
                print(f"Invalid value: {q} = {option_value}")
                continue
            
            # Get question text and options
            question_text = questions[q]['question_text']
            options = questions[q].get('options', {})

            # Handle multiple choice questions
            if options and option_value in options:
                # Only include option text, not option values
                options_text = "\n".join([v for k, v in options.items()])
                # Add question, all option texts (without values) and selected option value
                profile_info_list.append(f"{question_text}\n{options_text}\n{option_value}")
            # Handle fill-in questions
            elif not options:
                profile_info_list.append(f"{question_text}\n{option_value}")
            else:
                print(f"Warning: Option value {option_value} not found in options for question {q}")
                
    return profile_info_list

def process_perspective_question_with_analysis(user, question, perspective, story, questions_data, answer_model):
    """
    Process a single perspective question and generate answer and analysis
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)

    # Try to load perspective-specific prompt template
    try:
        # Create directory (if it doesn't exist)
        os.makedirs("Q_prompts", exist_ok=True)
        
        prompt_file_path = f"Q_prompts/prompt_{perspective}.txt"
    
        # Read prompt template
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        # Replace placeholders
        prompt = prompt_template.format(
            profile_text=story,
            question_text_with_options=question_text_with_options
        )
        
    except Exception as e:
        print(f"Error loading prompt template for {perspective} perspective: {e}")
        # Fall back to basic prompt
        prompt = (
            f"Please simulate the role of a user with the following user profile and answer the question as if you were this person.\n\n"
            f"User Profile:\n{story}\n\n"
            f"Please provide answers to the question below in the format 'Answer: [option number]', then on a new line provide your analysis of why this user would choose this option.\n\n"
            f"Question:\n{question_text_with_options}"
        )
    
    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_llm_api(messages, answer_model)
        answer, analysis = extract_answer_and_analysis(response)
        return answer, analysis, response, prompt
    except Exception as e:
        print(f"Error processing {perspective} question for User {row_number}, Question {question_id}: {e}")
        return None, f"Error: {str(e)}", f"Error: {str(e)}", prompt

def coordinator_decision(question_id, cognitive_data, affective_data, behavioral_data, questions_data, coordinator_model):
    """
    When answers from three perspectives are inconsistent, use coordinator to make final decision
    """
    # Construct coordinator prompt
    question_text = questions_data[question_id].get("question_text", "No question text available.")
    options = questions_data[question_id].get("options", {})
    options_text = format_options(options)
    
    coordinator_prompt = f"""
You are a coordinator in a user simulation system, and you need to synthesize analyses from three different perspectives to make a final decision.

Question: {question_text}
Options: {options_text}

Cognitive perspective answer: {cognitive_data['answer']}
Cognitive perspective analysis: {cognitive_data['analysis']}

Emotional perspective answer: {affective_data['answer']}
Emotional perspective analysis: {affective_data['analysis']}

Behavioral perspective answer: {behavioral_data['answer']}
Behavioral perspective analysis: {behavioral_data['analysis']}

Consider:
• How their thoughts, feelings, and behavioral tendencies might interact in this situation
• Which aspects of their psychology seem most influential here
• Where their different perspectives align or create tension

Format your response exactly as follows:
Answer: [option number]
Analysis: [your reasoning for this decision]
"""
    
    messages = [{"role": "user", "content": coordinator_prompt}]
    coordinator_response = call_llm_api(messages, coordinator_model)
    
    # Extract coordinator's answer and analysis
    final_answer, coordinator_analysis = extract_answer_and_analysis(coordinator_response)
    
    return final_answer, coordinator_analysis, coordinator_response

def average_decision(cognitive_data, affective_data, behavioral_data):
    """
    Decision using averaging method: calculate the average of three perspective answers and round
    """
    # Collect valid answers
    valid_answers = []
    if cognitive_data['answer'] is not None:
        valid_answers.append(cognitive_data['answer'])
    if affective_data['answer'] is not None:
        valid_answers.append(affective_data['answer'])
    if behavioral_data['answer'] is not None:
        valid_answers.append(behavioral_data['answer'])
    
    # If no valid answers, return None
    if not valid_answers:
        return None, "no_valid_answers", {}
    
    # At least one valid answer, but may be None
    if len(valid_answers) == 1:
        return valid_answers[0], "single_answer", {}
    
    # If all answers are the same, just return that value
    if len(set(valid_answers)) == 1:
        return valid_answers[0], "unanimous", {}
    
    # Calculate the average and round to nearest integer
    average = sum(valid_answers) / len(valid_answers)
    ceiling_average = math.ceil(average)  # Use ceiling for rounding
    
    return ceiling_average, "average_ceiling", {
        "average": average,
        "valid_answers": valid_answers
    }

def collaborative_decision(cognitive_data, affective_data, behavioral_data, question_options, questions_data, question_id, coordinator_model, use_coordinator=True):
    """
    Make collaborative decision based on answers from three perspectives
    """
    # Ensure all answers are valid
    valid_answers = []
    if cognitive_data['answer'] is not None:
        valid_answers.append(cognitive_data['answer'])
    if affective_data['answer'] is not None:
        valid_answers.append(affective_data['answer'])
    if behavioral_data['answer'] is not None:
        valid_answers.append(behavioral_data['answer'])
    
    # If no valid answers, return None
    if not valid_answers:
        return None, "no_valid_answers", {}
    
    # If only one valid answer, return directly
    if len(valid_answers) == 1:
        return valid_answers[0], "single_answer", {}
    
    # Check if all answers are the same
    if len(set(valid_answers)) == 1:
        return valid_answers[0], "unanimous", {}
    
    # Choose decision method based on settings
    if use_coordinator:
        # Use coordinator for decision
        final_answer, coordinator_analysis, coordinator_response = coordinator_decision(
            question_id, 
            cognitive_data, 
            affective_data, 
            behavioral_data, 
            questions_data,
            coordinator_model
        )
        return final_answer, "coordinator", {
            "coordinator_analysis": coordinator_analysis,
            "coordinator_response": coordinator_response
        }
    else:
        # Use averaging decision
        return average_decision(cognitive_data, affective_data, behavioral_data)

def load_data(csv_file):
    data = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader, start=1):
            row['Row_Number'] = i
            data.append(row)
    return data

def load_questions_json(questions_file):
    with open(questions_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_nature_options_json(nature_options_file):
    with open(nature_options_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_split_from_json(json_file):
    """
    Load grouping data from the specified JSON file.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("test_set", [])
    profile_qns_train = data.get("train_set", [])

    return questions, profile_qns_train

def calculate_normalized_errors_with_accuracy(results, questions_data):
    """
    Calculate normalized MAE and accuracy
    """
    all_normalized_mae = []
    all_accuracies = []
    correct_count = 0
    total_count = 0
    
    # Iterate through each subject's results
    for subject_id, responses in results.items():
        for response in responses:
            question_id = response.get("question_id")
            predicted_answer = response.get("answer")
            original_answer = response.get("original_answer")
            
            # Skip invalid data
            if predicted_answer is None or original_answer is None:
                continue
                
            # Get options for this question
            options = questions_data[question_id].get("options", {})
            
            # If this question has multiple options
            if options:
                # Check if original answer is valid
                if str(original_answer) not in options.keys():
                    continue
                    
                # Calculate maximum possible difference
                max_diff = len(options) - 1
                
                # Calculate accuracy
                if predicted_answer == original_answer:
                    correct_count += 1
                total_count += 1
                
                # Calculate normalized absolute error
                norm_mae_error = abs(predicted_answer - original_answer) / max_diff
                all_normalized_mae.append(norm_mae_error)
    
    # Calculate overall accuracy and MAE
    accuracy = correct_count / total_count if total_count > 0 else 0
    mae = np.mean(all_normalized_mae) if all_normalized_mae else None
    
    return accuracy, mae

def process_multi_agent_question(user, question, story, profile_qns, questions_data, answer_model, coordinator_model, use_coordinator=True):
    """
    Process a single question using multi-perspective collaboration
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Process questions from three perspectives
    cognitive_data = {}
    affective_data = {}
    behavioral_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks for three perspectives
        cognitive_future = executor.submit(
            process_perspective_question_with_analysis,
            user, question, "cognitive", story, questions_data, answer_model
        )
        
        affective_future = executor.submit(
            process_perspective_question_with_analysis,
            user, question, "affective", story, questions_data, answer_model
        )
        
        behavioral_future = executor.submit(
            process_perspective_question_with_analysis,
            user, question, "behavioral", story, questions_data, answer_model
        )
        
        # Get results
        try:
            cognitive_answer, cognitive_analysis, cognitive_response, cognitive_prompt = cognitive_future.result()
            cognitive_data = {
                "answer": cognitive_answer,
                "analysis": cognitive_analysis,
                "response": cognitive_response,
                "prompt": cognitive_prompt
            }
        except Exception as e:
            print(f"Error in cognitive perspective for User {row_number}, Q {question_id}: {e}")
            cognitive_data = {
                "answer": None,
                "analysis": f"Error: {str(e)}",
                "response": f"Error: {str(e)}",
                "prompt": None
            }
        
        try:
            affective_answer, affective_analysis, affective_response, affective_prompt = affective_future.result()
            affective_data = {
                "answer": affective_answer,
                "analysis": affective_analysis,
                "response": affective_response,
                "prompt": affective_prompt
            }
        except Exception as e:
            print(f"Error in affective perspective for User {row_number}, Q {question_id}: {e}")
            affective_data = {
                "answer": None,
                "analysis": f"Error: {str(e)}",
                "response": f"Error: {str(e)}",
                "prompt": None
            }
        
        try:
            behavioral_answer, behavioral_analysis, behavioral_response, behavioral_prompt = behavioral_future.result()
            behavioral_data = {
                "answer": behavioral_answer,
                "analysis": behavioral_analysis,
                "response": behavioral_response,
                "prompt": behavioral_prompt
            }
        except Exception as e:
            print(f"Error in behavioral perspective for User {row_number}, Q {question_id}: {e}")
            behavioral_data = {
                "answer": None,
                "analysis": f"Error: {str(e)}",
                "response": f"Error: {str(e)}",
                "prompt": None
            }
    
    # Get question options data
    question_options = questions_data.get(question_id, {}).get("options", {})
    
    # Make collaborative decision
    final_answer, decision_method, decision_data = collaborative_decision(
        cognitive_data, 
        affective_data, 
        behavioral_data, 
        question_options,
        questions_data,
        question_id,
        coordinator_model,
        use_coordinator
    )
    
    # Return results
    return {
        "question_id": question_id,
        "cognitive": cognitive_data,
        "affective": affective_data,
        "behavioral": behavioral_data,
        "answer": final_answer,
        "decision_method": decision_method,
        "decision_data": decision_data
    }

def incremental_profile_test(
    csv_file, 
    questions_file, 
    nature_options_file, 
    split_dir,
    sample_size, 
    story_model,
    answer_model,
    coordinator_model=None,
    max_profile_qns=None,
    step_size=10,
    random_seed=42,
    use_coordinator=True,
    result_file='incremental_results.json'
):
    """
    Conduct incremental testing, gradually increasing the number of profile questions to test impact on performance
    Using multi-perspective collaboration method to generate answers
    
    Parameters:
    - csv_file: Path to CSV data file
    - questions_file: Path to questions JSON file
    - nature_options_file: Path to natural language options JSON file
    - split_dir: Path to grouping data JSON file
    - sample_size: Number of users to sample
    - story_model: Model for generating stories
    - answer_model: Model for answering questions
    - coordinator_model: Model for coordinating decisions
    - max_profile_qns: Maximum number of profile questions, if None use all questions
    - step_size: Number of questions to add in each increment
    - random_seed: Random seed
    - use_coordinator: Whether to use coordinator for decision making
    - result_file: Path to save results file
    """
    # If coordinator model not specified, use answer model
    if coordinator_model is None:
        coordinator_model = answer_model
    
    # Set random seed
    random.seed(random_seed)
    
    # Load data
    data = load_data(csv_file)
    questions_data = load_questions_json(questions_file)
    nature_options_data = load_nature_options_json(nature_options_file)
    
    # Load question grouping data
    questions, profile_qns_train = load_split_from_json(split_dir)
    
    # Randomly sample data
    sampled_data = random.sample(data, sample_size)
    
    # If maximum questions not specified, use all training set questions
    if max_profile_qns is None:
        max_profile_qns = len(profile_qns_train)
    else:
        max_profile_qns = min(max_profile_qns, len(profile_qns_train))
    
    # Create dictionary to store all results
    decision_method = "coordinator" if use_coordinator else "average"
    incremental_results = {
        'metadata': {
            'setting': 'multi_agent_voter',
            'sample_size': sample_size,
            'story_model': story_model,
            'answer_model': answer_model,
            'coordinator_model': coordinator_model,
            'decision_method': decision_method,
            'max_profile_qns': max_profile_qns,
            'step_size': step_size,
            'random_seed': random_seed,
            'test_questions': questions,
            'profile_questions': profile_qns_train
        },
        'results': []
    }
    
    # Loop to increase profile question count
    # Loop starts from 0
    for num_profile_qns in range(0, max_profile_qns + step_size, step_size):
        actual_num = min(num_profile_qns, max_profile_qns)
        print(f"\n--- Testing with {actual_num} profile questions ---")
        
        # Select profile questions, handle case of 0
        selected_profile_qns = []
        if actual_num > 0:
            selected_profile_qns = random.sample(profile_qns_train, actual_num)
        
        # Prepare user data
        user_data = []
        
        for row in sampled_data:
            # Randomly select profile questions for each user individually
            selected_profile_qns = []
            if actual_num > 0:
                selected_profile_qns = random.sample(profile_qns_train, actual_num)
            
            # Generate profile text
            if selected_profile_qns:
                profile_text_list = generate_profile_text_list(row, selected_profile_qns, nature_options_data)
            else:
                profile_text_list = []  # Empty profile text list
            
            # Prepare test questions
            questions_list = [
                {"question": q, "embedding": None} for q in questions if q in questions_data
            ]
            
            # Add to user data
            user_data.append({
                "Row_Number": row["Row_Number"],
                "profile_texts": profile_text_list,
                "questions": questions_list,
                "original_answers": {q: int(row[q]) for q in questions if q in row},
                "selected_profile_qns": selected_profile_qns  # Optional: save selected questions for analysis
            })
        
        # Generate stories for each user, set to empty if no profile information
        print(f"Generating stories for {len(user_data)} users...")
        stories = {}
        
        for user in tqdm(user_data, desc="Generating stories"):
            row_number = user["Row_Number"]
            profile_texts = user["profile_texts"]
            
            if profile_texts:
                profile_text = generate_full_profile_text(profile_texts)
                # Generate story
                story = generate_story_with_llm(profile_text, story_model)
            else:
                # When no profile information, set to empty string
                story = ""
            
            stories[str(row_number)] = story
        
        # Generate answers for each user
        print(f"Generating answers using multi-agent collaboration...")
        all_results = {}
        
        # Create task queue
        all_tasks = []
        for user in user_data:
            row_number = user["Row_Number"]
            story = stories.get(str(row_number), "")
            
            for question in user["questions"]:
                all_tasks.append((user, question, row_number, story))
        
        # Store task results
        task_results = {}
        
        # Process all tasks in parallel
        with tqdm(total=len(all_tasks), desc=f"Processing questions with {actual_num} profile items") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                # Submit all tasks
                futures = {}
                for i, (user, question, row_number, story) in enumerate(all_tasks):
                    future = submit_with_delay(
                        executor,
                        process_multi_agent_question,
                        user, question, story, selected_profile_qns, questions_data, 
                        answer_model, coordinator_model, use_coordinator
                    )
                    futures[future] = (row_number, question)
                
                # Process results
                for future in concurrent.futures.as_completed(futures):
                    row_number, question = futures[future]
                    question_id = question["question"]
                    try:
                        result = future.result()
                        
                        # Store results
                        if row_number not in task_results:
                            task_results[row_number] = []
                        
                        # Find original answer
                        original_answer = None
                        for u in user_data:
                            if u["Row_Number"] == row_number:
                                original_answer = u["original_answers"].get(question_id, None)
                                break
                        
                        # Add original answer to results
                        result["original_answer"] = original_answer
                        task_results[row_number].append(result)
                        
                    except Exception as e:
                        print(f"Error processing question for User {row_number}, Q {question_id}: {e}")
                        
                        if row_number not in task_results:
                            task_results[row_number] = []
                        
                        # Find original answer
                        original_answer = None
                        for u in user_data:
                            if u["Row_Number"] == row_number:
                                original_answer = u["original_answers"].get(question_id, None)
                                break
                                
                        task_results[row_number].append({
                            "question_id": question_id,
                            "error": f"Processing error: {str(e)}",
                            "original_answer": original_answer
                        })
                        
                    finally:
                        pbar.update(1)
        
        # Calculate performance metrics
        accuracy, mae = calculate_normalized_errors_with_accuracy(task_results, questions_data)
        
        print(f"Results with {actual_num} profile questions:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        # Add current step results to overall results
        incremental_results['results'].append({
            'num_profile_questions': actual_num,
            'accuracy': accuracy,
            'mae': mae,
            'detailed_results': copy.deepcopy(task_results) # Save detailed results
        })
        
        # Save results after each iteration to prevent loss in case of interruption
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(incremental_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nIncremental testing completed. Results saved to {result_file}")
    
    # Return final results
    return incremental_results

# Function to visualize results
def plot_incremental_results(result_file, output_file):
    """
    Visualize incremental test results
    
    Parameters:
    - result_file: Path to JSON results file
    - output_file: Path to output image file
    """
    import matplotlib.pyplot as plt
    
    # Load results
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract data
    profile_nums = [r['num_profile_questions'] for r in results['results']]
    accuracies = [r['accuracy'] for r in results['results']]
    maes = [r['mae'] for r in results['results']]
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot accuracy curve
    ax1.plot(profile_nums, accuracies, 'o-', label='Accuracy', color='blue')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Effect of Profile Information Quantity on Performance')
    ax1.grid(True)
    ax1.legend()
    
    # Plot MAE curve (lower is better)
    ax2.plot(profile_nums, maes, 'o-', label='MAE (lower is better)', color='red')
    ax2.set_xlabel('Number of Profile Questions')
    ax2.set_ylabel('MAE')
    ax2.grid(True)
    ax2.legend()
    
    # Save chart
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Results visualization saved to {output_file}")

# Main function
if __name__ == "__main__":
    # File paths
    csv_file = '../WVS_dataset/WVS_Cross-National_Wave_7_csv_v6_0.csv'
    questions_file = '../WVS_dataset/questions.json'
    nature_options_file = '../WVS_dataset/nature_options.json'
    split_dir = "../data_split/fromCV_SPLIT_FOLD_5.json"
    
    # Experiment parameters
    sample_size = 20  # Number of users to sample
    story_model = "llama-3.1-8b"  # Model for generating stories
    answer_model = "llama-3.1-8b"  # Model for answering questions
    coordinator_model = "llama-3.1-8b"  # Model for coordinating decisions
    #"gpt-4o-mini"
    
    max_profile_qns = 232  # Maximum number of profile questions to use
    step_size = 58  # Number of questions to add in each step
    random_seed = 42  # Random seed
    use_coordinator = True  # Set to False to use averaging instead of coordinator
    
    # Results file
    decision_method = "coordinator" if use_coordinator else "average"
    result_file = f'0519_incremental_results_multi_agent_{decision_method}_{answer_model}_sample{sample_size}_step{step_size}_seed{random_seed}.json'
    output_plot = result_file.replace('.json', '.png')
    
    # Run incremental test
    incremental_results = incremental_profile_test(
        csv_file, 
        questions_file, 
        nature_options_file, 
        split_dir,
        sample_size, 
        story_model,
        answer_model,
        coordinator_model,
        max_profile_qns=max_profile_qns,
        step_size=step_size,
        random_seed=random_seed,
        use_coordinator=use_coordinator,
        result_file=result_file
    )
    
    # Visualize results
    plot_incremental_results(result_file, output_plot)