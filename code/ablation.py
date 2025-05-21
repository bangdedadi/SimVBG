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
import os  # For folder operations
import math  # For mathematical calculations

ALL_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q65", "Q66", "Q67", "Q68", "Q69", "Q70", "Q71", "Q72", "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87", "Q88", "Q89", "Q90", "Q91", "Q92", "Q93", "Q94", "Q95", "Q96", "Q97", "Q98", "Q99", "Q100", "Q101", "Q102", "Q103", "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q112", "Q113", "Q114", "Q115", "Q116", "Q117", "Q118", "Q119", "Q120", "Q121", "Q122", "Q123", "Q124", "Q125", "Q126", "Q127", "Q128", "Q129", "Q130", "Q131", "Q132", "Q133", "Q134", "Q135", "Q136", "Q137", "Q138", "Q139", "Q140", "Q141", "Q142", "Q143", "Q144", "Q145", "Q146", "Q147", "Q148", "Q149", "Q150", "Q151", "Q152", "Q153", "Q154", "Q155", "Q156", "Q157", "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q164", "Q165", "Q166", "Q167", "Q168", "Q169", "Q170", "Q171", "Q172", "Q173", "Q174", "Q175", "Q176", "Q177", "Q178", "Q179", "Q180", "Q181", "Q182", "Q183", "Q184", "Q185", "Q186", "Q187", "Q188", "Q189", "Q190", "Q191", "Q192", "Q193", "Q194", "Q195", "Q196", "Q197", "Q198", "Q199", "Q200", "Q201", "Q202", "Q203", "Q204", "Q205", "Q206", "Q207", "Q208", "Q209", "Q210", "Q211", "Q212", "Q213", "Q214", "Q215", "Q216", "Q217", "Q218", "Q219", "Q220", "Q221", "Q222", "Q223", "Q224", "Q225", "Q226", "Q227", "Q228", "Q229", "Q230", "Q231", "Q232", "Q233", "Q234", "Q235", "Q236", "Q237", "Q238", "Q239", "Q240", "Q241", "Q242", "Q243", "Q244", "Q245", "Q246", "Q247", "Q248", "Q249", "Q250", "Q251", "Q252", "Q253", "Q254", "Q255", "Q256", "Q257", "Q258", "Q259", "Q260", "Q261", "Q262", "Q263", "Q264", "Q265", "Q266", "Q267", "Q268", "Q269", "Q270", "Q271", "Q272", "Q273", "Q274", "Q275", "Q276", "Q277", "Q278", "Q279", "Q280", "Q281", "Q282", "Q283", "Q284", "Q285", "Q286", "Q287", "Q288", "Q289", "Q290"]

# Initialize different clients
def init_client(model_name):
    if model_name in ["gpt-3.5-turbo", "text-davinci-003","gpt-4o-mini",  "claude-3-haiku-20240307","deepseek-v3-250324" ]:
        return openai.OpenAI(
            base_url="your_base_url_here",
            api_key="your_api_key_here",
            http_client=httpx.Client(base_url="your_base_url_here", follow_redirects=True),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def submit_with_delay(executor, func, *args, delay=0.1, **kwargs):
    future = executor.submit(func, *args, **kwargs)
    time.sleep(delay)  # Delay submitting the next task
    return future

# **Call LLM API function - Modified to accept model parameter**
def call_llm_api(messages, model):
    client = init_client(model)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )
    return completion.choices[0].message.content

def call_llm_api_stream(messages, model):
    client = init_client(model)
    full_response = ""
    try:
        # Use streaming response
        for chunk in client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            stream=True  # Enable streaming response
        ):
            # Extract content and accumulate to complete response
            if chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    # You can add progress printing here, e.g: print(".", end="", flush=True)
    except Exception as e:
        print(f"Stream API call error: {e}")
        # If streaming call fails, try non-streaming call
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                stream=False
            )
            full_response = completion.choices[0].message.content
        except Exception as e2:
            print(f"Non-streaming API call also failed: {e2}")
            raise e2
    
    return full_response

# **Helper function: Extract GPT's first number and analysis**
def extract_answer_and_analysis(response):
    """
    Extract answer and analysis from LLM response
    Format is 'Answer: [number]\nAnalysis: [analysis text]'
    """
    answer_match = re.search(r"Answer:\s*(\d+)", response)
    answer = int(answer_match.group(1)) if answer_match else None
    
    analysis_match = re.search(r"Analysis:(.*?)(?=Answer:|$)", response, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    
    # If Analysis marker not found but there's other text, use all content except the Answer line as analysis
    if not analysis and answer is not None:
        parts = response.split('\n', 1)
        if len(parts) > 1:
            analysis = parts[1].strip()
    
    return answer, analysis

# **Helper function: Extract GPT's first number**
def extract_first_number(response):
    match = re.search(r"\b\d+\b", response)
    if match:
        return int(match.group())
    else:
        return None

# **Generate natural language form of question**
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

# Format options as text
def format_options(options):
    if not options:
        return "No options available"
    return "\n".join([f"{key}: {value}" for key, value in options.items()])

# **Generate user profile text for Setting 1 and Setting 2**
def generate_full_profile_text(profile_texts):
    return "\n".join(profile_texts)

# **New: Combine story and original profile**
def combine_story_and_profile(story, profile_text):
    """
    Concatenate the generated story and original profile information in specified format
    
    Parameters:
    - story: Generated user story
    - profile_text: Original profile text
    
    Returns:
    - Combined text
    """
    combined_text = f"brief story:\n{story}\n\noriginal profile:\n{profile_text}"
    return combined_text

# **Helper function: Get profile text for specific perspective**
def get_perspective_profile(profile_qns, row, nature_options_data, questions_data, perspective, use_nature_options=True):
    """
    Filter relevant questions based on perspective and generate corresponding profile text
    
    Parameters:
    - profile_qns: List of all profile questions
    - row: User answer data row
    - nature_options_data: Natural language options data
    - questions_data: Questions data
    - perspective: Perspective type ('cognitive', 'affective', 'behavioral')
    - use_nature_options: Whether to use nature_options_data to generate profile
    
    Returns:
    - Profile text list for the perspective
    """
    # Get corresponding perspective questions set
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

# **Helper function: Generate profile text list (natural language form)**
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
                #else:
                #    print(f"Warning: Option value {option_value} not found in options for question {q}")
            
            # Handle fill-in questions
            elif template:
                # Use template to generate natural language
                profile_info_list.append(template.format(value=option_value))
            
            # Default handling
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
            #else:
            #    print(f"Warning: Option value {option_value} not found in options for question {q}")
                
    return profile_info_list

# **Generate unified story**
def generate_story_with_llm(profile_text, model):
    """
    Generate user story
    
    Parameters:
    - profile_text: User profile text
    - model: Model used for story generation
    
    Returns:
    - Generated story
    """
    # Read story prompt template from file or create default template
    try:
        # Create directory (if not exists)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Try to load story prompt template
        with open("Q_prompts/storymodule.txt", "r", encoding="utf-8") as f:
            story_template = f.read()
            
    except FileNotFoundError:
        # Create default template
        print("storymodule.txt not found, using default template.")
        default_template = (
            "Please generate a coherent and engaging story about a person with the following profile information:\n\n"
            "{profile_text}\n\n"
            "Create a narrative that captures this person's background, values, opinions, and life experiences. "
            "Include their thought processes, beliefs, emotional reactions, relationships, behavioral patterns, "
            "and how they typically act in various situations. "
            "The story should be in third person and between 400-600 words."
        )
        
        # Create directory (if not exists)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Save default template
        with open("Q_prompts/story.txt", "w", encoding="utf-8") as f:
            f.write(default_template)
        
        story_template = default_template

    # Replace placeholders
    prompt_content = story_template.format(profile_text=profile_text)
    #print(prompt_content)
    messages = [{"role": "user", "content": prompt_content}]
    #return call_llm_api(messages, model)
    return call_llm_api_stream(messages, model)

# **New: Modify story function**
def modify_story_with_llm(profile_text, current_story, model):
    """
    Create an improved story based on current story and original profile.
    
    Parameters:
    - profile_text: Original user profile text
    - current_story: Current story
    - model: Model used for story generation
    
    Returns:
    - Modified story
    """
    # Read story modification prompt template from file or create default template
    try:
        # Create directory (if not exists)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Try to load story modification prompt template
        with open("Q_prompts/story_modify2.txt", "r", encoding="utf-8") as f:
            modify_template = f.read()
            
    except FileNotFoundError:
        # Create default template
        print("story_modify2.txt not found, using default template.")
        default_template = (
            "Please improve the following story to better reflect the user profile information. The story should "
            "remain coherent and engaging while more accurately capturing the details, values, opinions, and life "
            "experiences from the profile.\n\n"
            "ORIGINAL PROFILE INFORMATION:\n{profile_text}\n\n"
            "CURRENT STORY:\n{current_story}\n\n"
            "Please provide an improved version of the story that:\n"
            "1. More accurately represents all the information in the profile\n"
            "2. Maintains narrative coherence and engagement\n"
            "3. Is written in third person\n"
            "4. Is between 400-600 words\n\n"
            "IMPROVED STORY:"
        )
        
        # Save default template
        with open("Q_prompts/story_modify2.txt", "w", encoding="utf-8") as f:
            f.write(default_template)
        
        modify_template = default_template

    # Replace placeholders
    prompt_content = modify_template.format(
        profile_text=profile_text,
        current_story=current_story
    )

    messages = [{"role": "user", "content": prompt_content}]
    return call_llm_api(messages, model)

# **New: Load previously generated stories from cache**
def load_stories_from_cache(cache_file):
    """
    Load previously generated stories from cache file.
    If file doesn't exist, return empty dictionary.
    """
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Cache file '{cache_file}' not found. Returning empty dict.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{cache_file}'. Returning empty dict.")
        return {}

# Fix evaluate_story_accuracy function
def evaluate_story_accuracy(story, profile_qns, row, questions_data, model):
    """
    Evaluate story accuracy on profile questions.
    
    Parameters:
    - story: Story to evaluate
    - profile_qns: List of profile questions
    - row: Original user data row
    - questions_data: Questions data
    - model: Model for evaluation
    
    Returns:
    - accuracy: Accuracy rate
    - results: Detailed evaluation results
    """
    # Filter valid profile questions - only select those existing in both row and questions_data
    valid_profile_qns = [
        q for q in profile_qns 
        if q in row and q in questions_data and row[q] not in ('-1', '-2', '-3', '-4', '-5')
    ]
    
    # If no valid questions, return zero accuracy
    if not valid_profile_qns:
        return 0, {"error": "No valid profile questions found", "valid_questions": 0}
    
    # Create natural language form of profile questions
    profile_questions_text = "\n\n".join([
        generate_question_text_with_options(q, questions_data) 
        for q in valid_profile_qns
    ])
    
    # Construct evaluation prompt
    prompt = (
        f"Please read the following user story and answer the questions as if you were this person. "
        f"For each question, provide only the number of your answer choice.\n\n"
        f"USER STORY:\n{story}\n\n"
        f"QUESTIONS:\n{profile_questions_text}\n\n"
        f"Format your answers as 'Q[question_number]: [answer_number]' with one answer per line."
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Call LLM to get answers
        response = call_llm_api(messages, model)
        
        # Extract answers
        results = {}
        pattern = re.compile(r"(Q\d+):\s*(\d+)")
        matches = pattern.findall(response)
        
        for qid, answer in matches:
            if qid in valid_profile_qns:
                results[qid] = int(answer)
        
        # Calculate accuracy
        correct_count = 0
        total_count = 0
        
        evaluation_details = []
        
        for q in valid_profile_qns:
            if q in results:
                try:
                    original_answer = int(row[q])
                    predicted_answer = results[q]
                    
                    is_correct = original_answer == predicted_answer
                    if is_correct:
                        correct_count += 1
                    
                    # Record evaluation details for each question
                    evaluation_details.append({
                        "question_id": q,
                        "original_answer": original_answer,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct
                    })
                    
                    total_count += 1
                except (ValueError, KeyError):
                    # Skip questions that can't be processed
                    continue
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Return accuracy and detailed results
        return accuracy, {
            "correct_count": correct_count,
            "total_count": total_count,
            "evaluation_details": evaluation_details,
            "answers": results,
            "prompt": prompt,
            "response": response,
            "valid_questions": len(valid_profile_qns)
        }
        
    except Exception as e:
        print(f"Error in evaluate_story_accuracy: {e}")
        return 0, {"error": str(e)}

# Fixed optimize_story function
def optimize_user_story(profile_text, profile_qns, row, questions_data, model, num_variations=5, max_iterations=3):
    """
    Optimize story through iterations.
    
    Parameters:
    - profile_text: User profile text
    - profile_qns: List of profile questions
    - row: Original user data row
    - questions_data: Questions data
    - model: Model for story generation
    - num_variations: Number of variations to generate in each iteration
    - max_iterations: Maximum number of iterations
    
    Returns:
    - best_story: Best story
    - optimization_history: Optimization history
    """
    # Initialize optimization history
    optimization_history = []
    
    # Generate initial story
    initial_story = generate_story_with_llm(profile_text, model)
    
    try:
        # Evaluate initial story
        initial_accuracy, initial_eval_results = evaluate_story_accuracy(
            initial_story, profile_qns, row, questions_data, model
        )
    except Exception as e:
        print(f"Error evaluating initial story: {e}")
        initial_accuracy = 0
        initial_eval_results = {"error": str(e)}
    
    # Record initial story information
    optimization_history.append({
        "iteration": 0,
        "story": initial_story,
        "accuracy": initial_accuracy,
        "eval_results": initial_eval_results
    })
    
    # Initialize best story and accuracy
    best_story = initial_story
    best_accuracy = initial_accuracy
    
    print(f"Initial story accuracy: {initial_accuracy:.4f}")
    
        # Iterate optimization
    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}/{max_iterations}")
        
        # Generate multiple story variations
        variation_results = []
        modified_stories = []
        
        # Stage 1: Generate variations in parallel
        print("  Generating story variations...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit tasks
            futures = []
            for i in range(num_variations):
                future = submit_with_delay(
                    executor,
                    modify_story_with_llm,
                    profile_text,
                    best_story,
                    model
                )
                futures.append(future)
            
            # Collect generated stories, don't evaluate immediately
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    modified_story = future.result()
                    modified_stories.append((i, modified_story))
                except Exception as e:
                    print(f"  Error generating variation {i+1}: {e}")
        
        # Stage 2: Evaluate all variations in parallel
        print("  Evaluating story variations...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as eval_executor:
            # Submit evaluation tasks
            eval_futures = {}
            for i, modified_story in modified_stories:
                future = eval_executor.submit(
                    evaluate_story_accuracy,
                    modified_story, 
                    profile_qns, 
                    row, 
                    questions_data, 
                    model
                )
                eval_futures[future] = (i, modified_story)
            
            # Collect evaluation results
            for future in concurrent.futures.as_completed(eval_futures):
                i, modified_story = eval_futures[future]
                try:
                    accuracy, eval_results = future.result()
                    
                    # Record variation results
                    variation_results.append({
                        "story": modified_story,
                        "accuracy": accuracy,
                        "eval_results": eval_results
                    })
                    
                    print(f"  Variation {i+1}/{len(modified_stories)}: Accuracy = {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  Error evaluating variation {i+1}: {e}")
        
        # Find the best variation
        if variation_results:
            best_variation = max(variation_results, key=lambda x: x["accuracy"])
            
            # Record best variation of this iteration
            optimization_history.append({
                "iteration": iteration,
                "story": best_variation["story"],
                "accuracy": best_variation["accuracy"],
                "eval_results": best_variation["eval_results"],
                "all_variations": variation_results
            })
            
            # Check for improvement
            if best_variation["accuracy"] > best_accuracy:
                best_story = best_variation["story"]
                best_accuracy = best_variation["accuracy"]
                print(f"  Found better story! New accuracy: {best_accuracy:.4f}")
            else:
                print(f"  No improvement in this iteration. Best accuracy remains: {best_accuracy:.4f}")
                # If no improvement, terminate early
                break
        else:
            print("  No valid variations generated in this iteration.")
            break
    
    return best_story, optimization_history

# **Modified: Unified story generation function with story optimization**
def generate_optimized_unified_stories(user_data, profile_qns, nature_options_data, questions_data, story_model, sampled_data, cache_file=None, use_nature_options=True, enable_optimization=True, num_variations=5, max_iterations=3):
    """
    Generate an optimized unified story for each user.
    
    Parameters:
    - user_data: User data list
    - profile_qns: Profile questions list
    - nature_options_data: Natural language options data
    - questions_data: Questions data
    - story_model: Model for story generation
    - sampled_data: Original sample data
    - cache_file: Cache file path
    - use_nature_options: Whether to use nature_options_data to generate profile
    - enable_optimization: Whether to enable story optimization
    - num_variations: Number of variations to generate in each iteration
    - max_iterations: Maximum number of iterations
    
    Returns:
    - Generated stories dictionary, format {user_id: story}
    """
    # Create results dictionary
    unified_stories = {}
    
    # Create mapping of original data rows for quick lookup
    original_data_map = {row['Row_Number']: row for row in sampled_data}
    
    # Create optimization history directory and dictionary
    optimization_history_dir = "story_optimization_history"
    if enable_optimization:
        os.makedirs(optimization_history_dir, exist_ok=True)
        # Create global optimization history dictionary, keyed by user ID
        all_optimization_history = {}
    
    def generate_user_story(user, original_data_row):
        """Generate (optimized) story for single user"""
        row_number = user["Row_Number"]
        
        # Use all of user's profile text
        profile_texts = user["profile_texts"]
        full_profile_text = generate_full_profile_text(profile_texts)
        
        try:
            if enable_optimization:
                # Use optimization process to generate story
                print(f"\nOptimizing story for User {row_number}...")
                best_story, optimization_history = optimize_user_story(
                    full_profile_text, 
                    profile_qns, 
                    original_data_row,
                    questions_data, 
                    story_model, 
                    num_variations=num_variations, 
                    max_iterations=max_iterations
                )
                
                # Add optimization history to global dictionary
                all_optimization_history[str(row_number)] = optimization_history
                
                story = best_story
            else:
                # Directly generate story
                story = generate_story_with_llm(full_profile_text, story_model)
        except Exception as e:
            print(f"Error generating story for User {row_number}: {e}")
            story = full_profile_text  # If failed, use original text
        
        return row_number, story
    
    # Use ThreadPoolExecutor for parallel processing
    with tqdm(total=len(user_data), desc="Generating unified stories") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {
                executor.submit(
                    generate_user_story, 
                    user, 
                    original_data_map.get(user['Row_Number'], {})
                ): user['Row_Number'] 
                for user in user_data
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    row_number, story = future.result()
                    unified_stories[str(row_number)] = story
                except Exception as e:
                    print(f"Error in parallel processing for story: {e}")
                finally:
                    pbar.update(1)
    
    # If cache file path provided, save generated stories
    if cache_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else ".", exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(unified_stories, f, indent=4, ensure_ascii=False)
    
    # If optimization enabled, save all users' optimization history to a file
    if enable_optimization:
        history_file = os.path.join(optimization_history_dir, "all_users_optimization_history.json")
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(all_optimization_history, f, indent=4, ensure_ascii=False)
    
    return unified_stories

# **New: Average method for decision making**
def average_decision(cognitive_data, affective_data, behavioral_data):
    """
    Use average method for decision: calculate average of three perspective answers and round
    
    Parameters:
    - cognitive_data: Cognitive perspective answer data {answer, analysis, response}
    - affective_data: Affective perspective answer data {answer, analysis, response}
    - behavioral_data: Behavioral perspective answer data {answer, analysis, response}
    
    Returns:
    - final_answer: Final result of average decision
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
    ceiling_average = math.ceil(average)  # Use ceiling function
    
    return ceiling_average, "average_ceiling", {
        "average": average,
        "valid_answers": valid_answers
    }

# **New: Coordinator decision function**
def coordinator_decision(question_id, cognitive_data, affective_data, behavioral_data, questions_data, coordinator_model):
    """
    When answers from three perspectives are inconsistent, use coordinator to make final decision
    
    Parameters:
    - question_id: Question ID
    - cognitive_data: Cognitive perspective answer data {answer, analysis}
    - affective_data: Affective perspective answer data {answer, analysis}
    - behavioral_data: Behavioral perspective answer data {answer, analysis}
    - questions_data: Questions data
    - coordinator_model: Model for coordination
    
    Returns:
    - final_answer: Coordinator's final decision
    - coordinator_response: Coordinator's complete response
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

# **Modified: Multi-perspective collaborative decision function, add option to use average**
def collaborative_decision(cognitive_data, affective_data, behavioral_data, question_options, questions_data, question_id, coordinator_model, use_coordinator=True):
    """
    Make collaborative decision based on answers from three perspectives
    
    Parameters:
    - cognitive_data: Cognitive perspective answer data {answer, analysis, response}
    - affective_data: Affective perspective answer data {answer, analysis, response}
    - behavioral_data: Behavioral perspective answer data {answer, analysis, response}
    - question_options: Question options data
    - questions_data: All questions data
    - question_id: Question ID
    - coordinator_model: Model for coordination
    - use_coordinator: Whether to use coordinator (True) or average decision (False)
    
    Returns:
    - final_answer: Final decision
    - method: Decision method used
    - decision_data: Decision-related data
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
        # Use average decision
        return average_decision(cognitive_data, affective_data, behavioral_data)

# Extract numerical value from GPT response, only extract first number after Qn
def extract_answers_from_response(response, questions):
    """
    Extract answers from GPT's response text, format 'Qn: number'
    """
    pattern = re.compile(r"\b\d+\b")  # Match first number sequence
    matches = pattern.search(response)  # Use search to find only first match
    if matches:
        first_number = int(matches.group())
        return first_number

# **Read CSV file**
def load_data(csv_file):
    data = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader, start=1):
            row['Row_Number'] = i
            data.append(row)
    return data

# **Load JSON file**
def load_questions_json(questions_file):
    with open(questions_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_nature_options_json(nature_options_file):
    with open(nature_options_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_profile_questions_text(profile_qns, nature_options_data):
    """
    Convert Profile questions to natural language questions and generate complete text.
    """
    question_text_list = []
    for q in profile_qns:
        if q in nature_options_data:
            question_data = nature_options_data[q]
            question_text = question_data.get("question_text", f"Unknown question for {q}")
            options = question_data.get("options", {})
            if options:
                # Add options to question text
                options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
                question_text_list.append(f"{question_text}\nOptions:\n{options_text}")
            else:
                question_text_list.append(question_text)
    return "\n".join([f"Q{idx + 1}: {text}" for idx, text in enumerate(question_text_list)])

# **New helper function: Load group data from JSON file**
def load_split_from_json(json_file):
    """
    Load group data from specified JSON file.
    :param json_file: JSON file path
    :return: seed_information, questions, profile_qns_train, profile_qns_val, other_groups
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract group content
    #seed_information = data.get("seed_information", [])
    questions = data.get("test_set", [])
    profile_qns_train = data.get("train_set", [])
    #profile_qns_val = data.get("validation_set", [])
    #other_groups = data.get("other_groups", None)  # If this field doesn't exist in JSON, return None

    #return seed_information, questions, profile_qns_train, profile_qns_val, other_groups
    return questions, profile_qns_train

# **New: Generate answers function using combined text (story+original profile) with perspective collaboration**
def generate_answers_with_combined_collaborative_agents(user_data, questions_data, nature_options_data, output_file, stories, profile_qns, sampled_data, answer_model=None, coordinator_model=None, use_nature_options=True, use_coordinator=True):
    """
    Generate answers using combined text (story+original profile) with perspective collaboration.
    Generate answers and analyses from three different perspectives for each user's questions, if answers are inconsistent, use coordinator to make final decision.
    
    Parameters:
    - user_data: User data list
    - questions_data: Questions data
    - nature_options_data: Natural language options data
    - output_file: Output file path
    - stories: Generated stories dictionary {user_id: story}
    - profile_qns: Profile questions list
    - sampled_data: Original sample data
    - answer_model: Model for generating answers
    - coordinator_model: Model for coordination, if None use answer_model
    - use_nature_options: Whether to use nature_options_data to generate profile
    - use_coordinator: Whether to use coordinator (True) or average decision (False)
    """
    # If coordinator model not specified, use answer model
    if coordinator_model is None:
        coordinator_model = answer_model
        
    results = defaultdict(dict)
    
    # Create mapping of original data rows for quick lookup
    original_data_map = {row['Row_Number']: row for row in sampled_data}
    
    # Create task queue
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_data_row = original_data_map.get(row_number, {})
        
        # Get user's story
        user_story = stories.get(str(row_number), "")
        
        # Get original profile text
        profile_texts = user["profile_texts"]
        original_profile_text = generate_full_profile_text(profile_texts)
        
        # Combine story and original profile
        combined_text = combine_story_and_profile(user_story, original_profile_text)
        
        for question in user["questions"]:
            question_id = question["question"]
            # Create tasks for three perspectives for each user's question
            all_tasks.append((user, question, "cognitive", row_number, question_id, combined_text))
            all_tasks.append((user, question, "affective", row_number, question_id, combined_text))
            all_tasks.append((user, question, "behavioral", row_number, question_id, combined_text))
    
    # Store all task results
    task_results = {}
    
    # Process all tasks in parallel
    with tqdm(total=len(all_tasks), desc="Processing perspective questions") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to thread pool, don't wait for results immediately
            futures = {}
            for i, (user, question, perspective, row_number, question_id, combined_text) in enumerate(all_tasks):
                # Use delayed submission to prevent API overload
                future = executor.submit(
                    process_combined_perspective_question_with_analysis,
                    user, question, perspective, combined_text, answer_model, questions_data
                )
                futures[future] = (row_number, question_id, perspective)
                # Small delay to prevent API requests clustering
                time.sleep(0.05)
            
            # Process completed futures
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id, perspective = futures[future]
                try:
                    answer, analysis, response, prompt = future.result()
                    # Store result
                    if (row_number, question_id) not in task_results:
                        task_results[(row_number, question_id)] = {}
                    task_results[(row_number, question_id)][perspective] = {
                        "answer": answer,
                        "analysis": analysis,
                        "response": response,
                        "prompt": prompt
                    }
                except Exception as e:
                    print(f"Error in {perspective} for User {row_number}, Q {question_id}: {e}")
                    # Record error information
                    if (row_number, question_id) not in task_results:
                        task_results[(row_number, question_id)] = {}
                    task_results[(row_number, question_id)][perspective] = {
                        "answer": None,
                        "analysis": f"Error: {str(e)}",
                        "response": f"Error: {str(e)}",
                        "prompt": None
                    }
                finally:
                    pbar.update(1)
    
    # Integrate results and make decisions
    decision_method = "coordinator" if use_coordinator else "average"
    print(f"Aggregating results and making {decision_method} decisions...")
    
    # Calculate total number of decisions
    total_decisions = sum(len(user["questions"]) for user in user_data)
    
    # Prepare all tasks to process
    decision_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_answers = user["original_answers"]
        
        for question in user["questions"]:
            question_id = question["question"]
            key = (row_number, question_id)
            
            if key in task_results:
                decision_tasks.append((row_number, question_id, original_answers, task_results[key]))
            else:
                # This question has no results, also add to tasks for unified processing
                decision_tasks.append((row_number, question_id, original_answers, None))
    
    # Create result storage structure
    decision_results = {}
    
    # Process decision tasks in parallel
    with tqdm(total=len(decision_tasks), desc=f"Making {decision_method} decisions") as decision_pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as decision_executor:
            # Submit all decision tasks
            futures = {}
            for task in decision_tasks:
                row_number, question_id, original_answers, results_for_question = task
                
                future = decision_executor.submit(
                    process_single_decision,
                    row_number,
                    question_id,
                    results_for_question,
                    questions_data,
                    original_answers.get(question_id, None),
                    coordinator_model,
                    use_coordinator
                )
                futures[future] = (row_number, question_id)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    decision_result = future.result()
                    
                    # Organize results
                    if row_number not in decision_results:
                        decision_results[row_number] = []
                    
                    decision_results[row_number].append(decision_result)
                    
                except Exception as e:
                    print(f"Error in decision for User {row_number}, Q {question_id}: {e}")
                    
                    # Record error
                    if row_number not in decision_results:
                        decision_results[row_number] = []
                    
                    decision_results[row_number].append({
                        "question_id": question_id,
                        "error": f"Decision error: {str(e)}",
                        "original_answer": original_answers.get(question_id, None)
                    })
                    
                finally:
                    decision_pbar.update(1)
    
    # Organize final results
    results = {}
    for user in user_data:
        row_number = user["Row_Number"]
        
        # If there are decision results for this user
        if row_number in decision_results:
            results[row_number] = decision_results[row_number]
        else:
            # If no results, record error
            results[row_number] = [{
                "question_id": q["question"],
                "error": "No decision results",
                "original_answer": user["original_answers"].get(q["question"], None)
            } for q in user["questions"]]
    
    # Save answer results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

# **New: Process single perspective question using combined text and provide analysis**
def process_combined_perspective_question_with_analysis(user, question, perspective, combined_text, answer_model, questions_data):
    """
    Process single perspective question using combined text and generate answer and analysis
    
    Parameters:
    - user: User data
    - question: Question data
    - perspective: Perspective name ('cognitive', 'affective', 'behavioral')
    - combined_text: Combined text (story+original profile)
    - answer_model: Model for generating answers
    
    Returns:
    - answer: Extracted answer number
    - analysis: Extracted analysis text
    - response: Complete LLM response
    - prompt: Prompt used
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)

    # Load perspective-specific prompt template from Q_prompts folder
    try:
        # Create directory (if not exists)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Try to load perspective-specific prompt template
        prompt_file_path = f"Q_prompts/prompt_{perspective}.txt"
    
        # Read prompt template
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        # Replace placeholders
        prompt = prompt_template.format(
            profile_text=combined_text,
            question_text_with_options=question_text_with_options
        )
        
    except Exception as e:
        print(f"Error loading prompt template for {perspective} perspective: {e}")
        # Fall back to modified basic prompt, requesting answer and analysis
        prompt = (
            f"Please simulate the role of a user with the following user profile and answer the question as if you were this person.\n\n"
            f"User Profile:\n{combined_text}\n\n"
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

# **Modified: Multi-perspective collaborative answer generation function - Modified to accept model parameter**
def generate_answers_with_collaborative_agents(user_data, questions_data, nature_options_data, output_file, perspective_stories, profile_qns, sampled_data, answer_model=None, coordinator_model=None, use_nature_options=True, use_coordinator=True):
    """
    Generate answers using multi-perspective collaboration. Each perspective provides answer and analysis, if answers are inconsistent, use coordinator to make final decision.
    
    Parameters:
    - use_coordinator: Whether to use coordinator (True) or average decision (False)
    """
    # If coordinator model not specified, use answer model
    if coordinator_model is None:
        coordinator_model = answer_model
        
    results = defaultdict(dict)
    
    # Create mapping of original data rows for quick lookup
    original_data_map = {row['Row_Number']: row for row in sampled_data}
    
    # Create task queue
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_data_row = original_data_map.get(row_number, {})
        
        for question in user["questions"]:
            question_id = question["question"]
            # Create tasks for three perspectives for each user's question
            all_tasks.append((user, question, "cognitive", row_number, question_id, original_data_row))
            all_tasks.append((user, question, "affective", row_number, question_id, original_data_row))
            all_tasks.append((user, question, "behavioral", row_number, question_id, original_data_row))
    
    # Store all task results
    task_results = {}
    
    # Process all tasks in parallel
    with tqdm(total=len(all_tasks), desc="Processing perspective questions") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to thread pool, don't wait for results immediately
            futures = {}
            for i, (user, question, perspective, row_number, question_id, original_data_row) in enumerate(all_tasks):
                # Use delayed submission to prevent API overload
                future = executor.submit(
                    process_perspective_question_with_analysis,
                    user, question, perspective, perspective_stories, profile_qns,
                    original_data_row, questions_data, nature_options_data, answer_model, use_nature_options
                )
                futures[future] = (row_number, question_id, perspective)
                # Small delay to prevent API requests clustering
                time.sleep(0.05)
            
            # Process completed futures
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id, perspective = futures[future]
                try:
                    answer, analysis, response, prompt = future.result()
                    # Store result
                    if (row_number, question_id) not in task_results:
                        task_results[(row_number, question_id)] = {}
                    task_results[(row_number, question_id)][perspective] = {
                        "answer": answer,
                        "analysis": analysis,
                        "response": response,
                        "prompt": prompt
                    }
                except Exception as e:
                    print(f"Error in {perspective} for User {row_number}, Q {question_id}: {e}")
                    # Record error information
                    if (row_number, question_id) not in task_results:
                        task_results[(row_number, question_id)] = {}
                    task_results[(row_number, question_id)][perspective] = {
                        "answer": None,
                        "analysis": f"Error: {str(e)}",
                        "response": f"Error: {str(e)}",
                        "prompt": None
                    }
                finally:
                    pbar.update(1)
    
    # Integrate results and make decisions - Add progress bar and parallel processing
    decision_method = "coordinator" if use_coordinator else "average"
    print(f"Aggregating results and making {decision_method} decisions...")
    
    # Calculate total number of decisions
    total_decisions = sum(len(user["questions"]) for user in user_data)
    
    # Prepare all tasks to process
    decision_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_answers = user["original_answers"]
        
        for question in user["questions"]:
            question_id = question["question"]
            key = (row_number, question_id)
            
            if key in task_results:
                decision_tasks.append((row_number, question_id, original_answers, task_results[key]))
            else:
                # This question has no results, also add to tasks for unified processing
                decision_tasks.append((row_number, question_id, original_answers, None))
    
    # Create result storage structure
    decision_results = {}
    
    # Process decision tasks in parallel
    with tqdm(total=len(decision_tasks), desc=f"Making {decision_method} decisions") as decision_pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as decision_executor:
            # Submit all decision tasks
            futures = {}
            for task in decision_tasks:
                row_number, question_id, original_answers, results_for_question = task
                
                future = decision_executor.submit(
                    process_single_decision,
                    row_number,
                    question_id,
                    results_for_question,
                    questions_data,
                    original_answers.get(question_id, None),
                    coordinator_model,
                    use_coordinator
                )
                futures[future] = (row_number, question_id)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    decision_result = future.result()
                    
                    # Organize results
                    if row_number not in decision_results:
                        decision_results[row_number] = []
                    
                    decision_results[row_number].append(decision_result)
                    
                except Exception as e:
                    print(f"Error in decision for User {row_number}, Q {question_id}: {e}")
                    
                    # Record error
                    if row_number not in decision_results:
                        decision_results[row_number] = []
                    
                    decision_results[row_number].append({
                        "question_id": question_id,
                        "error": f"Decision error: {str(e)}",
                        "original_answer": original_answers.get(question_id, None)
                    })
                    
                finally:
                    decision_pbar.update(1)
    
    # Organize final results
    results = {}
    for user in user_data:
        row_number = user["Row_Number"]
        
        # If there are decision results for this user
        if row_number in decision_results:
            results[row_number] = decision_results[row_number]
        else:
            # If no results, record error
            results[row_number] = [{
                "question_id": q["question"],
                "error": "No decision results",
                "original_answer": user["original_answers"].get(q["question"], None)
            } for q in user["questions"]]
    
    # Save answer results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

# **Helper function: Process single perspective question and provide analysis**
def process_perspective_question_with_analysis(user, question, perspective, perspective_stories, profile_qns, original_data_row, questions_data, nature_options_data, answer_model, use_nature_options=True):
    """
    Process single perspective question and generate answer and analysis
    
    Parameters:
    - user: User data
    - question: Question data
    - perspective: Perspective name ('cognitive', 'affective', 'behavioral')
    - perspective_stories: Perspective stories dictionary
    - profile_qns: Profile questions list
    - original_data_row: Original data row
    - questions_data: Questions data
    - nature_options_data: Natural language options data
    - answer_model: Model for generating answers
    - use_nature_options: Whether to use nature_options_data to generate profile
    
    Returns:
    - answer: Extracted answer number
    - analysis: Extracted analysis text
    - response: Complete LLM response
    - prompt: Prompt used
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Get story for this perspective
    if str(row_number) in perspective_stories and perspective in perspective_stories[str(row_number)]:
        story = perspective_stories[str(row_number)][perspective]
    else:
        # If no story for this perspective, get profile text for this perspective
        perspective_profile_texts = get_perspective_profile(
            profile_qns, original_data_row, nature_options_data, questions_data, perspective, use_nature_options
        )
        
        # If no questions found for this perspective, use all profile text
        if not perspective_profile_texts:
            perspective_profile_texts = user["profile_texts"]
            
        story = generate_full_profile_text(perspective_profile_texts)
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)

    # Load perspective-specific prompt template from Q_prompts folder
    try:
        # Create directory (if not exists)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Try to load perspective-specific prompt template
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
        # Fall back to modified basic prompt, requesting answer and analysis
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

# Modified: Process single decision, add parameter to control whether to use coordinator
def process_single_decision(row_number, question_id, results_for_question, questions_data, original_answer, coordinator_model, use_coordinator=True):
    """
    Process single decision task
    
    Parameters:
    - row_number: User row number
    - question_id: Question ID
    - results_for_question: Perspective results for this question
    - questions_data: Questions data
    - original_answer: Original answer
    - coordinator_model: Coordinator model
    - use_coordinator: Whether to use coordinator (True) or average decision (False)
    
    Returns:
    - Decision record
    """
    # If no results or results is None, return error
    if results_for_question is None:
        return {
            "question_id": question_id,
            "error": "No results available for this question",
            "original_answer": original_answer
        }
    
    # Get data from three perspectives
    cognitive_data = results_for_question.get("cognitive", {"answer": None, "analysis": "", "response": ""})
    affective_data = results_for_question.get("affective", {"answer": None, "analysis": "", "response": ""})
    behavioral_data = results_for_question.get("behavioral", {"answer": None, "analysis": "", "response": ""})
    
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
    
    # Return decision record
    return {
        "question_id": question_id,
        "cognitive": cognitive_data,
        "affective": affective_data,
        "behavioral": behavioral_data,
        "answer": final_answer,
        "decision_method": decision_method,
        "decision_data": decision_data,
        "original_answer": original_answer
    }

# **Modified: Simple perspective collaborative answer generation function, add parameter to control whether to use coordinator**
def generate_answers_with_simple_collaborative_agents(user_data, questions_data, nature_options_data, output_file, profile_qns, sampled_data, answer_model=None, coordinator_model=None, use_nature_options=True, use_coordinator=True):
    """
    Generate answers using simple perspective collaboration. Directly use original profile data to generate answers and analyses from three different perspectives for each user's questions, if answers are inconsistent, use coordinator to make final decision.
    
    Parameters:
    - user_data: User data list
    - questions_data: Questions data
    - nature_options_data: Natural language options data
    - output_file: Output file path
    - profile_qns: Profile questions list
    - sampled_data: Original sample data
    - answer_model: Model for generating answers
    - coordinator_model: Model for coordination, if None use answer_model
    - use_nature_options: Whether to use nature_options_data to generate profile
    - use_coordinator: Whether to use coordinator (True) or average decision (False)
    """
    # If coordinator model not specified, use answer model
    if coordinator_model is None:
        coordinator_model = answer_model
        
    results = defaultdict(dict)
    
    # Create mapping of original data rows for quick lookup
    original_data_map = {row['Row_Number']: row for row in sampled_data}
    
    # Create task queue
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_data_row = original_data_map.get(row_number, {})
        
        for question in user["questions"]:
            question_id = question["question"]
            # Create tasks for three perspectives for each user's question
            all_tasks.append((user, question, "cognitive", row_number, question_id, original_data_row))
            all_tasks.append((user, question, "affective", row_number, question_id, original_data_row))
            all_tasks.append((user, question, "behavioral", row_number, question_id, original_data_row))
    
    # Store all task results
    task_results = {}
    
    # Process all tasks in parallel
    with tqdm(total=len(all_tasks), desc="Processing perspective questions") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to thread pool, don't wait for results immediately
            futures = {}
            for i, (user, question, perspective, row_number, question_id, original_data_row) in enumerate(all_tasks):
                # Use delayed submission to prevent API overload
                future = executor.submit(
                    process_simple_perspective_question_with_analysis,
                    user, question, perspective, profile_qns,
                    original_data_row, questions_data, nature_options_data, answer_model, use_nature_options
                )
                futures[future] = (row_number, question_id, perspective)
                # Small delay to prevent API requests clustering
                time.sleep(0.05)
            
            # Process completed futures
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id, perspective = futures[future]
                try:
                    answer, analysis, response, prompt = future.result()
                    # Store result
                    if (row_number, question_id) not in task_results:
                        task_results[(row_number, question_id)] = {}
                    task_results[(row_number, question_id)][perspective] = {
                        "answer": answer,
                        "analysis": analysis,
                        "response": response,
                        "prompt": prompt
                    }
                except Exception as e:
                    print(f"Error in {perspective} for User {row_number}, Q {question_id}: {e}")
                    # Record error information
                    if (row_number, question_id) not in task_results:
                        task_results[(row_number, question_id)] = {}
                    task_results[(row_number, question_id)][perspective] = {
                        "answer": None,
                        "analysis": f"Error: {str(e)}",
                        "response": f"Error: {str(e)}",
                        "prompt": None
                    }
                finally:
                    pbar.update(1)
    
    # Integrate results and make decisions
    decision_method = "coordinator" if use_coordinator else "average"
    print(f"Aggregating results and making {decision_method} decisions...")
    
    # Calculate total number of decisions
    total_decisions = sum(len(user["questions"]) for user in user_data)
    
    # Prepare all tasks to process
    decision_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_answers = user["original_answers"]
        
        for question in user["questions"]:
            question_id = question["question"]
            key = (row_number, question_id)
            
            if key in task_results:
                decision_tasks.append((row_number, question_id, original_answers, task_results[key]))
            else:
                # This question has no results, also add to tasks for unified processing
                decision_tasks.append((row_number, question_id, original_answers, None))
    
    # Create result storage structure
    decision_results = {}
    
    # Process decision tasks in parallel
    with tqdm(total=len(decision_tasks), desc=f"Making {decision_method} decisions") as decision_pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as decision_executor:
            # Submit all decision tasks
            futures = {}
            for task in decision_tasks:
                row_number, question_id, original_answers, results_for_question = task
                
                future = decision_executor.submit(
                    process_single_decision,
                    row_number,
                    question_id,
                    results_for_question,
                    questions_data,
                    original_answers.get(question_id, None),
                    coordinator_model,
                    use_coordinator
                )
                futures[future] = (row_number, question_id)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    decision_result = future.result()
                    
                    # Organize results
                    if row_number not in decision_results:
                        decision_results[row_number] = []
                    
                    decision_results[row_number].append(decision_result)
                    
                except Exception as e:
                    print(f"Error in decision for User {row_number}, Q {question_id}: {e}")
                    
                    # Record error
                    if row_number not in decision_results:
                        decision_results[row_number] = []
                    
                    decision_results[row_number].append({
                        "question_id": question_id,
                        "error": f"Decision error: {str(e)}",
                        "original_answer": original_answers.get(question_id, None)
                    })
                    
                finally:
                    decision_pbar.update(1)
    
    # Organize final results
    results = {}
    for user in user_data:
        row_number = user["Row_Number"]
        
        # If there are decision results for this user
        if row_number in decision_results:
            results[row_number] = decision_results[row_number]
        else:
            # If no results, record error
            results[row_number] = [{
                "question_id": q["question"],
                "error": "No decision results",
                "original_answer": user["original_answers"].get(q["question"], None)
            } for q in user["questions"]]
    
    # Save answer results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

# **Helper function: Process simple perspective question and provide analysis**
def process_simple_perspective_question_with_analysis(user, question, perspective, profile_qns, original_data_row, questions_data, nature_options_data, answer_model, use_nature_options=True):
    """
    Process single perspective question using original profile data and generate answer and analysis
    
    Parameters:
    - user: User data
    - question: Question data
    - perspective: Perspective name ('cognitive', 'affective', 'behavioral')
    - profile_qns: Profile questions list
    - original_data_row: Original data row
    - questions_data: Questions data
    - nature_options_data: Natural language options data
    - answer_model: Model for generating answers
    - use_nature_options: Whether to use nature_options_data to generate profile
    
    Returns:
    - answer: Extracted answer number
    - analysis: Extracted analysis text
    - response: Complete LLM response
    - prompt: Prompt used
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Get profile text for this perspective
    perspective_profile_texts = get_perspective_profile(
        profile_qns, original_data_row, nature_options_data, questions_data, perspective, use_nature_options
    )
    
    # If no questions found for this perspective, use all profile text
    if not perspective_profile_texts:
        perspective_profile_texts = user["profile_texts"]
        
    profile_text = generate_full_profile_text(perspective_profile_texts)
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)

    # Load perspective-specific prompt template from Q_prompts folder
    try:
        # Create directory (if not exists)
        os.makedirs("Q_prompts", exist_ok=True)
        
        # Try to load perspective-specific prompt template
        prompt_file_path = f"Q_prompts/prompt_{perspective}.txt"
    
        # Read prompt template
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        # Replace placeholders
        prompt = prompt_template.format(
            profile_text=profile_text,
            question_text_with_options=question_text_with_options
        )
        
    except Exception as e:
        print(f"Error loading prompt template for {perspective} perspective: {e}")
        # Fall back to modified basic prompt, requesting answer and analysis
        prompt = (
            f"Please simulate the role of a user with the following user profile and answer the question as if you were this person.\n\n"
            f"User Profile:\n{profile_text}\n\n"
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

# **New: Generate answers using single agent function**
def generate_answers_with_single_agent(user_data, questions_data, stories, output_file, profile_qns, sampled_data, answer_model=None):
    """
    Generate answers using single agent, without three-module parallel answering and aggregation steps
    
    Parameters:
    - user_data: User data list
    - questions_data: Questions data
    - stories: Generated stories dictionary {user_id: story}
    - output_file: Output file path
    - profile_qns: Profile questions list
    - sampled_data: Original sample data
    - answer_model: Model for generating answers
    """
    results = defaultdict(dict)
    
    # Create mapping of original data rows for quick lookup
    original_data_map = {row['Row_Number']: row for row in sampled_data}
    
    # Create task queue
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_data_row = original_data_map.get(row_number, {})
        
        # Get user's story - only use story, don't combine with profile
        user_story = stories.get(str(row_number), "")
        
        for question in user["questions"]:
            question_id = question["question"]
            # Directly pass in story instead of combined text
            all_tasks.append((user, question, row_number, question_id, user_story))
    
    # Store all task results
    task_results = {}
    
    # Process all tasks in parallel
    with tqdm(total=len(all_tasks), desc="Processing questions with single agent") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to thread pool
            futures = {}
            for i, (user, question, row_number, question_id, story) in enumerate(all_tasks):
                # Use delayed submission to prevent API overload
                future = executor.submit(
                    process_single_agent_question,
                    user, question, story, answer_model, questions_data
                )
                futures[future] = (row_number, question_id)
                # Small delay to prevent API requests clustering
                time.sleep(0.05)
            
            # Process completed futures
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    answer, response, prompt = future.result()
                    
                    # Store result
                    if row_number not in task_results:
                        task_results[row_number] = []
                    
                    # Since single module only returns answer, not analysis, directly construct result
                    task_results[row_number].append({
                        "question_id": question_id,
                        "answer": answer,
                        "response": response,
                        "prompt": prompt,
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id, None)
                    })
                    
                except Exception as e:
                    print(f"Error processing question for User {row_number}, Q {question_id}: {e}")
                    
                    # Record error
                    if row_number not in task_results:
                        task_results[row_number] = []
                    
                    task_results[row_number].append({
                        "question_id": question_id,
                        "error": f"Processing error: {str(e)}",
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id, None)
                    })
                    
                finally:
                    pbar.update(1)
    
    # Save answer results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(task_results, f, indent=4, ensure_ascii=False)

# **New: Process single question function - for single module setup**
def process_single_agent_question(user, question, story, answer_model, questions_data):
    """
    Process question using single module and return answer directly, only use story without profile
    
    Parameters:
    - user: User data
    - question: Question data
    - story: User story
    - answer_model: Model for generating answers
    - questions_data: Questions data
    
    Returns:
    - answer: Extracted answer number
    - response: Complete LLM response
    - prompt: Prompt used
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)
    
    # Use simplified single module prompt - only pass in story, don't use profile
    prompt = (
        f"Question:{question_text_with_options}\n"
        f"User profile: {story}\n"  # Here directly use story as user profile
        f"Consider the question context and the user's background from the provided profile when formulating your response. "
        f"Aim for a balanced perspective that respects accuracy while reflecting the user's viewpoint.\n"
        f"Answer format: 'Q：option you selected'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_llm_api(messages, answer_model)
        
        # Extract number from response
        answer = extract_first_number(response)
        
        return answer, response, prompt
    except Exception as e:
        print(f"Error processing question for User {row_number}, Question {question_id}: {e}")
        return None, f"Error: {str(e)}", prompt

# **New: Generate answers using original profile function**
# **Fixed: Generate answers using original profile function**
def generate_answers_with_original_profile(user_data, questions_data, output_file, profile_qns, sampled_data, answer_model=None):
    """
    Generate answers using original profile directly, without generating story, and without three-module parallel answering and aggregation steps
    
    Parameters:
    - user_data: User data list
    - questions_data: Questions data
    - output_file: Output file path
    - profile_qns: Profile questions list
    - sampled_data: Original sample data
    - answer_model: Model for generating answers
    """
    results = defaultdict(dict)
    
    # Create mapping of original data rows for quick lookup
    original_data_map = {row['Row_Number']: row for row in sampled_data}
    
    # Create task queue
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        original_data_row = original_data_map.get(row_number, {})
        
        # Use generate_profile_text_list_withquestions to generate original text
        # Make sure to ignore use_nature_options parameter here, directly use questions_data
        profile_texts = generate_profile_text_list_withquestions(original_data_row, profile_qns, questions_data)
        original_profile_text = generate_full_profile_text(profile_texts)
        
        for question in user["questions"]:
            question_id = question["question"]
            all_tasks.append((user, question, row_number, question_id, original_profile_text))
    
    # Store all task results
    task_results = {}
    
    # Process all tasks in parallel
    with tqdm(total=len(all_tasks), desc="Processing questions with original profile") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to thread pool
            futures = {}
            for i, (user, question, row_number, question_id, original_profile_text) in enumerate(all_tasks):
                # Use delayed submission to prevent API overload
                future = executor.submit(
                    process_original_profile_question,
                    user, question, original_profile_text, answer_model, questions_data
                )
                futures[future] = (row_number, question_id)
                # Small delay to prevent API requests clustering
                time.sleep(0.05)
            
            # Process completed futures
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    answer, response, prompt = future.result()
                    
                    # Store result
                    if row_number not in task_results:
                        task_results[row_number] = []
                    
                    # Construct result
                    task_results[row_number].append({
                        "question_id": question_id,
                        "answer": answer,
                        "response": response,
                        "prompt": prompt,
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id, None)
                    })
                    
                except Exception as e:
                    print(f"Error processing question for User {row_number}, Q {question_id}: {e}")
                    
                    # Record error
                    if row_number not in task_results:
                        task_results[row_number] = []
                    
                    task_results[row_number].append({
                        "question_id": question_id,
                        "error": f"Processing error: {str(e)}",
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id, None)
                    })
                    
                finally:
                    pbar.update(1)
    
    # Save answer results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(task_results, f, indent=4, ensure_ascii=False)

# **New: Process single question using original profile function**
def process_original_profile_question(user, question, original_profile_text, answer_model, questions_data):
    """
    Process question using original profile and return answer directly
    
    Parameters:
    - user: User data
    - question: Question data
    - original_profile_text: Original profile text
    - answer_model: Model for generating answers
    - questions_data: Questions data
    
    Returns:
    - answer: Extracted answer number
    - response: Complete LLM response
    - prompt: Prompt used
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)
    
    # Use provided prompt template
    prompt = (
        f"Question:{question_text_with_options}\n"
        f"User profile: {original_profile_text}\n"
        f"Consider both the question context and the user's background when formulating your response. "
        f"Aim for a balanced perspective that respects accuracy while reflecting the user's viewpoint.\n"
        f"Answer format: 'Q：option you selected'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_llm_api(messages, answer_model)
        
        # Extract number from response
        answer = extract_first_number(response)
        
        return answer, response, prompt
    except Exception as e:
        print(f"Error processing question for User {row_number}, Question {question_id}: {e}")
        return None, f"Error: {str(e)}", prompt
    
# **New: Generate answers without profile function**
def generate_answers_with_blank_profile(user_data, questions_data, output_file, answer_model=None):
    """
    Generate answers without using any profile information, directly ask questions and get answers
    
    Parameters:
    - user_data: User data list
    - questions_data: Questions data
    - output_file: Output file path
    - answer_model: Model for generating answers
    """
    results = defaultdict(dict)
    
    # Create task queue
    all_tasks = []
    for user in user_data:
        row_number = user["Row_Number"]
        
        for question in user["questions"]:
            question_id = question["question"]
            all_tasks.append((user, question, row_number, question_id))
    
    # Store all task results
    task_results = {}
    
    # Process all tasks in parallel
    with tqdm(total=len(all_tasks), desc="Processing questions with blank profile") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to thread pool
            futures = {}
            for i, (user, question, row_number, question_id) in enumerate(all_tasks):
                # Use delayed submission to prevent API overload
                future = executor.submit(
                    process_blank_profile_question,
                    user, question, answer_model, questions_data
                )
                futures[future] = (row_number, question_id)
                # Small delay to prevent API requests clustering
                time.sleep(0.05)
            
            # Process completed futures
            for future in concurrent.futures.as_completed(futures):
                row_number, question_id = futures[future]
                try:
                    answer, response, prompt = future.result()
                    
                    # Store result
                    if row_number not in task_results:
                        task_results[row_number] = []
                    
                    # Construct result
                    task_results[row_number].append({
                        "question_id": question_id,
                        "answer": answer,
                        "response": response,
                        "prompt": prompt,
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id, None)
                    })
                    
                except Exception as e:
                    print(f"Error processing question for User {row_number}, Q {question_id}: {e}")
                    
                    # Record error
                    if row_number not in task_results:
                        task_results[row_number] = []
                    
                    task_results[row_number].append({
                        "question_id": question_id,
                        "error": f"Processing error: {str(e)}",
                        "original_answer": user_data[next((i for i, u in enumerate(user_data) if u["Row_Number"] == row_number), 0)]["original_answers"].get(question_id, None)
                    })
                    
                finally:
                    pbar.update(1)
    
    # Save answer results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(task_results, f, indent=4, ensure_ascii=False)

# **New: Process blank profile single question function**
def process_blank_profile_question(user, question, answer_model, questions_data):
    """
    Process question without using any profile information, directly return answer
    
    Parameters:
    - user: User data
    - question: Question data
    - answer_model: Model for generating answers
    - questions_data: Questions data
    
    Returns:
    - answer: Extracted answer number
    - response: Complete LLM response
    - prompt: Prompt used
    """
    row_number = user["Row_Number"]
    question_id = question["question"]
    
    # Construct prompt
    question_text_with_options = generate_question_text_with_options(question_id, questions_data)
    
    # Use simple prompt, without any user information
    prompt = (
        f"Question:{question_text_with_options}\n"
        f"Please answer the question.\n"
        f"Answer format: 'Q：option you selected'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_llm_api(messages, answer_model)
        
        # Extract number from response
        answer = extract_first_number(response)
        
        return answer, response, prompt
    except Exception as e:
        print(f"Error processing question for User {row_number}, Question {question_id}: {e}")
        return None, f"Error: {str(e)}", prompt

# **Modified main function, add new baseline settings**
def main(
    csv_file, 
    questions_file, 
    nature_options_file, 
    sample_size, 
    embedding_output_file, 
    edited_profile_cache, 
    results_output_file, 
    topk_num, 
    setting, 
    num_backstories, 
    wordnum, 
    case_dir, 
    split_dir, 
    story_model,  # Model used for story generation
    answer_model,  # Model used for answer generation
    coordinator_model=None,  # Model used by coordinator, default is None, will use answer_model
    random_seed=42,  # Random seed
    load_stories=False,  # Whether to load stories from cache
    distribution=False,
    use_nature_options=True,  # Control whether to use nature_options_data
    use_coordinator=True,  # Control whether to use coordinator
    optimize_story=False,  # New parameter: control whether to optimize story
    num_story_variations=5,  # New parameter: number of variations to generate in each iteration
    max_story_iterations=3  # New parameter: maximum number of iterations
):
    # If coordinator model not specified, use answer model
    if coordinator_model is None:
        coordinator_model = answer_model
    
    enable_story_optimization = optimize_story  # Rename variable
    # Load data
    data = load_data(csv_file)
    questions_data = load_questions_json(questions_file)
    nature_options_data = load_nature_options_json(nature_options_file)

    split_json_file = split_dir
    questions, profile_qns_train= load_split_from_json(split_json_file)

    # Random sample data
    random.seed(random_seed)
    sampled_data = random.sample(data, sample_size)
    profile_qns=profile_qns_train
    
    # Create cache directory for multi-agent simulation
    multi_agent_cache_dir = f"stories/multi_agent_sample={sample_size}_model={story_model.split('-', 1)[-1]}_seed={random_seed}"
    if optimize_story:
        multi_agent_cache_dir += "_optimized"
    os.makedirs(multi_agent_cache_dir, exist_ok=True)

    if setting in ["multi_agent_voter", "origintext_voter", "combined_profile_voter", "single_answer", "origin_full", "blank_profile"]:
        # **Prepare user data**
        print(f"Preparing data for {setting} setting...")
        user_data = []
        for row in sampled_data:
            # Decide which method to use for profile text generation based on use_nature_options parameter
            if use_nature_options:
                profile_text_list = generate_profile_text_list(row, profile_qns, nature_options_data)
            else:
                profile_text_list = generate_profile_text_list_withquestions(row, profile_qns, questions_data)
                
            questions_list = [
                {"question": q, "embedding": None} for q in questions if q in questions_data
            ]
            user_data.append({
                "Row_Number": row["Row_Number"],
                "profile_texts": profile_text_list,
                "questions": questions_list,
                "original_answers": {q: int(row[q]) for q in questions if q in row}
            })

        perspective_stories = None  # Initialize story variable
        stories = None  # Initialize unified stories dictionary
            
        # Process settings that need stories
        if setting in ["multi_agent_voter", "combined_profile_voter", "single_answer"]:
            story_type = "optimized" if enable_story_optimization else "unified"
            print(f"Generating {story_type} stories using {story_model}...")
            
            # Cache file path
            story_cache_name = "optimized_stories.json" if optimize_story else "unified_stories.json"
            story_cache_file = os.path.join(multi_agent_cache_dir, story_cache_name)
            
            # Check if there are cached stories
            if load_stories and os.path.exists(story_cache_file):
                # Try to load cached stories
                try:
                    with open(story_cache_file, "r", encoding="utf-8") as f:
                        stories = json.load(f)
                        print(f"Loaded {len(stories)} users' stories from cache.")
                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"Error loading cached stories. Generating new stories...")
                    stories = None
            
            # If cached stories not loaded or no valid cache found, generate new stories
            if stories is None or not stories:
                if enable_story_optimization:  # Use new variable name
                    # Use optimized story generation method
                    stories = generate_optimized_unified_stories(
                        user_data, 
                        profile_qns, 
                        nature_options_data,
                        questions_data,
                        story_model,
                        sampled_data,
                        cache_file=story_cache_file,
                        use_nature_options=use_nature_options,
                        enable_optimization=True,
                        num_variations=num_story_variations,
                        max_iterations=max_story_iterations
                    )
                else:
                    # Use basic story generation method
                    stories = generate_optimized_unified_stories(
                        user_data, 
                        profile_qns, 
                        nature_options_data,
                        questions_data,
                        story_model,
                        sampled_data,
                        cache_file=story_cache_file,
                        use_nature_options=use_nature_options,
                        enable_optimization=False
                    )
            
            if setting == "multi_agent_voter":
                # Convert to perspective stories format, make all perspectives use same story
                perspective_stories = {}
                for user_id, story in stories.items():
                    perspective_stories[user_id] = {
                        "cognitive": story,
                        "affective": story,
                        "behavioral": story
                    }
        
        # Generate answers - choose different answer generation method based on setting
        if setting == "multi_agent_voter":
            decision_method = "coordinator" if use_coordinator else "average"
            print(f"Generating answers using multi-agent collaboration with {answer_model} and {decision_method} decision...")
            generate_answers_with_collaborative_agents(
                user_data,
                questions_data,
                nature_options_data,
                results_output_file,
                perspective_stories,
                profile_qns,
                sampled_data,
                answer_model=answer_model,
                coordinator_model=coordinator_model,
                use_nature_options=use_nature_options,
                use_coordinator=use_coordinator
            )
        elif setting == "origintext_voter":
            decision_method = "coordinator" if use_coordinator else "average"
            print(f"Generating answers using simple collaboration with {answer_model} and {decision_method} decision...")
            generate_answers_with_simple_collaborative_agents(
                user_data,
                questions_data,
                nature_options_data,
                results_output_file,
                profile_qns,
                sampled_data,
                answer_model=answer_model,
                coordinator_model=coordinator_model,
                use_nature_options=use_nature_options,
                use_coordinator=use_coordinator
            )
        elif setting == "combined_profile_voter":
            decision_method = "coordinator" if use_coordinator else "average"
            print(f"Generating answers using combined profile collaboration with {answer_model} and {decision_method} decision...")
            generate_answers_with_combined_collaborative_agents(
                user_data,
                questions_data,
                nature_options_data,
                results_output_file,
                stories,  # Pass in unified stories dictionary
                profile_qns,
                sampled_data,
                answer_model=answer_model,
                coordinator_model=coordinator_model,
                use_nature_options=use_nature_options,
                use_coordinator=use_coordinator
            )
        elif setting == "single_answer":
            print(f"Generating answers using single agent with {answer_model}...")
            generate_answers_with_single_agent(
                user_data,
                questions_data,
                stories,  # Pass in unified stories dictionary
                results_output_file,
                profile_qns,
                sampled_data,
                answer_model=answer_model
            )
        # Add two new baseline settings
        elif setting == "origin_full":
            print(f"Generating answers using original profile (without story) with {answer_model}...")
            generate_answers_with_original_profile(
            user_data,
            questions_data,
            results_output_file,
            profile_qns,
            sampled_data,
            answer_model=answer_model
            )
        elif setting == "blank_profile":
            print(f"Generating answers without any profile information with {answer_model}...")
            generate_answers_with_blank_profile(
                user_data,
                questions_data,
                results_output_file,
                answer_model=answer_model
            )
    else:
        raise ValueError(f"Unsupported setting: {setting}")

# **Run main program**
if __name__ == "__main__":
    csv_file = '../WVS_dataset/WVS_Cross-National_Wave_7_csv_v6_0.csv'
    questions_file = '../WVS_dataset/questions.json'
    nature_options_file = '../WVS_dataset/nature_options.json'
    embedding_output_file = "../user_embeddings.json"
    edited_profile_cache = "../edited_profiles.json"
    
    # Model parameters
    #story_model = "gpt-3.5-turbo"  # Model used for story generation
    story_model = "deepseek-v3-250324"
    answer_model = "deepseek-v3-250324"  # Model used for question answering
    #answer_model = "claude-3-haiku-20240307"    # Model used for question answering
    coordinator_model = "deepseek-v3-250324"  # Model used for decision coordination, use stronger model
    #coordinator_model = "claude-3-haiku-20240307"    # Model used for question answering
    #"gpt-3.5-turbo"，"gpt-4o-mini","deepseek-v3-250324"
    
    # Experiment parameters
    sample_size = 100
    random_seed = 42
    
    # New control parameters
    use_nature_options = False # Set to False to use questions_data for profile generation
    use_coordinator = True  # Set to False to use average instead of coordinator
    optimize_story = False  # Set to True to enable story optimization
    num_story_variations = 5  # Number of variations to generate in each iteration
    max_story_iterations = 3  # Maximum number of iterations
    
    # Generate test name, include more parameter information
    decision_method = "coordinator" if use_coordinator else "average"
    story_method = "OptimizedStory" if optimize_story else "BasicStory"
    
    # Setting - added "origin_full" and "blank_profile" options
    setting = "origintext_voter"  # Options: "multi_agent_voter", "origintext_voter", "combined_profile_voter", "single_answer", "origin_full", "blank_profile"
    
    # Adjust test name based on setting
    if setting == "origin_full":
        use_nature_options = False
        test_name = f"baseline-origin_full-{answer_model}-sample={sample_size}-SEED={random_seed}"
    elif setting == "blank_profile":
        test_name = f"baseline-blank_profile-{answer_model}-sample={sample_size}-SEED={random_seed}"
    else:
        test_name = f"final-nostory-split=CV5-with{decision_method.capitalize()}-{story_method}-{setting}-{story_model}-{answer_model}-{coordinator_model}-sample={sample_size}-SEED={random_seed}"
        if not use_nature_options:
            test_name += "-UseQuestionData"
    results_output_file = "answers/"+test_name+".json"
    
    topk_num = 3
    num_backstories = 10
    wordnum = 1000
    case_dir = test_name
    use_distribution=False
    split_dir= "../data_split/fromCV_SPLIT_FOLD_5.json"
    
    # Whether to load stories from cache
    load_stories = True
    
    main(
        csv_file, 
        questions_file, 
        nature_options_file, 
        sample_size, 
        embedding_output_file, 
        edited_profile_cache, 
        results_output_file, 
        topk_num, 
        setting, 
        num_backstories, 
        wordnum, 
        case_dir=case_dir, 
        split_dir=split_dir,
        story_model=story_model,     # Pass in story generation model
        answer_model=answer_model,   # Pass in question answering model
        coordinator_model=coordinator_model,  # Pass in coordinator model
        random_seed=random_seed,     # Pass in random seed
        load_stories=load_stories,   # Whether to load stories from cache
        distribution=use_distribution,
        use_nature_options=use_nature_options,  # Whether to use nature_options_data
        use_coordinator=use_coordinator,  # Whether to use coordinator
        optimize_story=optimize_story,  # Whether to optimize story
        num_story_variations=num_story_variations,  # Number of variations to generate in each iteration
        max_story_iterations=max_story_iterations  # Maximum number of iterations
    )