from datetime import datetime
import os
import pandas as pd
import logging
import json
from pathlib import Path
import hashlib
import time
import re
from collections import Counter

def get_experiment_start_date():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_hash(string, algorithm="sha256"):
    """
    Calculates the hash of a given string using the specified algorithm.
    Defaults to SHA-256 if no algorithm is provided.
    """

    hash_object = getattr(hashlib, algorithm)()
    hash_object.update(string.encode('utf-8'))
    return hash_object.hexdigest()


def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timer_end_message = f"  :):):) Finished {func.__name__!r} in {run_time:.4f} seconds"
        print(timer_end_message)
        return value

    return wrapper_timer

# def map_response_info_seeking(text):
#     label = "na"
#     all_found_labels = []
#     # Define categories (using lowercase for case-insensitive matching)
#     categories = [
#         "not english",
#         "information seeking",
#         "coding",
#         "no request",
#         "reformulation",
#         "content creation"
#     ]
#     # Define letter mappings
#     letter_mappings = {
#         "1": "not english",
#         "2": "information seeking",
#         "3": "coding",
#         "4": "no request",
#         "5": "reformulation",
#         "6": "content creation",
#     }
#     # Clean the text by removing unnecessary formatting
#     text_cleaned = text.lower().strip()
#     text_cleaned = text_cleaned.replace("**", "").replace("\n", " ").strip()

#     # Extract category number if present in specific patterns
#     import re
#     match = re.search(r"(classification|answer):\s*(\d)", text_cleaned)
#     if match:
#         category_number = match.group(2)
#         if category_number in letter_mappings:
#             return letter_mappings[category_number]

#     # Check for category keywords in the cleaned text
#     for category in categories:
#         if category in text_cleaned:
#             all_found_labels.append(category)
#     # If only one category is found, use that
#     if len(all_found_labels) == 1:
#         label = all_found_labels[0]
#     # Otherwise check if text starts with category name or letter
#     else:
#         text_stripped = text_cleaned.strip()
#         # Check for category name at start
#         for category in categories:
#             if text_stripped.startswith(category):
#                 label = category
#                 break
#         # Check for letter notation at start (e.g., "4" for "no request")
#         if label == "na" and text_stripped and text_stripped[0] in letter_mappings:
#             label = letter_mappings[text_stripped[0]]
#     return label

# def map_response2label(text):
#     letter_output_mappings = {
#         '1': 'not english',
#         '2': 'information seeking',
#         '3': 'coding',
#         '4': 'no request',
#         '5': 'reformulation',
#         '6': 'content creation'
#     }
    
#     text_lower = text.lower()
    
#     # Find all matching categories by name
#     category_matches = []
#     for code, label in letter_output_mappings.items():
#         if label in text_lower:
#             category_matches.append(label)
    
#     # Find all digit references (1–6) as standalone or near word boundaries
#     digit_matches = re.findall(r'\b[1-6]\b', text)
    
#     # Filter out digits that are part of larger numbers
#     filtered_digit_matches = []
#     for match in digit_matches:
#         if re.search(rf'\b{match}\b', text):  # Ensure it's a standalone digit
#             filtered_digit_matches.append(match)
    
#     digit_labels = [letter_output_mappings[d] for d in filtered_digit_matches]
    
#     # Combine all detected labels
#     all_labels = category_matches + digit_labels
    
#     if not all_labels:
#         return "na"
    
#     # Return most common label
#     most_common = Counter(all_labels).most_common(1)[0][0]
#     return most_common

def map_response2label(text):
    letter_output_mappings = {
        '1': 'not english',
        '2': 'information seeking',
        '3': 'coding',
        '4': 'no request',
        '5': 'reformulation',
        '6': 'content creation'
    }
    
    text_lower = text.lower()
    
    # Find all matching categories by name
    category_matches = []
    for code, label in letter_output_mappings.items():
        if label in text_lower:
            category_matches.append(label)
    
    # Find all digit references (1–6) as standalone or near word boundaries
    digit_matches = re.findall(r'\b[1-6]\b', text_lower)
    
    # Filter out digits that are part of larger numbers or unrelated contexts
    filtered_digit_matches = []
    for match in digit_matches:
        # Ensure the digit is not part of a larger number
        if re.search(rf'\b{match}\b', text_lower) and not re.search(r'\d{2,}', text_lower):
            filtered_digit_matches.append(match)
    
    digit_labels = [letter_output_mappings[d] for d in filtered_digit_matches]
    
    # Combine all detected labels
    all_labels = category_matches + digit_labels
    
    if not all_labels:
        return "na"
    
    # Return most common label
    most_common = Counter(all_labels).most_common(1)[0][0]
    return most_common