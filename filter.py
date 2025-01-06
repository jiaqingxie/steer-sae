import re


def evaluate_correction(file_path):
    """
    Evaluates the correction of model answers by segmenting examples
    from 'Question' to the nearest 'Num of total question'.

    Args:
    file_path (str): Path to the input text file.

    Returns:
    dict: A dictionary with total examples, false examples corrected, and correction percentage.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize variables
    examples = []
    current_example = []
    for line in lines:
        if line.startswith("Question:"):
            if current_example:  # Save the current example
                examples.append("\n".join(current_example))
                current_example = []
        current_example.append(line.strip())
        if line.startswith("Num of total question"):
            if current_example:  # End of an example
                examples.append("\n".join(current_example))
                current_example = []
    if current_example:
        examples.append("\n".join(current_example))  # Add the last example

    false_examples = 0
    corrected = 0

    for example in examples:
        # Extract information using regex
        is_correct_match = re.search(r"Is correct: (True|False)", example)
        model_completion_match = re.search(r"Model Completion: (.+?)(?:Question|Is correct|Num of total question)",
                                           example, re.DOTALL)
        answers_match = re.search(r"Answers: ([0-9.\-:\/+=*]+)", example)

        # Skip if required fields are missing
        if not is_correct_match or not model_completion_match or not answers_match:
            continue

        is_correct = is_correct_match.group(1) == "True"

        # Only process incorrect cases
        if not is_correct:
            false_examples += 1
            answers = answers_match.group(1).strip()
            model_completion = model_completion_match.group(1).strip()

            # Define a function to filter valid standalone numbers
            def extract_standalone_numbers(text):
                """
                Extract numbers from text that are not part of a mathematical expression.
                - Exclude numbers before '='.
                - Exclude numbers surrounded by '+', '-', '*', '/'.
                """
                # Remove anything after '=' (including '=' itself)
                if '=' in text:
                    text = text.split('=')[-1]

                # Find all standalone numbers
                all_matches = re.finditer(r"(?<![\+\-*/=])\b(-?\d+\.?\d*|[-+]?\d+/[-+]?\d+)\b(?![\+\-*/=])", text)
                valid_numbers = []
                for match in all_matches:
                    number = match.group(1)
                    if "/" in number:  # Handle fractions
                        try:
                            numerator, denominator = map(float, number.split('/'))
                            if denominator != 0:
                                valid_numbers.append(numerator / denominator)
                        except ValueError:
                            continue
                    else:
                        try:
                            valid_numbers.append(float(number))
                        except ValueError:
                            continue
                return valid_numbers

            # Extract valid standalone numbers from model completion
            valid_numbers = extract_standalone_numbers(model_completion)

            if len(valid_numbers) < 2:
                continue  # Skip if fewer than 2 valid numbers

            # Get the second-to-last and third-to-last numbers as candidates
            corrected_candidates = valid_numbers[-2:]  # Take last two valid numbers

            try:
                ground_truth_answer = float(answers)
            except ValueError:
                continue

            # Check if any of the candidate answers matches the ground truth
            if ground_truth_answer in corrected_candidates:
                corrected += 1
                # print(example)
                # print("-----------------------")

    correction_percentage = (corrected / false_examples * 100) if false_examples > 0 else 0
    return {
        "Total Examples": len(examples),
        "False Examples": false_examples,
        "Corrected": corrected,
        "Correction Percentage": correction_percentage
    }


# # Re-run the function on the file
file_path = 'sae_9b_asdiv_0shot_C400_T4_omega0.5_6782.txt'
results_segments_fixed = evaluate_correction(file_path)
print(results_segments_fixed)
