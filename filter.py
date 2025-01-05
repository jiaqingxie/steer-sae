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
        answers_match = re.search(r"Answers: ([0-9.\-:]+)", example)

        # Skip if required fields are missing
        if not is_correct_match or not model_completion_match or not answers_match:
            continue

        is_correct = is_correct_match.group(1) == "True"

        # Only process incorrect cases
        if not is_correct:
            false_examples += 1
            answers = answers_match.group(1).strip()
            model_completion = model_completion_match.group(1).strip()

            # Check if Answers contains a colon ":"
            if ":" in answers:
                # Extract numbers before and after the colon in Answers
                answer_parts = re.findall(r"-?\d+\.?\d*", answers)
                if len(answer_parts) != 2:
                    continue  # Skip if there are not exactly two numbers
                answer_result = float(answer_parts[0]) / float(answer_parts[1])

                # Extract numbers before and after the colon in Model Completion
                model_parts = re.findall(r"-?\d+\.?\d*", model_completion)
                if len(model_parts) < 2:
                    continue  # Skip if there are not enough numbers
                # print(answer_parts[0])
                if float(model_parts[-1]) == 0:
                    # print(model_parts)
                    continue
                model_result = float(model_parts[-2]) / float(model_parts[-1])

                # Compare the calculated results
                if abs(answer_result - model_result) < 1e-6:  # Tolerance for floating-point comparison
                    corrected += 1
            else:
                # Extract all numbers from the model completion
                numbers = [float(num) for num in re.findall(r"-?\d+\.?\d*", model_completion)]
                if len(numbers) < 3:
                    continue  # Skip if there are not enough numbers

                # Get the second-to-last and third-to-last numbers as candidates
                corrected_candidates = [numbers[-2], numbers[-3]]

                try:
                    ground_truth_answer = float(answers)
                except ValueError:
                    continue

                # Check if any of the candidate answers matches the ground truth
                if ground_truth_answer in corrected_candidates:
                    corrected += 1

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
