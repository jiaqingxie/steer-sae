import re
def post_inspection_correction(text):
    """
    Post-inspection function to extract the second last valid numeric result from the text.
    Handles integers, floats, fractions, and currency values.
    """
    # Find all valid numbers, including integers, floats, and currency-like values
    matches = re.findall(r"(-?\d+/\d+|-?\d+\.?\d*|\$\d+\.?\d*)", text)

    if len(matches) < 2:
        return "INVALID"  # Less than two valid numbers found, no correction possible

    processed_values = []
    for match in matches:
        if "/" in match:  # Fraction format, convert to float
            numerator, denominator = map(float, match.split("/"))
            value = numerator / denominator
        elif "$" in match:  # Currency format, strip $ and convert to float
            value = float(match.replace("$", ""))
        else:  # Integer or float
            value = float(match) if "." in match else int(match)

        processed_values.append(value)

    # Return the second last value in the list
    return processed_values[-2]

