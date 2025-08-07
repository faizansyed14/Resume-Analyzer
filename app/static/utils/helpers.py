# app/utils/helpers.py
import os

# This file is for general utility functions that might be used across different
# parts of your application but don't belong to a specific service.
# For example:
# - Custom data validation helpers
# - Common formatting functions
# - Helper functions for file operations that are not specific to resumes/jobs parsing

def calculate_average(numbers: list) -> float:
    """Calculates the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

# Example of a generic file deletion helper (already done in routes, but could be here)
def delete_file_if_exists(filepath: str):
    """Deletes a file if it exists, suppresses errors."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error deleting file {filepath}: {e}")


