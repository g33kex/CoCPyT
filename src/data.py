"""Data helper functions."""
import datasets

def format_instruction(sample):
    """Format dataset sample for instruction tuning."""
    return f"""<s>user
{sample['conversations'][0]['value']}
assistant
{sample['conversations'][1]['value']}"""

def format_question(question):
    """Format question for inference."""
    return f"""<s>user
{question}
assistant
"""

def load_dataset(cfg):
    """Returns the dataset."""
    return datasets.load_dataset("json", data_files=str(cfg.data.dataset_path), split='train[:100]').train_test_split(test_size=cfg.data.test_size)
