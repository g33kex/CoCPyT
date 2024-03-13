"""Data helper functions."""
import datasets

def format_instruction(sample, tokenizer):
    """Format dataset sample for instruction tuning."""
    return tokenizer.apply_chat_template(sample['conversations'], tokenize=False)

def load_dataset(cfg):
    """Returns the dataset."""
    dataset = datasets.load_dataset("json", data_files=str(cfg.data.dataset_path), split='train')
    return dataset.shuffle().select(range(cfg.data.nb_samples)).train_test_split(test_size=cfg.data.test_size)
