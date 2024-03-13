from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import torch
class GenerationCallback(TrainerCallback):
    """A custom callback that generates text completions at the end of each evaluation phase."""
    def __init__(self, tokenizer, model, eval_dataset, comet_experiment=None, num_samples=5, max_new_tokens=200):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = eval_dataset.shuffle().select(range(num_samples))
        self.max_new_tokens = max_new_tokens
        self.comet_experiment = comet_experiment

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of the evaluation phase."""
        device = self.model.device  # Use the same device as the model
        self.model.eval()  # Ensure the model is in evaluation mode

        for sample in self.eval_dataset:
            if sample['conversations'][0]['from'] == 'system':
                conversations = sample['conversations'][0:2]
            else:
                conversations = [sample['conversations'][0]]
            prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            completion = self.generate_completion(prompt, device)
            print(f"Completion: {completion}\n")
            if self.comet_experiment is not None:
                self.comet_experiment.log_text(completion, step=state.global_step)


    def generate_completion(self, prompt, device):
        """Generates a completion for a given prompt."""
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

class CometCallback(TrainerCallback):
    """Custom comet callback."""

    def __init__(self, comet_experiment):
        self.comet_experiment = comet_experiment

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, logs=None, **kwargs):
        if state.is_world_process_zero:
            self.comet_experiment.log_metrics(logs, step=state.global_step, epoch=state.epoch)
