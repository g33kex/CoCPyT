"""Train the model."""
# Using a String-based type annotation in order to avoid importing anything
# outside of this function.
def train(cfg: "Config") -> int:
    """Train the model and return validation score."""
    import logging

    logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(cfg.experiment.name)
    log.info("train() invoked for experiment %s.", cfg.experiment.name)
    log.info("Loading Python modules...")
    if "comet" in cfg:
        import comet_ml
        from comet_ml.integration.pytorch import watch
        from src.callbacks import CometCallback 
    import torch
    import random
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer
    from src.data import load_dataset, format_instruction, format_question
    from src.callbacks import GenerationCallback 
    import warnings

    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
    warnings.filterwarnings("ignore", message="Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")

    log.info("Finished loading Python modules.")

    # Set the seed for reproducibility for numpy, torch and python.random
    torch.manual_seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    hyprparams = {
        **dict(cfg.model),
        **dict(cfg.data),
        **dict(cfg.train),
        "seed": cfg.experiment.seed,
    }

    # Initialize Comet
    comet_experiment = None
    callbacks = []
    if "comet" in cfg:
        comet_experiment = comet_ml.Experiment(
            project_name=cfg.comet.project,
            workspace=cfg.comet.workspace,
        )
        comet_experiment.set_name(cfg.experiment.name)
        comet_experiment.log_parameters(hyprparams)
        comet_experiment.log_code(folder="src")
        callbacks.append(CometCallback(comet_experiment))

    log.info("Starting experiment with config:\n%s", hyprparams)

    # Load dataset
    dataset = load_dataset(cfg)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.model_path),
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )

    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.model_path))
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Generate completions at evaluation step
    callbacks.append(GenerationCallback(tokenizer, model, dataset['test'], comet_experiment))

    model_args = TrainingArguments(
        output_dir=cfg.experiment.checkpoints_path,
        num_train_epochs=cfg.train.n_epochs,
        per_device_train_batch_size=cfg.data.batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=1,
        save_total_limit=3,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=cfg.train.base_lr,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        max_seq_length=1024,
        args=model_args,
        tokenizer=tokenizer,
        formatting_func=format_instruction,
        packing=True,
        callbacks=callbacks,
    )

    # Initial evaluation
    trainer.evaluate()
    
    # Start training
    trainer.train()

    metrics = trainer.evaluate()
    log.info("Done. Validation loss %s", metrics['eval_loss'])

    return metrics['eval_loss']
