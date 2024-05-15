from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments


def finetune(model_path, dataset_path, epochs, batch_size, lr_rate, quant):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False, 
        loftq_config = None, 
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    Instruction:
    {}

    Input:
    {}

    Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token 
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    from datasets import load_dataset
    dataset = load_dataset(dataset_path)
    dataset = dataset.map(formatting_prompts_func, batched = True,)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = True,
        args = TrainingArguments(
            batch_size = batch_size,
            gradient_accumulation_steps = 4,
            num_train_epochs = epochs,
            learning_rate = lr_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer_stats = trainer.train()
    model.save_pretrained("./outputs")
    tokenizer.save_pretrained("./outputs")
    if(quant=="Q4"):
        model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if(quant=="Q8"):
        model.save_pretrained_gguf("model", tokenizer)

if __name__ == '__main__':
    finetune()