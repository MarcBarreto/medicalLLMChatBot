# MedicalLLMChatBot

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Files in the Repository](#2-files-in-the-repository)  
3. [Model and Dataset Details](#3-model-and-dataset-details)  
4. [Techniques and Libraries Used](#4-techniques-and-libraries-used)  
5. [Code Overview](#5-code-overview)  
6. [License](#6-license)  

---

## 1. Introduction

**MedicalLLMChatBot** is a conversational chatbot designed to assist in medical contexts using advanced Large Language Models (LLMs).  
The project is developed in Jupyter Notebook and leverages fine-tuned models to provide accurate and context-aware medical responses.  

This chatbot:  
- Uses the **NousResearch/Llama-2-7b-chat-hf** model as the base LLM.  
- Fine-tunes the model with the **Llama2-MedTuned-Instructions** dataset.  

---

## 2. Files in the Repository

- **`medicalLLMChatBot.ipynb`**:  
  The main Jupyter Notebook containing the implementation of the chatbot, including fine-tuning, quantization, and the conversational structure.  

---

## 3. Model and Dataset Details

### Model
- Base Model: **[NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)**  
- Quantization and fine-tuning techniques are applied to optimize the model for performance and resource efficiency.  

### Dataset
- Dataset used for fine-tuning: **[Llama2-MedTuned-Instructions](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions)**  
- The dataset contains instructions tailored for medical contexts, enhancing the chatbot’s ability to handle domain-specific queries.  

---

## 4. Techniques and Libraries Used

### Techniques
1. **QLoRA**:  
   - Quantized LoRA (Low-Rank Adaptation) is used to reduce model size and computational requirements while maintaining performance.  
   - Key configurations:  
     ```python
     use_4bit = True
     bnb_4bit_quant_type = 'nf4'
     bnb_4bit_compute_dtype = 'float16'
     use_double_quant = False
     ```
2. **BitsAndBytes**:  
   - Implements 4-bit quantization to optimize memory and computational efficiency.  

3. **Fine-Tuning**:  
   - LoRA fine-tuning is applied using the following configuration:  
     ```python
     peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
     ```

### Libraries
- **Hugging Face Transformers**:  
  For model architecture, tokenization, and pipelines.  
- **LangChain**:  
  Used for structuring the conversational framework.  
- **bitsandbytes**:  
  Provides efficient quantization methods.  

---

## 5. Code Overview

### Key Snippets

#### Quantization Configuration
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_double_quant
)
```
#### Fine-Tuning and Model Preparation
```python
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
```
#### Text Generation Pipeline
```python
pipe = pipeline(
    task='text-generation',
    model=merged_model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    use_cache=False,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    top_p=0.7,
    temperature=0.4
)
llm_pipeline = HuggingFacePipeline(pipeline=pipe)
```
## 6. License
This project is licensed under the [MIT License](LICENSE).
Feel free to use, modify, and distribute the project as needed.
