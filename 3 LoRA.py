
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
import numpy as np


# ## Task 1: Dataset Preparation [15 minutes]
# Load IMDB dataset
dataset = load_dataset("imdb")

# Split dataset into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Explore dataset structure
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Features: {train_dataset.features}")

# Examine sample entries
print("\nSample entry:")
print(f"Text: {train_dataset[0]['text'][:200]}...")
print(f"Label: {train_dataset[0]['label']} (0=negative, 1=positive)")

# Check label distribution
train_labels = [example['label'] for example in train_dataset]
print(f"\nLabel distribution in training set:")
print(f"Negative (0): {train_labels.count(0)}")
print(f"Positive (1): {train_labels.count(1)}")

# Limit dataset size for faster training (optional)
train_dataset = train_dataset.select(range(5000))
test_dataset = test_dataset.select(range(1000))
print(f"\nUsing {len(train_dataset)} training and {len(test_dataset)} test samples")


# ## Task 2: Model Setup and Initialization [20 minutes]

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,  # Binary classification for sentiment
    ignore_mismatched_sizes=True
)

# Move model to device
model.to(device)

# Test tokenizer with sample text
sample_text = "This movie was amazing!"
tokens = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
print(f"\nSample tokenization:")
print(f"Input IDs shape: {tokens['input_ids'].shape}")
print(f"Attention mask shape: {tokens['attention_mask'].shape}")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print(f"\nModel loaded successfully: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")

# ## Task 3: Implement LoRA and Fine-Tune [40 minutes]
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import numpy as np

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification
    r=8,  # Rank of adaptation
    lora_alpha=16,  # LoRA scaling parameter
    lora_dropout=0.1,  # LoRA dropout
    target_modules=["query", "value"]  # Target attention modules
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=512
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    #evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting LoRA fine-tuning...")
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"\nEvaluation Results: {eval_results}")

# Test prediction on sample text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
    
    sentiment = "positive" if predicted_class.item() == 1 else "negative"
    confidence = predictions[0][predicted_class].item()
    
    return sentiment, confidence

# Test predictions
test_texts = [
    "This movie was absolutely fantastic!",
    "I hated this film, it was terrible.",
    "The movie was okay, nothing special."
]

print("\nSample Predictions:")
for text in test_texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted: {sentiment} (confidence: {confidence:.3f})\n")

