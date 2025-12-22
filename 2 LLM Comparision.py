import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import requests
import json


# Setup local models - these are free and don't require API keys
# We'll use smaller models that can run locally

# Model 1: DistilGPT-2 (lightweight GPT-2 variant)
model1_name = "distilgpt2"
generator1 = pipeline('text-generation', model=model1_name, max_length=150)

# Model 2: GPT-2 small
model2_name = "gpt2"
generator2 = pipeline('text-generation', model=model2_name, max_length=150)

print("✅ Local models loaded successfully!")
print(f"Model 1: {model1_name}")
print(f"Model 2: {model2_name}")

# Optional: Setup access to free online APIs (no authentication required)
# Hugging Face Inference API (free tier available)

def query_hf_api(model_id, prompt, max_tokens=100):
    """Query Hugging Face's free inference API"""
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}  # Optional: add your free HF token
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return f"Error: {e}"

# Test function (will work even without token for many models)
print("🌐 Online API function ready (optional)")

# your code here to test both models with a simple prompt
prompt = "The future of technology involves"
resp1 = generator1(prompt)#, max_length=len(prompt.split()) + 50, temparature=0.7, do_sample=True, pad_token_id=50256)
resp2 = generator2(prompt)#, max_length=len(prompt.split()) + 50, temparature=0.7, do_sample=True, pad_token_id=50256)
#resp3 = query_hf_api('distilgpt2', prompt)
#resp4 = query_hf_api('distilgpt2', prompt)
print(f"{resp1=}")
print(f"{resp2=}")
#print(f"{resp3=}")
#print(f"{resp4=}")


# ## Task 2: Baseline Prompt Evaluation [15 minutes]

# Define baseline prompt
baseline_prompt = "Explain the concept of sustainability in simple terms."

def compare_models(prompt, temperature=0.7, max_tokens=100):
    """Compare responses from multiple models"""
    results = {}
    
    # Model 1 response
    try:
        response1 = generator1(prompt, max_length=len(prompt.split()) + max_tokens, 
                              temperature=temperature, do_sample=True, pad_token_id=50256)
        results['DistilGPT-2'] = response1[0]['generated_text'][len(prompt):].strip()
    except Exception as e:
        results['DistilGPT-2'] = f"Error: {e}"
    
    # Model 2 response
    try:
        response2 = generator2(prompt, max_length=len(prompt.split()) + max_tokens, 
                              temperature=temperature, do_sample=True, pad_token_id=50256)
        results['GPT-2'] = response2[0]['generated_text'][len(prompt):].strip()
    except Exception as e:
        results['GPT-2'] = f"Error: {e}"
    
    return results

# Test the function
print("🔍 Testing baseline prompt comparison...")
baseline_results = compare_models(baseline_prompt)

for model, response in baseline_results.items():
    print(f"\n{model}:")
    print(f"{response}")
    print("-" * 50)


# Test your prompt and compare outputs
my_baseline_prompt = "Explain donald trumps view on global warming"  # Add your prompt here
res = compare_models(my_baseline_prompt)
for model, response in res.items():
    print(f"\n{model}:")
    print(f"{response}")
    print("-" * 50)


# ## Task 3: Prompt Variation and Comparison [15 minutes]
# Different prompting techniques
prompts = {
    "Basic": "Explain sustainability.",
    "Context": "As an environmental expert, explain sustainability to a 10-year-old.",
    "Few-shot": """Q: What is recycling?
A: Recycling is reusing materials to make new products instead of throwing them away.

Q: What is sustainability?
A:""",
    "Detailed": "Provide a comprehensive explanation of sustainability that covers environmental, economic, and social aspects."
}

# Test all prompt variations
all_results = {}
for prompt_type, prompt in prompts.items():
    print(f"\n{'='*20} {prompt_type.upper()} PROMPT {'='*20}")
    print(f"Prompt: {prompt}")
    print("\nResponses:")
    
    results = compare_models(prompt, temperature=0.7)
    all_results[prompt_type] = results
    
    for model, response in results.items():
        print(f"\n{model}:")
        print(response)
        print("-" * 30)


# Test your prompt and compare outputs
#my_prompt_variation = ""  # Add your creative prompt here


# ## Task 4: Creativity vs. Precision Tuning [15 minutes]

# Test different temperature settings
creative_prompt = "Write a short story about a world where plants can talk."
factual_prompt = "List the top 5 renewable energy sources and their efficiency ratings."

temperatures = [0.1, 0.5, 0.9]

def test_temperature_effects(prompt, description):
    print(f"\n{'='*15} {description} {'='*15}")
    print(f"Prompt: {prompt}")
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        results = compare_models(prompt, temperature=temp, max_tokens=80)
        
        for model, response in results.items():
            print(f"{model}: {response[:100]}...")

# Test creative task
test_temperature_effects(creative_prompt, "CREATIVE TASK")

# Test factual task
test_temperature_effects(factual_prompt, "FACTUAL TASK")


# Test both with different temperature settings
#my_creative_prompt = ""  # Add your creative prompt
#my_factual_prompt = ""   # Add your factual prompt
