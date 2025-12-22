from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ## Task 1: Evaluate Different Prompts [20 minutes]
# Experiment with a range of text prompts to understand their impact on image creation.

# Set up Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#                                                safety_checker=None,
#                                                requires_safety_checker=False)
#SSHAH--
model_path = './model/stable_diffusion_pipeline'
#pipe.save_pretrained(model_path)
pipe = StableDiffusionPipeline.from_pretrained(model_path, dtype=torch.float16 if device == "cuda" else torch.float32,
                                                safety_checker=None,
                                                requires_safety_checker=False)
#SSHAH--

pipe = pipe.to(device)
print(f"Pipeline loaded on {device}")

# Define different prompt styles for testing
prompts = {
    "simple": "A bustling market at dawn with vibrant stalls",
    "abstract": "A whimsical landscape with floating islands and waterfalls",
    "detailed": "A majestic castle on a cliff overlooking a stormy ocean, painted in the style of romantic realism, golden hour lighting, dramatic clouds",
    "artistic": "An ethereal forest spirit dancing among luminescent mushrooms, digital art, fantasy style, soft glowing effects",
    "minimalist": "A single red rose on white background"
}

# Generate images for each prompt style
generated_images = {}
for style, prompt in prompts.items():
    print(f"Generating image for {style} prompt: '{prompt}'")
    
    image = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=torch.manual_seed(42)
    ).images[0]
    
    generated_images[style] = image
    image.save("img/" + style + ".png")
    
import sys
sys.exit()

# Display the results
fig, axes = plt.subplots(1, len(prompts), figsize=(20, 4))
for i, (style, image) in enumerate(generated_images.items()):
    axes[i].imshow(image)
    axes[i].set_title(f'{style.capitalize()} Prompt', fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Analysis
print("\n📊 Prompt Analysis:")
print("• Simple prompts produce clear, focused images")
print("• Detailed prompts allow for more artistic control")
print("• Abstract prompts encourage creative interpretation")
print("• Artistic style prompts benefit from specific technique mentions")


# ## Task 2: Parameter Exploration [25 minutes]
# Adjust key generation parameters to control image output styles.
# 1. Experiment with Guidance Scale for prompt adherence vs creative freedom
# 2. Test different Seeds to produce varied images from the same prompt
# 3. Adjust Steps to balance detail quality with generation time
# 4. Document parameter effects on image characteristics

# Select a base prompt for parameter exploration
base_prompt = "A serene mountain lake at sunset with misty forests"

# Test different guidance scales
guidance_scales = [5, 7.5, 10, 15, 20]
guidance_images = []

print("🎛️ Testing Guidance Scale Effects:")
for scale in guidance_scales:
    image = pipe(
        base_prompt,
        num_inference_steps=20,
        guidance_scale=scale,
        generator=torch.manual_seed(42)
    ).images[0]
    guidance_images.append(image)
    print(f"Generated image with guidance scale: {scale}")

# Display guidance scale comparison
fig, axes = plt.subplots(1, len(guidance_scales), figsize=(20, 4))
for i, (scale, image) in enumerate(zip(guidance_scales, guidance_images)):
    axes[i].imshow(image)
    axes[i].set_title(f'Guidance: {scale}', fontsize=10)
    axes[i].axis('off')
plt.suptitle('Effect of Guidance Scale on Image Generation')
plt.tight_layout()
plt.show()

# Test different seeds
seeds = [42, 123, 456, 789, 999]
seed_images = []

print("\n🎲 Testing Seed Variations:")
for seed in seeds:
    image = pipe(
        base_prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=torch.manual_seed(seed)
    ).images[0]
    seed_images.append(image)
    print(f"Generated image with seed: {seed}")

# Display seed comparison
fig, axes = plt.subplots(1, len(seeds), figsize=(20, 4))
for i, (seed, image) in enumerate(zip(seeds, seed_images)):
    axes[i].imshow(image)
    axes[i].set_title(f'Seed: {seed}', fontsize=10)
    axes[i].axis('off')
plt.suptitle('Effect of Different Seeds on Image Generation')
plt.tight_layout()
plt.show()

# Test different step counts
step_counts = [10, 20, 30, 50]
step_images = []

print("\n⏱️ Testing Inference Steps:")
for steps in step_counts:
    image = pipe(
        base_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=torch.manual_seed(42)
    ).images[0]
    step_images.append(image)
    print(f"Generated image with {steps} steps")

# Display steps comparison
fig, axes = plt.subplots(1, len(step_counts), figsize=(16, 4))
for i, (steps, image) in enumerate(zip(step_counts, step_images)):
    axes[i].imshow(image)
    axes[i].set_title(f'{steps} Steps', fontsize=10)
    axes[i].axis('off')
plt.suptitle('Effect of Inference Steps on Image Quality')
plt.tight_layout()
plt.show()

# Parameter analysis
print("\n📈 Parameter Analysis:")
print("• Guidance Scale 5-10: More creative, diverse outputs")
print("• Guidance Scale 15-20: Stricter prompt adherence, less variation")
print("• Different seeds: Significant composition and detail variations")
print("• More steps: Generally improved detail and coherence")

# ## Task 3: Comparative Analysis [15 minutes]
# Create a portfolio of images with varied prompts and parameters.
# 1. Generate a diverse set of images using different prompt-parameter combinations
# 2. Arrange outputs in a visual grid for comparison
# 3. Analyze stylistic differences and quality variations
# 4. Identify optimal combinations for specific creative goals

# Create a comprehensive portfolio with varied combinations
portfolio_configs = [
    {"prompt": "A cyberpunk cityscape at night", "guidance": 7.5, "steps": 20, "seed": 42},
    {"prompt": "A cyberpunk cityscape at night", "guidance": 15, "steps": 20, "seed": 42},
    {"prompt": "A peaceful zen garden with cherry blossoms", "guidance": 7.5, "steps": 30, "seed": 123},
    {"prompt": "A peaceful zen garden with cherry blossoms", "guidance": 7.5, "steps": 30, "seed": 456},
    {"prompt": "Abstract geometric patterns in vibrant colors", "guidance": 10, "steps": 25, "seed": 789},
    {"prompt": "A majestic dragon flying over mountains", "guidance": 12, "steps": 35, "seed": 999}
]

# Generate portfolio images
portfolio_images = []
portfolio_metadata = []

print("🎨 Generating Portfolio Images:")
for i, config in enumerate(portfolio_configs):
    print(f"Image {i+1}/6: {config['prompt'][:30]}...")
    
    image = pipe(
        config['prompt'],
        num_inference_steps=config['steps'],
        guidance_scale=config['guidance'],
        generator=torch.manual_seed(config['seed'])
    ).images[0]
    
    portfolio_images.append(image)
    portfolio_metadata.append(config)

# Create comprehensive portfolio display
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (image, config) in enumerate(zip(portfolio_images, portfolio_metadata)):
    axes[i].imshow(image)
    title = f"{config['prompt'][:25]}...\nG:{config['guidance']} S:{config['steps']} Seed:{config['seed']}"
    axes[i].set_title(title, fontsize=9)
    axes[i].axis('off')

plt.suptitle('Generated Image Portfolio - Parameter Exploration', fontsize=16)
plt.tight_layout()
plt.show()

# Detailed analysis function
def analyze_image_characteristics(images, configs):
    """Analyze the characteristics of generated images"""
    print("\n🔍 Portfolio Analysis:")
    print("\n1. Prompt Adherence:")
    for i, config in enumerate(configs):
        adherence = "High" if config['guidance'] >= 12 else "Medium" if config['guidance'] >= 8 else "Low"
        print(f"   Image {i+1}: {adherence} adherence (guidance: {config['guidance']})")
    
    print("\n2. Detail Quality:")
    for i, config in enumerate(configs):
        quality = "High" if config['steps'] >= 30 else "Medium" if config['steps'] >= 20 else "Low"
        print(f"   Image {i+1}: {quality} detail (steps: {config['steps']})")
    
    print("\n3. Creative Diversity:")
    prompt_groups = {}
    for i, config in enumerate(configs):
        base_prompt = config['prompt'][:30]
        if base_prompt in prompt_groups:
            prompt_groups[base_prompt].append(i+1)
        else:
            prompt_groups[base_prompt] = [i+1]
    
    for prompt, image_nums in prompt_groups.items():
        if len(image_nums) > 1:
            print(f"   Images {image_nums}: Variations of '{prompt}...'")

# Run analysis
analyze_image_characteristics(portfolio_images, portfolio_metadata)

# Save portfolio images
print("\n💾 Saving Portfolio Images:")
for i, (image, config) in enumerate(zip(portfolio_images, portfolio_metadata)):
    filename = f"portfolio_image_{i+1}_g{config['guidance']}_s{config['steps']}_seed{config['seed']}.png"
    image.save("img/" + filename)
    print(f"Saved: {filename}")

# Final recommendations
print("\n🎯 Recommendations for Creative Work:")
print("• For precise brand imagery: Use guidance scale 12-15")
print("• For creative exploration: Use guidance scale 5-8")
print("• For high detail work: Use 30+ inference steps")
print("• For rapid prototyping: Use 15-20 inference steps")
print("• Always test multiple seeds for prompt variations")

