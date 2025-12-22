# # 👩‍💻 **Image Captioning and Classification with VLMs**

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# ## Task 1: Image Captioning with BLIP [20 minutes]
# 1. Load a pre-trained BLIP model using Hugging Face tools.
# 2. Select a diverse set of images from your dataset.
# 3. Generate and analyze captions for each image.

# Load the pre-trained BLIP model and processor
#blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#blip_model.save_pretrained("./model/blip-image-captioning-base")
#blip_processor.save_pretrained("./model/blip-image-captioning-base")
#import sys
#sys.exit()
blip_model = BlipForConditionalGeneration.from_pretrained("./model/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("./model/blip-image-captioning-base")

print("BLIP model loaded successfully!")
print(f"Model has {sum(p.numel() for p in blip_model.parameters())} parameters")

# Load diverse set of images for testing
image_urls = [
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",  # Cat
    "https://images.unsplash.com/photo-1552053831-71594a27632d?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",  # Dog
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",  # Landscape
    "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",  # Food
]

images = []
for url in image_urls:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        images.append(image)
    except:
        print(f"Failed to load image from {url}")

print(f"Successfully loaded {len(images)} images for captioning")

# Generate captions for each image
captions = []
for i, image in enumerate(images):
    # Prepare the image for model input
    inputs = blip_processor(images=image, return_tensors="pt")
    
    # Generate caption
    output = blip_model.generate(**inputs, max_length=50, num_beams=5)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    captions.append(caption)
    
    print(f"Image {i+1} Caption: {caption}")

# Display images with their captions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (image, caption) in enumerate(zip(images, captions)):
    if i < len(axes):
        axes[i].imshow(image)
        axes[i].set_title(f"BLIP: {caption}", fontsize=10, wrap=True)
        axes[i].axis('off')

plt.tight_layout()
plt.show()

# Analyze caption quality
print("\nCaption Analysis:")
print("=" * 30)
for i, caption in enumerate(captions):
    print(f"Image {i+1}: {caption}")
    print(f"  - Length: {len(caption.split())} words")
    print(f"  - Contains descriptive elements: {'Yes' if any(word in caption.lower() for word in ['cat', 'dog', 'mountain', 'food', 'plate', 'sitting', 'lying', 'beautiful']) else 'No'}")

# - BLIP combines vision and language understanding for automatic captioning.
# - Pre-trained models can generate human-like descriptions without fine-tuning.
# - Image diversity is important for evaluating model robustness.


# ## Task 2: Zero-Shot Classification with CLIP [20 minutes]
# Perform zero-shot image classification using CLIP.
# 1. Initialize a pre-trained CLIP model.
# 2. Compile a list of descriptive labels relevant to your images.
# 3. Compute similarity scores and deduce the most likely label.
# 4. Analyze classification accuracy and model confidence.

# Load the CLIP model and processor
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("./model/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("./model/clip-vit-base-patch32")

print("CLIP model loaded successfully!")
print(f"Model has {sum(p.numel() for p in clip_model.parameters())} parameters")

# Define comprehensive label sets for different categories
animal_labels = ["a cat", "a dog", "a bird", "a horse", "a cow", "a sheep"]
general_labels = ["an animal", "a landscape", "food", "a vehicle", "a person", "architecture"]
detailed_labels = [
    "a fluffy cat", "a playful dog", "a mountain landscape", "a delicious meal",
    "a sunny day", "indoor scene", "outdoor scene", "natural environment"
]

# Test each image with different label sets
results = []

for img_idx, image in enumerate(images):
    print(f"\nClassifying Image {img_idx + 1}:")
    print("=" * 30)
    
    # Test with different label sets
    for label_set_name, labels in [("General", general_labels), ("Detailed", detailed_labels)]:
        # Encode text and image
        inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
        
        # Generate predictions
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Get top predictions
        top_probs, top_indices = probs[0].topk(3)
        
        print(f"\n{label_set_name} Labels - Top 3 Predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"  {i+1}. {labels[idx]}: {prob:.3f}")
        
        results.append({
            'image_idx': img_idx,
            'label_set': label_set_name,
            'top_label': labels[top_indices[0]],
            'confidence': top_probs[0].item(),
            'all_probs': probs[0].tolist()
        })

# Visualize classification results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, image in enumerate(images):
    if i < len(axes):
        # Get results for this image
        img_results = [r for r in results if r['image_idx'] == i]
        
        axes[i].imshow(image)
        title = f"Image {i+1}\n"
        for result in img_results:
            title += f"{result['label_set']}: {result['top_label']} ({result['confidence']:.2f})\n"
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')

plt.tight_layout()
plt.show()

# Analyze classification confidence
print("\nClassification Analysis:")
print("=" * 40)
for result in results:
    print(f"Image {result['image_idx']+1} ({result['label_set']}): {result['top_label']} - Confidence: {result['confidence']:.3f}")


# ## Task 3: Comparative Analysis and Evaluation [5 minutes]
# Compare and evaluate the outputs from both models.
# 1. Analyze the strengths and limitations of BLIP captions vs CLIP classifications.
# 2. Evaluate model performance on different types of images.
# 3. Discuss practical applications and use cases for each approach.
# 4. Reflect on how VLMs can be integrated into real-world systems.

# Comprehensive comparison and analysis
print("\n" + "="*60)
print("COMPREHENSIVE VLM ANALYSIS")
print("="*60)

# Compare BLIP captions with CLIP classifications
comparison_data = []
for i in range(len(images)):
    blip_caption = captions[i]
    general_result = next(r for r in results if r['image_idx'] == i and r['label_set'] == 'General')
    detailed_result = next(r for r in results if r['image_idx'] == i and r['label_set'] == 'Detailed')
    
    comparison_data.append({
        'image_idx': i + 1,
        'blip_caption': blip_caption,
        'clip_general': general_result['top_label'],
        'clip_detailed': detailed_result['top_label'],
        'general_confidence': general_result['confidence'],
        'detailed_confidence': detailed_result['confidence']
    })

print("\n🔍 Model Comparison by Image:")
for data in comparison_data:
    print(f"\nImage {data['image_idx']}:")
    print(f"  BLIP Caption: '{data['blip_caption']}'")
    print(f"  CLIP General: '{data['clip_general']}' (conf: {data['general_confidence']:.3f})")
    print(f"  CLIP Detailed: '{data['clip_detailed']}' (conf: {data['detailed_confidence']:.3f})")

# Performance analysis
avg_general_confidence = np.mean([d['general_confidence'] for d in comparison_data])
avg_detailed_confidence = np.mean([d['detailed_confidence'] for d in comparison_data])
avg_caption_length = np.mean([len(d['blip_caption'].split()) for d in comparison_data])

print(f"\n📊 Performance Metrics:")
print(f"  Average CLIP General Confidence: {avg_general_confidence:.3f}")
print(f"  Average CLIP Detailed Confidence: {avg_detailed_confidence:.3f}")
print(f"  Average BLIP Caption Length: {avg_caption_length:.1f} words")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Confidence comparison
x = np.arange(len(comparison_data))
width = 0.35
axes[0, 0].bar(x - width/2, [d['general_confidence'] for d in comparison_data], width, label='General Labels', alpha=0.8)
axes[0, 0].bar(x + width/2, [d['detailed_confidence'] for d in comparison_data], width, label='Detailed Labels', alpha=0.8)
axes[0, 0].set_xlabel('Image Index')
axes[0, 0].set_ylabel('CLIP Confidence')
axes[0, 0].set_title('CLIP Classification Confidence by Label Type')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([f'Img {i+1}' for i in range(len(comparison_data))])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Caption length distribution
caption_lengths = [len(d['blip_caption'].split()) for d in comparison_data]
axes[0, 1].bar(range(len(caption_lengths)), caption_lengths, color='lightcoral', alpha=0.8)
axes[0, 1].set_xlabel('Image Index')
axes[0, 1].set_ylabel('Caption Length (words)')
axes[0, 1].set_title('BLIP Caption Length by Image')
axes[0, 1].set_xticks(range(len(caption_lengths)))
axes[0, 1].set_xticklabels([f'Img {i+1}' for i in range(len(comparison_data))])
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Model agreement analysis
agreement_scores = []
for i, data in enumerate(comparison_data):
    # Simple keyword matching between BLIP caption and CLIP labels
    caption_words = set(data['blip_caption'].lower().split())
    general_words = set(data['clip_general'].lower().split())
    detailed_words = set(data['clip_detailed'].lower().split())
    
    general_overlap = len(caption_words.intersection(general_words)) / max(len(caption_words), 1)
    detailed_overlap = len(caption_words.intersection(detailed_words)) / max(len(caption_words), 1)
    
    agreement_scores.append(max(general_overlap, detailed_overlap))

axes[0, 2].bar(range(len(agreement_scores)), agreement_scores, color='lightgreen', alpha=0.8)
axes[0, 2].set_xlabel('Image Index')
axes[0, 2].set_ylabel('Agreement Score')
axes[0, 2].set_title('BLIP-CLIP Agreement (Keyword Overlap)')
axes[0, 2].set_xticks(range(len(agreement_scores)))
axes[0, 2].set_xticklabels([f'Img {i+1}' for i in range(len(comparison_data))])
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Model strengths comparison
strengths_data = {
    'BLIP Strengths': ['Detailed descriptions', 'Natural language', 'Contextual understanding', 'Creative captions'],
    'CLIP Strengths': ['Zero-shot classification', 'Fast inference', 'Flexible labels', 'Reliable confidence']
}

axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.9, 'Model Strengths Comparison', ha='center', va='top', fontsize=14, fontweight='bold')

y_pos = 0.7
for model, strengths in strengths_data.items():
    axes[1, 0].text(0.05 if 'BLIP' in model else 0.55, y_pos, model, fontsize=12, fontweight='bold')
    for i, strength in enumerate(strengths):
        axes[1, 0].text(0.05 if 'BLIP' in model else 0.55, y_pos - 0.1*(i+1), f"• {strength}", fontsize=10)
    y_pos -= 0.5

# Plot 5: Application scenarios
axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.9, 'Practical Applications', ha='center', va='top', fontsize=14, fontweight='bold')

applications = [
    'Content Moderation: CLIP for quick classification',
    'Accessibility: BLIP for detailed descriptions',
    'E-commerce: Both for product categorization and descriptions',
    'Social Media: BLIP for alt-text, CLIP for content filtering',
    'Digital Libraries: Combined for comprehensive metadata'
]

for i, app in enumerate(applications):
    axes[1, 1].text(0.05, 0.8 - i*0.15, f"• {app}", fontsize=10, wrap=True)

# Plot 6: Performance summary
metrics = ['Caption Quality', 'Classification Accuracy', 'Speed', 'Flexibility', 'Detail Level']
blip_scores = [0.85, 0.70, 0.60, 0.75, 0.90]  # Example scores
clip_scores = [0.60, 0.85, 0.90, 0.85, 0.65]  # Example scores

x = np.arange(len(metrics))
width = 0.35

axes[1, 2].bar(x - width/2, blip_scores, width, label='BLIP', alpha=0.8, color='skyblue')
axes[1, 2].bar(x + width/2, clip_scores, width, label='CLIP', alpha=0.8, color='lightcoral')
axes[1, 2].set_xlabel('Capabilities')
axes[1, 2].set_ylabel('Performance Score')
axes[1, 2].set_title('BLIP vs CLIP Performance Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(metrics, rotation=45, ha='right')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final insights and recommendations
print("\n🎯 Key Insights and Recommendations:")
print("\n1. Model Complementarity:")
print("   • BLIP excels at generating rich, descriptive captions")
print("   • CLIP provides fast, accurate zero-shot classification")
print("   • Combined use maximizes information extraction from images")

print("\n2. Practical Deployment Considerations:")
print("   • BLIP: Best for accessibility, content generation, detailed analysis")
print("   • CLIP: Ideal for content filtering, quick categorization, search")
print("   • Label design significantly impacts CLIP performance")

print("\n3. Integration Strategies:")
print("   • Use CLIP for initial categorization, BLIP for detailed descriptions")
print("   • Combine outputs for comprehensive metadata generation")
print("   • Consider computational resources when choosing between models")

print("\n4. Future Improvements:")
print("   • Fine-tune models on domain-specific data")
print("   • Implement confidence thresholding for quality control")
print("   • Explore newer VLM architectures for enhanced performance")# Task 3


