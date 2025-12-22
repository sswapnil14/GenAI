import torch
from transformers import pipeline
import json
import re


# ## Task 1: Setup and Basic JSON Generation [15 minutes]
# 1. Setup a free local LLM and create helper functions for JSON parsing.
# 2. Test basic structured output generation.

# Setup free local LLM
model_name = "gpt2"
generator = pipeline('text-generation', 
                    model=model_name, 
                    max_length=300,
                    pad_token_id=50256)

def clean_and_parse_json(text):
    """Helper function to extract and clean JSON from model output"""
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON", "raw": json_str}
    return {"error": "No JSON found", "raw": text}

print("✅ Model and helper functions ready")

# Test basic JSON generation
test_prompt = """Create a JSON object with name and age:
{
  "name": "Alice",
  "age": 25
}

Create similar JSON for Bob, age 30:
{"""

response = generator(test_prompt, max_length=50, temperature=0.2, do_sample=True)
generated = response[0]['generated_text'][len(test_prompt):]
result = clean_and_parse_json("{" + generated)
print("Test result:", result)


# ## Task 2: Travel Itinerary JSON Generation [15 minutes]
# Create prompts that generate well-structured JSON for travel itineraries.
# 1. Design a prompt with exact JSON structure examples.
# 2. Generate travel itineraries for multiple destinations.
# 3. Validate the consistency of generated JSON.

def generate_travel_itinerary(destination, days=3):
    """Generate structured travel itinerary JSON"""
    
    prompt = f"""Create a travel itinerary in JSON format for {destination}. Follow this exact structure:

{{
  "destination": "{destination}",
  "duration_days": {days},
  "departure_date": "2024-06-15",
  "activities": [
    "Visit landmarks",
    "Try local cuisine"
  ],
  "accommodations": "Hotel",
  "budget_estimate": "$500-1000"
}}

Generate similar JSON for {destination}:
{{"""
    
    try:
        response = generator(prompt, 
                           max_length=len(prompt.split()) + 100,
                           temperature=0.3,
                           do_sample=True,
                           pad_token_id=50256)
        
        generated_text = response[0]['generated_text']
        generated_part = generated_text[len(prompt):]
        json_content = "{" + generated_part
        
        return clean_and_parse_json(json_content)
        
    except Exception as e:
        return {"error": str(e), "destination": destination}

# Test with multiple destinations
destinations = ["Paris", "Tokyo", "New York"]

for dest in destinations:
    print(f"\n=== {dest.upper()} ITINERARY ===")
    itinerary = generate_travel_itinerary(dest)
    print(json.dumps(itinerary, indent=2))


# ## Task 3: Function Simulation - Order Calculations [15 minutes]
# Simulate mathematical function logic using natural language prompts.
# 1. Create step-by-step calculation prompts.
# 2. Test order total calculations with tax and shipping.
# 3. Extract and verify numerical results from LLM outputs.

def simulate_order_calculation(items_list):
    """Simulate order total calculation using LLM"""
    
    items_text = "\n".join([f"- {item['name']}: ${item['price']} x {item['quantity']}" 
                           for item in items_list])
    
    prompt = f"""Calculate the total cost step by step:

Items:
{items_text}

Example:
- Laptop: $800 x 2 = $1600
- Mouse: $25 x 1 = $25
Subtotal: $1600 + $25 = $1625
Tax (8%): $1625 x 0.08 = $130
Total: $1625 + $130 = $1755

Calculate for items above:
Calculation:"""
    
    try:
        response = generator(prompt,
                           max_length=len(prompt.split()) + 80,
                           temperature=0.1,
                           do_sample=True,
                           pad_token_id=50256)
        
        generated_text = response[0]['generated_text']
        calculation = generated_text[len(prompt):].strip()
        
        # Extract total from calculation
        total_match = re.search(r'Total:\s*\$?([0-9,]+\.?[0-9]*)', calculation, re.IGNORECASE)
        extracted_total = total_match.group(1) if total_match else "Not found"
        
        return {
            "calculation_steps": calculation,
            "extracted_total": extracted_total,
            "items": items_list
        }
        
    except Exception as e:
        return {"error": str(e), "items": items_list}

# Test order calculation
test_order = [
    {"name": "Coffee", "price": 5.99, "quantity": 3},
    {"name": "Sandwich", "price": 8.50, "quantity": 2}
]

print("=== ORDER CALCULATION SIMULATION ===")
result = simulate_order_calculation(test_order)

if "error" not in result:
    print("Items:")
    for item in result["items"]:
        print(f"  {item['name']}: ${item['price']} x {item['quantity']}")
    print(f"\nLLM Calculation:\n{result['calculation_steps']}")
    print(f"\nExtracted Total: ${result['extracted_total']}")
    
    # Verify with actual calculation
    actual_total = sum(item['price'] * item['quantity'] for item in test_order)
    tax = actual_total * 0.08
    final_total = actual_total + tax
    print(f"Actual Total (with 8% tax): ${final_total:.2f}")
else:
    print(f"Error: {result['error']}")

# Enhanced function simulation example
def simulate_route_decision(routes_data):
    """Simulate route optimization decision-making"""
    
    routes_text = "\n".join([
        f"Route {i+1}: {route['name']} - Distance: {route['distance']}km, "
        f"Traffic: {route['traffic']}, Time: {route['time']}min, Cost: ${route['cost']}"
        for i, route in enumerate(routes_data)
    ])
    
    prompt = f"""Choose the optimal route based on distance, traffic, time, and cost.

Available Routes:
{routes_text}

Analysis:
1. Distance comparison:
2. Traffic impact:
3. Time efficiency:
4. Cost consideration:
5. Overall recommendation:

Decision:"""
    
    try:
        response = generator(prompt,
                           max_length=len(prompt.split()) + 100,
                           temperature=0.3,
                           do_sample=True,
                           pad_token_id=50256)
        
        generated_text = response[0]['generated_text']
        decision = generated_text[len(prompt):].strip()
        
        # Extract recommended route
        route_match = re.search(r'Route\s+(\d+)', decision, re.IGNORECASE)
        recommended_route = route_match.group(1) if route_match else "Not specified"
        
        return {
            "analysis": decision,
            "recommended_route": recommended_route,
            "routes": routes_data
        }
        
    except Exception as e:
        return {"error": str(e), "routes": routes_data}

# Test route optimization
sample_routes = [
    {"name": "Highway Route", "distance": 45, "traffic": "Light", "time": 35, "cost": 12.50},
    {"name": "City Route", "distance": 32, "traffic": "Heavy", "time": 55, "cost": 8.75}
]

print("\n=== ROUTE OPTIMIZATION SIMULATION ===")
route_result = simulate_route_decision(sample_routes)

if "error" not in route_result:
    print("Available Routes:")
    for i, route in enumerate(route_result["routes"], 1):
        print(f"  Route {i}: {route['name']} - {route['distance']}km, {route['time']}min, ${route['cost']}")
    
    print(f"\nLLM Analysis:\n{route_result['analysis'][:200]}...")
    print(f"\nRecommended Route: {route_result['recommended_route']}")
else:
    print(f"Error: {route_result['error']}")

