
import mlflow
import mlflow.pyfunc
from transformers import pipeline
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ## Task 1: Wrap the Model with MLflow [20 minutes]
# Load your pre-trained model
model_name = "distilgpt2"
generator = pipeline("text-generation", model=model_name)

print(f"Loaded model: {model_name}")

# Create MLflow model wrapper
class TextGenWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, generator, max_new_tokens=50):
        self.generator = generator
        self.max_new_tokens = max_new_tokens

    def predict(self, context, model_input):
        # Handle pandas Series/DataFrame or list-like input
        if isinstance(model_input, (pd.Series, pd.DataFrame)):
            prompts = model_input.squeeze().tolist()
        else:
            prompts = model_input
        
        # Generate text
        outputs = self.generator(
            prompts,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        
        # Extract generated text from Hugging Face output format
        return [out[0]["generated_text"] for out in outputs]

# Create the model wrapper
pyfunc_model = TextGenWrapper(generator)

# Save model with MLflow
mlflow.pyfunc.save_model(
    path="mlflow_model",
    python_model=pyfunc_model
)

print("Model successfully saved to 'mlflow_model'")


# ## Task 2: Create Streamlit Interface [20 minutes]
# Develop a user-friendly interface for text input and output.
# 1. Create Streamlit components for user input
# 2. Load the MLflow model within the interface
# 3. Handle text generation and display results
# 4. Add interactive elements for better user experience

import streamlit as st

# Set up the Streamlit interface
st.title("🤖 LLM-powered Text Generation")
st.markdown("Generate creative text using our pre-trained language model!")

# User input
user_input = st.text_area(
    "Enter your prompt:",
    placeholder="Once upon a time in a distant galaxy...",
    height=100
)

# Generation settings
col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max new tokens:", 10, 100, 50)
with col2:
    temperature = st.slider("Temperature:", 0.1, 1.0, 0.7)

# Generate button
if st.button("🚀 Generate Text"):
    if user_input.strip():
        with st.spinner("Generating text..."):
            try:
                # Load the MLflow model
                model = mlflow.pyfunc.load_model("mlflow_model")
                
                # Generate response
                result = model.predict([user_input])
                
                # Display result
                st.subheader("📝 Generated Text:")
                st.write(result[0])
                
            except Exception as e:
                st.error(f"Error generating text: {e}")
    else:
        st.warning("⚠️ Please enter a prompt to generate text!")

# Note about warnings
st.info("ℹ️ ScriptRunContext warnings in the console are normal when running Streamlit in notebooks.")

# ## Task 3: Test the Deployment [15 minutes]
# Launch and test your application using Streamlit.
# 1. Use shell command to run Streamlit properly
# 2. Test the text generation functionality
# 3. Verify the complete workflow works
# 4. Document the deployment URLs and access methods

# Launch Streamlit app using shell command
# This will start the Streamlit server and display available URLs

print("🚀 Launching Streamlit application...")
print("The app will display Local URL, Network URL, and possibly External URL")
print("Click on any of these URLs to access your deployed application")
print("\n⚠️ Note: The app will run until you stop the cell execution")