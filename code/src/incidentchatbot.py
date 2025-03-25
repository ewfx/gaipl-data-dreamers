import os
import torch
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.prompts import PromptTemplate
#from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Load the pre-trained incident classification model
def load_incident_model(model_path='./incident_management_model'):
    """
    Load the pre-trained incident classification model
    """
    try:
        # Ensure absolute path
        full_model_path = os.path.abspath(model_path)
        
        # Check if model directory exists
        if not os.path.exists(full_model_path):
            st.error(f"Model directory not found: {full_model_path}")
            return None, None, None

        # Load tokenizer and model from local directory
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        
        # Load label encoder
        df = pd.read_csv('C:/Users/srikr/OneDrive/Desktop/Hackathon/Incident_Management.csv')
        label_encoder = LabelEncoder()
        label_encoder.fit(df['Status'])
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Fallback model loading method
def load_model_fallback(model_name='bert-base-uncased'):
    """
    Fallback method to load a base model if local model fails
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(set(pd.read_csv('C:/Users/srikr/OneDrive/Desktop/Hackathon/Incident_Management.csv')['Status']))
        )
        
        df = pd.read_csv('C:/Users/srikr/OneDrive/Desktop/Hackathon/Incident_Management.csv')
        label_encoder = LabelEncoder()
        label_encoder.fit(df['Status'])
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Fallback model loading failed: {e}")
        return None, None, None

# Predict incident status
def predict_incident_status(model, tokenizer, incident_text, label_encoder):
    """
    Predict incident status using the trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        incident_text, 
        return_tensors='pt', 
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1)

    # Decode the predicted label
    predicted_status = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
    return predicted_status

# Create LangChain conversational chain
def create_incident_chain():
    """
    Create a conversational chain for incident management
    """
    # Use a pre-trained model for text generation
    text_generator = pipeline(
        "text-generation", 
        model="gpt2", 
        max_length=200
    )
    
    # Create HuggingFace LLM
    llm = HuggingFacePipeline(pipeline=text_generator)

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["incident_text", "status"],
        template="""
        You are an AI incident management assistant. 
        
        Incident Description: {incident_text}
        Predicted Incident Status: {status}
        
        Provide a helpful and professional response addressing the incident. 
        Include:
        - A brief summary of the incident
        - Potential next steps
        - Recommendations for resolution
        
        Response:"""
    )

    # Create LLM chain
    return LLMChain(llm=llm, prompt=prompt_template)

# Streamlit Application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Incident Management Chatbot", 
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and Description
    st.title("🤖 Incident Management Chatbot")
    st.write("An AI-powered assistant for managing and analyzing IT incidents.")

    # Attempt to load local model, fallback to base model if needed
    model, tokenizer, label_encoder = load_incident_model()
    
    # If local model fails, try fallback
    if model is None:
        st.warning("Couldn't load local model. Using fallback model.")
        model, tokenizer, label_encoder = load_model_fallback()

    # Verify model is loaded
    if model is None:
        st.error("Failed to load any model. Please check your setup.")
        return

    # Create incident chain
    incident_chain = create_incident_chain()

    # Sidebar for additional information
    st.sidebar.header("About")
    st.sidebar.info(
        "This chatbot uses a transformer-based model to classify "
        "incident statuses and provide intelligent responses."
    )

    # Chat input
    user_input = st.text_input("Describe your IT incident:", key="incident_input")

    # Process incident when user submits
    if user_input:
        try:
            # Predict incident status
            predicted_status = predict_incident_status(
                model, tokenizer, user_input, label_encoder
            )

            # Generate detailed response using LangChain
            response = incident_chain.run(
                incident_text=user_input, 
                status=predicted_status
            )

            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Incident Analysis")
                st.write(f"**Predicted Status:** {predicted_status}")

            with col2:
                st.subheader("Recommended Actions")
                st.write(response)

        except Exception as e:
            st.error(f"Error processing incident: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by Transformers and LangChain*")

# Run the Streamlit app
if __name__ == "__main__":
    main()