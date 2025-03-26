import os
import torch
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

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
            return None, None, None, None

        # Load tokenizer and model from local directory
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        
        # Load label encoder and dataframe
        df = pd.read_csv('C:/Users/srikr/OneDrive/Desktop/Hackathon/Incident_Management.csv')
        label_encoder = LabelEncoder()
        label_encoder.fit(df['Status'])
        
        return model, tokenizer, label_encoder, df
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def load_model_fallback(model_name='bert-base-uncased'):
    """
    Fallback method to load a base model if local model fails
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        df = pd.read_csv('C:/Users/srikr/OneDrive/Desktop/Hackathon/Incident_Management.csv')
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(set(df['Status']))
        )
        
        label_encoder = LabelEncoder()
        label_encoder.fit(df['Status'])
        
        return model, tokenizer, label_encoder, df
    except Exception as e:
        st.error(f"Fallback model loading failed: {e}")
        return None, None, None, None

def predict_incident_details(model, tokenizer, incident_text, label_encoder, df):
    """
    Predict incident status and find matching incident details
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
    
    # Find the most similar incident in the dataset
    df['similarity_score'] = df.apply(
        lambda row: len(set(incident_text.lower().split()) & set(row['Issue Summary'].lower().split())), 
        axis=1
    )
    
    # Get the most similar incident details
    most_similar_incident = df.loc[df['similarity_score'].idxmax()]
    
    return {
        'predicted_status': predicted_status,
        'resolution_summary': most_similar_incident.get('Resolution Summary', 'No resolution summary available'),
        'troubleshooting_link': most_similar_incident.get('Troubleshooting Doc Link', 'No troubleshooting link available')
    }

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
        
        Incident Description: {incident_text}
        Predicted Incident Status: {status}
                
        
        Response:"""
    )

    # Create LLM chain
    return LLMChain(llm=llm, prompt=prompt_template)

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
    .highlight-box {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and Description
    st.title("🤖 Incident Management Chatbot")
    st.write("An AI-powered assistant for managing and analyzing IT incidents.")

    # Attempt to load local model, fallback to base model if needed
    model, tokenizer, label_encoder, df = load_incident_model()
    
    # If local model fails, try fallback
    if model is None:
        st.warning("Couldn't load local model. Using fallback model.")
        model, tokenizer, label_encoder, df = load_model_fallback()

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
            # Predict incident details
            incident_details = predict_incident_details(
                model, tokenizer, user_input, label_encoder, df
            )

            # Generate detailed response using LangChain
            response = incident_chain.run(
                incident_text=user_input, 
                status=incident_details['predicted_status']
            )

            # Create three columns for detailed display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Incident Analysis")
                st.markdown(f"**Predicted Status:** {incident_details['predicted_status']}")

            with col2:
                st.subheader("Resolution Summary")
                st.markdown(f"**Details:** {incident_details['resolution_summary']}")

            with col3:
                st.subheader("Troubleshooting")
                # Check if link is a valid URL
                if incident_details['troubleshooting_link'].startswith(('http://', 'https://')):
                    st.markdown(f"**Link:** [{incident_details['troubleshooting_link']}]({incident_details['troubleshooting_link']})")
                else:
                    st.markdown(f"**Link:** {incident_details['troubleshooting_link']}")

            # Display AI-generated response
            st.subheader("Recommended Actions")
            st.markdown(f"<div class='highlight-box'>{response}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing incident: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by Transformers and LangChain*")

# Run the Streamlit app
if __name__ == "__main__":
    main()