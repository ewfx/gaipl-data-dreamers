import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

class IncidentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Custom dataset for incident management classification
        """
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def prepare_incident_data(csv_path):
    """
    Prepare incident data for model training
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Fill NaN values with empty strings to prevent errors
    df['Resolution Summary'] = df['Resolution Summary'].fillna('')
    df['Troubleshooting Doc Link'] = df['Troubleshooting Doc Link'].fillna('')
    
    # Combine all relevant text features
    df['incident_text'] = (
        df['Issue Summary'] + ' ' + 
        df['Department'] + ' ' + 
        df['Category'] + ' ' + 
        df['Resolution Summary'] + ' ' + 
        df['Troubleshooting Doc Link']
    )
    
    # Encode categorical labels
    label_encoder = LabelEncoder()
    df['encoded_status'] = label_encoder.fit_transform(df['Status'])
    
    return (
        df['incident_text'].tolist(), 
        df['encoded_status'].tolist(), 
        label_encoder
    )

def train_incident_model(
    train_texts, 
    train_labels, 
    model_name='bert-base-uncased',
    num_classes=None, 
    epochs=12,  # Slightly increased epochs
    batch_size=4,
    learning_rate=2e-5
):
    """
    Train a transformer-based incident classification model
    """
    # Determine number of classes
    if num_classes is None:
        num_classes = len(set(train_labels))

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Explicitly set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with explicit configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes,
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = IncidentDataset(X_train, y_train, tokenizer, max_length=256)
    val_dataset = IncidentDataset(X_val, y_val, tokenizer, max_length=256)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop with more detailed logging
    print("Starting model training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                val_loss += outputs.loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=1)
                val_accuracy += (preds == labels).float().mean().item()

        # Compute average metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {avg_val_accuracy:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, tokenizer

def predict_incident(model, tokenizer, incident_text, label_encoder, device=None):
    """
    Predict incident status for a given text
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()

    # Tokenize input with increased max length
    inputs = tokenizer(
        incident_text, 
        return_tensors='pt', 
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1)

    # Decode the predicted label
    predicted_status = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
    
    # Get confidence score
    confidence_score = predictions[0][predicted_class].item()
    
    return predicted_status, confidence_score

def main():
    # Path to your CSV file
    csv_path = 'C:/Users/srikr/OneDrive/Desktop/Hackathon/Incident_Sample.csv'
    
    # Prepare the data
    texts, labels, label_encoder = prepare_incident_data(csv_path)
    
    # Train the model
    trained_model, trained_tokenizer = train_incident_model(
        texts, 
        labels, 
        num_classes=len(set(labels))
    )

    # Example prediction
    sample_incidents = [
        "Network connectivity issues in IT department",
        "Printer not responding",
        "Software crashing on startup"
    ]

    print("\nPrediction Examples:")
    for incident in sample_incidents:
        predicted_status, confidence = predict_incident(
            trained_model, 
            trained_tokenizer, 
            incident, 
            label_encoder
        )
        print(f"Incident: {incident}")
        print(f"Predicted Status: {predicted_status}")
        print(f"Confidence Score: {confidence:.4f}\n")

    # Optional: Save the model
    model_save_path = './incident_management_model'
    trained_model.save_pretrained(model_save_path)
    trained_tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    main()