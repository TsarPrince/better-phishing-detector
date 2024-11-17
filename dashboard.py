import streamlit as st
import imaplib
import email
from email.header import decode_header
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from datetime import datetime
import base64
from joblib import load

from HmmPhishingDetector import HMMPhishingDetector
from MemmPhishingDetector import MEMMPhishingDetector

def preprocess_text(text):
    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_features(df):
  # drop na
  df.dropna(inplace=True,axis=0)
  df.drop_duplicates(inplace=True)

  # preprocess text
  df["Email Text"] = df["Email Text"].apply(preprocess_text)

  # load vectorizer
  tf = load('models/tfidf_vectorizer.pkl')

  # extract feature vector
  X = tf.transform(df["Email Text"]).toarray()
  return X

def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()

def fetch_emails(email_address, app_password, num_emails=10):
    try:
        # Connect to Gmail's IMAP server
        imap_server = "imap.gmail.com"
        imap = imaplib.IMAP4_SSL(imap_server)
        imap.login(email_address, app_password)
        
        # Select the mailbox (inbox)
        imap.select("INBOX")
        
        # Search for all emails
        _, message_numbers = imap.search(None, "ALL")
        
        email_data = []
        
        # Get the last num_emails emails
        for num in message_numbers[0].split()[-num_emails:]:
            _, msg_data = imap.fetch(num, "(RFC822)")
            email_body = msg_data[0][1]
            msg = email.message_from_bytes(email_body)
            
            subject = decode_header(msg["subject"])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            
            sender = decode_header(msg["from"])[0][0]
            if isinstance(sender, bytes):
                sender = sender.decode()
            
            date_str = msg["date"]
            
            # Get email body
            body = get_email_body(msg)
            if body:
                # Combine subject and body for analysis
                full_text = f"{subject}\n{body}"
                email_data.append({
                    "subject": subject,
                    "sender": sender,
                    "date": date_str,
                    "full_text": full_text
                })
        
        imap.logout()
        return pd.DataFrame(email_data)
    
    except Exception as e:
        st.error(f"Error fetching emails: {str(e)}")
        return None

  
def analyze_emails(df, models):
    results = []
    
    for idx, row in df.iterrows():
        text = row['full_text']
        preprocessed_text = preprocess_text(text)
        
        predictions = {}
        for model_name, model in models.items():
            if model_name in ['HMMPhishingDetector', 'MEMMPhishingDetector']:
                pred = model.predict([preprocessed_text])[0]
            else:
                # For ML models, use the vectorizer first
                X = get_features(pd.DataFrame({'Email Text': [preprocessed_text]}))
                pred = model.predict(X)[0]
            predictions[model_name] = pred
        
        results.append({
            'subject': row['subject'],
            'sender': row['sender'],
            'date': row['date'],
            **predictions
        })
    
    return pd.DataFrame(results)

def create_dashboard(email_address=None, app_password=None):
    st.set_page_config(layout="wide", page_title="Live Gmail Phishing Detection")
    
    st.title("📧 Live Gmail Phishing Detection Dashboard")
    
    # Sidebar for email credentials
    with st.sidebar:
        st.header("Email Configuration")
        if not email_address:
            email_address = st.text_input("Gmail Address")
        if not app_password:
            app_password = st.text_input("App Password", type="password")
        num_emails = st.slider("Number of emails to analyze", 5, 50, 10)
        
        st.markdown("""
        ### How to get App Password:
        1. Go to Google Account settings
        2. Enable 2-Step Verification
        3. Go to Security → App passwords
        4. Generate new app password
        """)
    
    if not email_address or not app_password:
        st.warning("Please enter your Gmail credentials in the sidebar")
        return
    
    # Load models
    models = {
        'Naive Bayes': load('models/1_model_naive_bayes.pkl'),
        'Logistic Regression': load('models/2_model_logistic_regression.pkl'),
        'SGD Classifier': load('models/3_model_sgd_classifier.pkl'),
        'Decision Tree': load('models/4_model_decision_tree.pkl'),
        'Random Forest': load('models/5_model_random_forest.pkl'),
        'MLP Classifier': load('models/6_model_mlp.pkl'),
        'HMMPhishingDetector': load('models/hmm_phishing_detector.pkl'),
        'MEMMPhishingDetector': load('models/memm_phishing_detector.pkl')
    }
    
    # Fetch and analyze emails
    with st.spinner("Fetching and analyzing emails..."):
        emails_df = fetch_emails(email_address, app_password, num_emails)
        if emails_df is not None:
            results_df = analyze_emails(emails_df, models)
            
            # Display summary metrics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_phishing = results_df[[col for col in results_df.columns 
                                         if col not in ['subject', 'sender', 'date']]].mean().mean()
                st.metric("Average Phishing Probability", f"{(1-avg_phishing)*100:.1f}%")
            
            with col2:
                total_emails = len(results_df)
                st.metric("Total Emails Analyzed", total_emails)
            
            with col3:
                consensus_phishing = results_df[[col for col in results_df.columns 
                                               if col not in ['subject', 'sender', 'date']]].mode(axis=1)[0].mean()
                st.metric("Consensus Phishing Emails", f"{(1-consensus_phishing)*100:.1f}%")
            
            # Create heatmap of model predictions
            st.subheader("Model Predictions Heatmap")
            prediction_cols = [col for col in results_df.columns 
                             if col not in ['subject', 'sender', 'date']]
            
            fig = go.Figure(data=go.Heatmap(
                z=results_df[prediction_cols].values,
                x=prediction_cols,
                y=results_df['subject'],
                colorscale=[
                    [0, 'rgb(255, 0, 0)'],  # Red for phishing
                    [1, 'rgb(0, 255, 0)']   # Green for legitimate
                ],
                showscale=True
            ))
            
            fig.update_layout(
                title="Model Predictions by Email",
                xaxis_title="Models",
                yaxis_title="Email Subject",
                height=400 + (len(results_df) * 20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("Detailed Analysis Results")
            st.dataframe(
                results_df.style.background_gradient(
                    subset=[col for col in results_df.columns 
                           if col not in ['subject', 'sender', 'date']],
                    cmap='RdYlGn'
                ),
                height=400
            )
    
if __name__ == "__main__":
    create_dashboard()