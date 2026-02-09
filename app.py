# Complete Fixed Code with Gemini API Integration - ALL ISSUES RESOLVED
import pandas as pd
import numpy as np
import streamlit as st
import hashlib
import secrets
from datetime import datetime, timedelta
import re
import time
import json
from textblob import TextBlob
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from dotenv import load_dotenv

# Add Gemini AI import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Install google-generativeai: pip install google-generativeai")

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv("D:/Medical Project/.env.txt")

# Security configuration
st.set_page_config(
    page_title="Telangana AI Healthcare Network",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TWILIO_API_KEY = os.getenv("TWILIO_API_KEY")

# Initialize session function - ADDED THIS
def initialize_session():
    """Initialize user session with security"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = secrets.token_urlsafe(16)
        st.session_state.session_start = datetime.now()
        st.session_state.attempts = 0
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {}
    if 'show_booking' not in st.session_state:
        st.session_state.show_booking = False
    if 'selected_hospital' not in st.session_state:
        st.session_state.selected_hospital = None
    if 'show_directions' not in st.session_state:
        st.session_state.show_directions = False
    if 'directions_hospital' not in st.session_state:
        st.session_state.directions_hospital = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'current_language' not in st.session_state:
        st.session_state.current_language = 'english'
    if 'show_symptom_checker' not in st.session_state:
        st.session_state.show_symptom_checker = False
    if 'show_reviews' not in st.session_state:
        st.session_state.show_reviews = False
    if 'appointment_submitted' not in st.session_state:
        st.session_state.appointment_submitted = False
    if 'highlight_hospital_id' not in st.session_state:
        st.session_state.highlight_hospital_id = None
    
    # Initialize AI agent session variables
    if 'symptom_text' not in st.session_state:
        st.session_state.symptom_text = ''
    if 'patient_age' not in st.session_state:
        st.session_state.patient_age = 30
    if 'patient_gender_idx' not in st.session_state:
        st.session_state.patient_gender_idx = 0
    if 'symptom_duration_idx' not in st.session_state:
        st.session_state.symptom_duration_idx = 0
    if 'symptom_severity_idx' not in st.session_state:
        st.session_state.symptom_severity_idx = 0
    if 'symptom_analysis_data' not in st.session_state:
        st.session_state.symptom_analysis_data = None
    if 'diagnosis_data' not in st.session_state:
        st.session_state.diagnosis_data = None
    if 'top_diagnosis' not in st.session_state:
        st.session_state.top_diagnosis = None
    if 'top_diagnosis_details' not in st.session_state:
        st.session_state.top_diagnosis_details = None
    if 'treatment_data' not in st.session_state:
        st.session_state.treatment_data = None
    if 'full_ai_report' not in st.session_state:
        st.session_state.full_ai_report = None
    
    # Track which AI agent to show
    if 'active_ai_agent' not in st.session_state:
        st.session_state.active_ai_agent = None

# AI-Powered Healthcare System with Enhanced Agents
class AISymptomChecker:
    def __init__(self):
        self.symptom_database = self.load_symptom_database()
        self.diagnosis_database = self.load_diagnosis_database()
        self.treatment_database = self.load_treatment_database()
        self.vectorizer = TfidfVectorizer()
        self.train_model()
        # Map body systems to diagnosis categories
        self.system_to_diagnosis_map = self.create_system_diagnosis_map()
        # Enhanced symptom-based medicine database
        self.symptom_medicine_database = self.create_symptom_medicine_database()
        
        # SIMPLIFIED GEMINI API SETUP - FIXED
        self.gemini_api_key = GEMINI_API_KEY
        self.gemini_model = None
        self.gemini_available = False
        
        if GEMINI_AVAILABLE and self.gemini_api_key and self.gemini_api_key.strip():
            try:
                # Configure Gemini
                genai.configure(api_key=self.gemini_api_key)
                
                # FIX: Simple and reliable approach
                # List available models first
                try:
                    available_models = genai.list_models()
                    model_names = [model.name for model in available_models]
                    st.info(f"Available Gemini models: {', '.join(model_names)}")
                    
                    # Try to use gemini-pro first (most common)
                    if 'models/gemini-pro' in model_names or 'gemini-pro' in model_names:
                        model_name = 'gemini-pro'
                        self.gemini_model = genai.GenerativeModel(model_name)
                        
                        # Test connection with a simple prompt
                        response = self.gemini_model.generate_content("Test connection")
                        if response and response.text:
                            self.gemini_available = True
                            st.success("✅ Gemini AI connected successfully!")
                        else:
                            st.warning("Gemini responded but no text returned")
                    
                    # If gemini-pro not available, try gemini-1.0-pro
                    elif 'models/gemini-1.0-pro' in model_names or 'gemini-1.0-pro' in model_names:
                        model_name = 'gemini-1.0-pro'
                        self.gemini_model = genai.GenerativeModel(model_name)
                        response = self.gemini_model.generate_content("Test")
                        if response and response.text:
                            self.gemini_available = True
                            st.success("✅ Gemini AI connected successfully with gemini-1.0-pro!")
                    
                    # Try any available model
                    else:
                        for model_name in model_names:
                            try:
                                if 'gemini' in model_name.lower():
                                    self.gemini_model = genai.GenerativeModel(model_name)
                                    response = self.gemini_model.generate_content("Hello")
                                    if response and response.text:
                                        self.gemini_available = True
                                        st.success(f"✅ Gemini AI connected with {model_name}!")
                                        break
                            except:
                                continue
                                
                except Exception as model_error:
                    st.warning(f"⚠️ Could not list models: {str(model_error)[:200]}")
                    
                    # Fallback: Try direct connection with common model names
                    common_models = ['gemini-pro', 'gemini-1.0-pro', 'models/gemini-pro']
                    for model_name in common_models:
                        try:
                            self.gemini_model = genai.GenerativeModel(model_name)
                            response = self.gemini_model.generate_content("Hello")
                            if response and response.text:
                                self.gemini_available = True
                                st.success(f"✅ Gemini AI connected with {model_name}!")
                                break
                        except:
                            continue
                
                if not self.gemini_available:
                    st.warning("⚠️ Could not connect to Gemini API. Using local AI only.")
                    
            except Exception as e:
                error_msg = str(e)
                # Don't show full error to user, just log it
                print(f"Gemini API Error: {error_msg[:200]}")
                self.gemini_available = False
        elif GEMINI_AVAILABLE and not self.gemini_api_key:
            st.warning("⚠️ Gemini API key not found in .env file")
        elif not GEMINI_AVAILABLE:
            st.info("ℹ️ google-generativeai package not installed. Using local AI only.")

    def create_system_diagnosis_map(self):
        """Create mapping between body systems and relevant diagnoses"""
        return {
            'neurology': ['Migraine', 'Stroke', 'Epilepsy', 'Parkinson\'s Disease'],
            'cardiology': ['Hypertension', 'Heart Attack', 'Angina', 'Arrhythmia'],
            'respiratory': ['Influenza (Flu)', 'COVID-19', 'Asthma', 'Pneumonia', 'Common Cold'],
            'gastroenterology': ['Gastroenteritis', 'Appendicitis', 'Food Poisoning', 'Irritable Bowel Syndrome'],
            'orthopedics': ['Arthritis', 'Fracture', 'Sprain', 'Back Pain'],
            'pediatrics': ['Common Cold', 'Influenza (Flu)', 'Chickenpox', 'Measles', 'Ear Infection'],
            'dermatology': ['Eczema', 'Psoriasis', 'Acne', 'Allergic Reaction'],
            'emergency': ['Appendicitis', 'Heart Attack', 'Stroke', 'Severe Injury']
        }

    def create_symptom_medicine_database(self):
        """Create comprehensive symptom-based medicine database"""
        return {
            # Pain and Fever Symptoms
            'headache': {
                'mild': ['Paracetamol 500mg every 4-6 hours', 'Ibuprofen 200-400mg every 6 hours'],
                'moderate': ['Paracetamol 1000mg every 6 hours', 'Ibuprofen 400-600mg every 6 hours', 'Aspirin 300-600mg every 4 hours'],
                'severe': ['Prescription pain relievers', 'Sumatriptan for migraine', 'Emergency care if sudden/severe']
            },
            'fever': {
                'mild': ['Paracetamol 500mg every 4-6 hours', 'Stay hydrated', 'Cool compress'],
                'moderate': ['Paracetamol 500-1000mg every 6 hours', 'Ibuprofen 400-600mg every 6 hours'],
                'severe': ['Seek medical attention', 'Paracetamol 1000mg + Ibuprofen 600mg alternated', 'Cool baths']
            },
            'body aches': {
                'mild': ['Ibuprofen 200-400mg every 6 hours', 'Warm compress', 'Gentle stretching'],
                'moderate': ['Ibuprofen 400-600mg every 6 hours', 'Naproxen 220mg every 8-12 hours', 'Epsom salt bath'],
                'severe': ['Prescription NSAIDs', 'Muscle relaxants if prescribed', 'Physical therapy']
            },
            'joint pain': {
                'mild': ['Ibuprofen 200-400mg every 6 hours', 'Topical pain relief creams', 'Rest'],
                'moderate': ['Ibuprofen 400-600mg every 6 hours', 'Glucosamine supplements', 'Compression'],
                'severe': ['Prescription anti-inflammatory', 'Joint injections', 'Physical therapy']
            },
            'back pain': {
                'mild': ['Ibuprofen 200-400mg every 6 hours', 'Heat therapy', 'Gentle exercise'],
                'moderate': ['Ibuprofen 400-600mg every 6 hours', 'Muscle relaxants if prescribed', 'Physical therapy'],
                'severe': ['Prescription pain medication', 'Emergency care if numbness/weakness', 'MRI evaluation']
            },
            
            # Respiratory Symptoms
            'cough': {
                'dry': ['Dextromethorphan 10-20mg every 4 hours', 'Honey and lemon', 'Stay hydrated'],
                'wet': ['Guaifenesin 200-400mg every 4 hours', 'Expectorants', 'Steam inhalation'],
                'severe': ['Prescription cough suppressants', 'Bronchodilators if wheezing', 'Chest X-ray if persistent']
            },
            'sore throat': {
                'mild': ['Warm salt water gargles 3-4 times daily', 'Throat lozenges', 'Honey tea'],
                'moderate': ['Ibuprofen for pain relief', 'Antibacterial throat spray', 'Rest voice'],
                'severe': ['Antibiotics if bacterial infection', 'Steroids if severe swelling', 'Emergency if breathing difficulty']
            },
            'runny nose': {
                'mild': ['Saline nasal spray', 'Antihistamines: Chlorpheniramine 4mg every 4-6 hours'],
                'moderate': ['Decongestants: Pseudoephedrine 30-60mg every 4-6 hours', 'Nasal corticosteroids'],
                'severe': ['Prescription nasal sprays', 'Allergy testing if chronic', 'Immunotherapy']
            },
            'nasal congestion': {
                'mild': ['Saline nasal spray', 'Steam inhalation', 'Elevate head while sleeping'],
                'moderate': ['Decongestants: Pseudoephedrine 60mg every 6 hours', 'Nasal corticosteroids'],
                'severe': ['Prescription nasal sprays', 'Sinus irrigation', 'ENT consultation']
            },
            'shortness of breath': {
                'mild': ['Rest', 'Sit upright', 'Avoid triggers'],
                'moderate': ['Bronchodilator inhaler if prescribed', 'Seek medical attention'],
                'severe': ['EMERGENCY - Call 108', 'Use emergency inhaler', 'Oxygen if available']
            },
            'wheezing': {
                'mild': ['Bronchodilator inhaler', 'Avoid triggers', 'Stay calm'],
                'moderate': ['Bronchodilator + corticosteroid inhaler', 'Seek medical help'],
                'severe': ['EMERGENCY - Call 108', 'Nebulizer treatment', 'Hospital admission']
            },
            
            # Gastrointestinal Symptoms
            'nausea': {
                'mild': ['Ginger tea or capsules', 'Small frequent meals', 'Avoid strong odors'],
                'moderate': ['Antacids if prescribed', 'Anti-nausea wristbands', 'Clear liquids'],
                'severe': ['Prescription anti-emetics', 'IV fluids if dehydrated', 'Hospital evaluation']
            },
            'vomiting': {
                'mild': ['Clear liquids only', 'Oral rehydration solution', 'Rest'],
                'moderate': ['Anti-nausea medication if prescribed', 'Bland diet when tolerated'],
                'severe': ['EMERGENCY if persistent', 'IV fluids', 'Anti-emetic injections']
            },
            'diarrhea': {
                'mild': ['Oral rehydration salts (ORS)', 'BRAT diet', 'Probiotics'],
                'moderate': ['Loperamide 4mg initially then 2mg after each loose stool (max 16mg/day)', 'Zinc supplements'],
                'severe': ['EMERGENCY if bloody/dehydrated', 'Antibiotics if bacterial', 'Hospital care']
            },
            'abdominal pain': {
                'mild': ['Apply warm compress', 'Peppermint tea', 'Avoid spicy foods'],
                'moderate': ['Antacids if prescribed', 'Bland diet', 'Medical evaluation'],
                'severe': ['EMERGENCY if severe/sudden', 'No pain medication before diagnosis', 'Immediate medical care']
            },
            'heartburn': {
                'mild': ['Antacids (Tums, Rolaids)', 'Avoid trigger foods', 'Elevate head'],
                'moderate': ['H2 blockers (Ranitidine)', 'Proton pump inhibitors if prescribed'],
                'severe': ['Prescription medications', 'Endoscopy if chronic', 'Lifestyle changes']
            },
            
            # Other Common Symptoms
            'fatigue': {
                'mild': ['Adequate rest and sleep', 'Stay hydrated', 'Balanced nutrition'],
                'moderate': ['Iron supplements if anemic', 'Vitamin B12', 'Thyroid check'],
                'severe': ['Medical evaluation required', 'Sleep study if needed', 'Chronic fatigue assessment']
            },
            'dizziness': {
                'mild': ['Sit or lie down', 'Stay hydrated', 'Move slowly'],
                'moderate': ['Medical evaluation', 'Balance exercises', 'Avoid sudden movements'],
                'severe': ['EMERGENCY if with chest pain', 'Neurological evaluation', 'Hospital care']
            },
            'rash': {
                'mild': ['Calamine lotion', 'Antihistamines: Cetirizine 10mg daily', 'Cool compress'],
                'moderate': ['Topical corticosteroids', 'Avoid allergens', 'Medical evaluation'],
                'severe': ['EMERGENCY if with swelling', 'Systemic steroids', 'Allergy testing']
            },
            'itching': {
                'mild': ['Moisturizers', 'Cool baths', 'Antihistamines'],
                'moderate': ['Topical corticosteroids', 'Avoid scratching', 'Identify triggers'],
                'severe': ['Prescription medications', 'Phototherapy', 'Immunosuppressants']
            },
            'insomnia': {
                'mild': ['Sleep hygiene', 'Relaxation techniques', 'Chamomile tea'],
                'moderate': ['Melatonin supplements', 'Cognitive behavioral therapy', 'Medical evaluation'],
                'severe': ['Prescription sleep aids', 'Sleep study', 'Psychiatric evaluation']
            },
            
            # Emergency Symptoms (Always severe)
            'chest pain': {
                'severe': [
                    'EMERGENCY - Call 108 immediately',
                    'Aspirin 325mg if no allergy (chewable)',
                    'Nitroglycerin if prescribed',
                    'Do NOT drive yourself to hospital'
                ]
            },
            'severe headache': {
                'severe': [
                    'EMERGENCY - Call 108',
                    'Do not take pain medication before evaluation',
                    'Neurological evaluation required',
                    'Possible stroke or aneurysm symptoms'
                ]
            },
            'loss of consciousness': {
                'severe': [
                    'EMERGENCY - Call 108',
                    'Check breathing and pulse',
                    'CPR if trained and no pulse',
                    'Do not give anything by mouth'
                ]
            },
            'severe bleeding': {
                'severe': [
                    'EMERGENCY - Call 108',
                    'Apply direct pressure with clean cloth',
                    'Elevate injured area above heart',
                    'Do not remove embedded objects'
                ]
            },
            'paralysis': {
                'severe': [
                    'EMERGENCY - Call 108',
                    'Do not move patient unless in danger',
                    'Neurological emergency',
                    'Time is critical for treatment'
                ]
            }
        }

    def load_symptom_database(self):
        """Load AI training data for symptoms and conditions"""
        return {
            'cardiology': [
                'chest pain', 'chest discomfort', 'shortness of breath', 'palpitations',
                'heart racing', 'dizziness', 'fainting', 'swollen legs', 'fatigue',
                'pain radiating to arm', 'sweating', 'nausea', 'irregular heartbeat',
                'high blood pressure', 'low blood pressure', 'chest tightness'
            ],
            'neurology': [
                'headache', 'migraine', 'dizziness', 'numbness', 'tingling',
                'seizures', 'memory loss', 'vision problems', 'speech difficulty',
                'weakness in limbs', 'balance problems', 'tremors', 'confusion',
                'loss of consciousness', 'coordination problems'
            ],
            'orthopedics': [
                'joint pain', 'back pain', 'neck pain', 'swelling', 'fracture',
                'limited movement', 'stiffness', 'muscle pain', 'bone pain',
                'sports injury', 'accident trauma', 'arthritis', 'sprain',
                'dislocation', 'reduced mobility'
            ],
            'gastroenterology': [
                'abdominal pain', 'stomach ache', 'nausea', 'vomiting', 'diarrhea',
                'constipation', 'bloating', 'heartburn', 'indigestion', 'blood in stool',
                'loss of appetite', 'weight loss', 'acid reflux', 'gas', 'stomach cramps'
            ],
            'respiratory': [
                'cough', 'shortness of breath', 'wheezing', 'chest congestion',
                'sore throat', 'runny nose', 'sneezing', 'fever', 'fatigue',
                'difficulty breathing', 'chest pain', 'loss of smell', 'loss of taste'
            ],
            'dermatology': [
                'rash', 'itching', 'redness', 'swelling', 'blisters',
                'dry skin', 'acne', 'eczema', 'psoriasis', 'skin infection',
                'allergic reaction', 'hives', 'skin discoloration'
            ],
            'pediatrics': [
                'fever', 'cough', 'cold', 'ear pain', 'rashes', 'allergies',
                'growth concerns', 'developmental delays', 'childhood vaccinations',
                'diarrhea', 'vomiting', 'poor appetite', 'sleep problems'
            ],
            'emergency': [
                'severe pain', 'unconscious', 'bleeding', 'difficulty breathing',
                'chest pressure', 'sudden weakness', 'severe headache', 'burn',
                'poisoning', 'major injury', 'stroke symptoms', 'heart attack symptoms'
            ]
        }

    def load_diagnosis_database(self):
        """Load common diagnoses with associated symptoms"""
        return {
            'Common Cold': {
                'symptoms': ['cough', 'runny nose', 'sore throat', 'sneezing', 'mild fever', 'fatigue'],
                'urgency': 'low',
                'specialty': 'general medicine',
                'body_systems': ['respiratory', 'pediatrics'],
                'description': 'Viral infection of the upper respiratory tract causing mild symptoms',
                'common_in': 'All age groups',
                'complications': ['Sinusitis', 'Ear infection', 'Bronchitis']
            },
            'Influenza (Flu)': {
                'symptoms': ['fever', 'cough', 'sore throat', 'body aches', 'fatigue', 'headache'],
                'urgency': 'medium',
                'specialty': 'general medicine',
                'body_systems': ['respiratory', 'pediatrics'],
                'description': 'Viral respiratory illness more severe than common cold',
                'common_in': 'All age groups, severe in elderly and children',
                'complications': ['Pneumonia', 'Bronchitis', 'Sinus infections']
            },
            'Migraine': {
                'symptoms': ['severe headache', 'nausea', 'sensitivity to light', 'sensitivity to sound'],
                'urgency': 'medium',
                'specialty': 'neurology',
                'body_systems': ['neurology'],
                'description': 'Neurological condition characterized by intense headaches',
                'common_in': 'Adults, especially women',
                'complications': ['Chronic migraine', 'Status migrainosus']
            },
            'Gastroenteritis': {
                'symptoms': ['diarrhea', 'vomiting', 'abdominal pain', 'nausea', 'fever'],
                'urgency': 'medium',
                'specialty': 'gastroenterology',
                'body_systems': ['gastroenterology', 'pediatrics'],
                'description': 'Inflammation of stomach and intestines usually from infection',
                'common_in': 'All age groups',
                'complications': ['Dehydration', 'Electrolyte imbalance']
            },
            'Urinary Tract Infection': {
                'symptoms': ['burning sensation during urination', 'frequent urination', 'pelvic pain'],
                'urgency': 'medium',
                'specialty': 'urology',
                'body_systems': ['urology'],
                'description': 'Infection in any part of urinary system',
                'common_in': 'Women more than men',
                'complications': ['Kidney infection', 'Sepsis']
            },
            'Hypertension': {
                'symptoms': ['headache', 'dizziness', 'chest pain', 'shortness of breath'],
                'urgency': 'high',
                'specialty': 'cardiology',
                'body_systems': ['cardiology'],
                'description': 'High blood pressure, often asymptomatic',
                'common_in': 'Adults, especially over 40',
                'complications': ['Heart attack', 'Stroke', 'Kidney damage']
            },
            'Diabetes': {
                'symptoms': ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision'],
                'urgency': 'medium',
                'specialty': 'endocrinology',
                'body_systems': ['endocrinology'],
                'description': 'Chronic condition affecting insulin production or use',
                'common_in': 'Adults, children with type 1',
                'complications': ['Neuropathy', 'Retinopathy', 'Kidney disease']
            },
            'Asthma': {
                'symptoms': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing'],
                'urgency': 'high',
                'specialty': 'pulmonology',
                'body_systems': ['respiratory'],
                'description': 'Chronic inflammatory disease of airways',
                'common_in': 'All age groups, common in children',
                'complications': ['Status asthmaticus', 'Respiratory failure']
            },
            'COVID-19': {
                'symptoms': ['fever', 'cough', 'shortness of breath', 'loss of taste', 'loss of smell'],
                'urgency': 'high',
                'specialty': 'infectious disease',
                'body_systems': ['respiratory'],
                'description': 'Viral respiratory disease caused by SARS-CoV-2',
                'common_in': 'All age groups',
                'complications': ['Pneumonia', 'ARDS', 'Long COVID']
            },
            'Appendicitis': {
                'symptoms': ['abdominal pain', 'nausea', 'vomiting', 'fever', 'loss of appetite'],
                'urgency': 'high',
                'specialty': 'emergency surgery',
                'body_systems': ['gastroenterology', 'emergency'],
                'description': 'Inflammation of appendix requiring emergency surgery',
                'common_in': 'Children and young adults',
                'complications': ['Ruptured appendix', 'Peritonitis']
            },
            'Ear Infection': {
                'symptoms': ['ear pain', 'fever', 'hearing loss', 'ear drainage', 'irritability'],
                'urgency': 'medium',
                'specialty': 'pediatrics',
                'body_systems': ['pediatrics'],
                'description': 'Infection in middle ear often following cold',
                'common_in': 'Children',
                'complications': ['Hearing loss', 'Mastoiditis']
            },
            'Bronchitis': {
                'symptoms': ['cough', 'shortness of breath', 'chest discomfort', 'fatigue', 'fever'],
                'urgency': 'medium',
                'specialty': 'pulmonology',
                'body_systems': ['respiratory'],
                'description': 'Inflammation of bronchial tubes',
                'common_in': 'Adults, especially smokers',
                'complications': ['Pneumonia', 'Chronic bronchitis']
            },
            'Sinusitis': {
                'symptoms': ['facial pain', 'headache', 'nasal congestion', 'loss of smell', 'cough'],
                'urgency': 'low',
                'specialty': 'ent',
                'body_systems': ['respiratory'],
                'description': 'Inflammation of sinuses',
                'common_in': 'Adults',
                'complications': ['Chronic sinusitis', 'Meningitis']
            },
            'Allergic Rhinitis': {
                'symptoms': ['sneezing', 'runny nose', 'itchy eyes', 'nasal congestion'],
                'urgency': 'low',
                'specialty': 'allergy',
                'body_systems': ['respiratory'],
                'description': 'Allergic inflammation of nasal airways',
                'common_in': 'All age groups',
                'complications': ['Sinusitis', 'Asthma exacerbation']
            },
            'Depression': {
                'symptoms': ['sadness', 'loss of interest', 'fatigue', 'sleep changes', 'appetite changes'],
                'urgency': 'medium',
                'specialty': 'psychiatry',
                'body_systems': ['psychiatry'],
                'description': 'Mood disorder causing persistent sadness',
                'common_in': 'Adults, adolescents',
                'complications': ['Suicidal thoughts', 'Substance abuse']
            },
            'Anxiety Disorder': {
                'symptoms': ['excessive worry', 'restlessness', 'fatigue', 'difficulty concentrating', 'sleep problems'],
                'urgency': 'medium',
                'specialty': 'psychiatry',
                'body_systems': ['psychiatry'],
                'description': 'Mental health disorder characterized by excessive anxiety',
                'common_in': 'All age groups',
                'complications': ['Panic attacks', 'Depression']
            },
            'Pneumonia': {
                'symptoms': ['cough with phlegm', 'fever', 'chills', 'difficulty breathing', 'chest pain'],
                'urgency': 'high',
                'specialty': 'pulmonology',
                'body_systems': ['respiratory'],
                'description': 'Infection that inflames air sacs in lungs',
                'common_in': 'All age groups, severe in elderly and infants',
                'complications': ['Respiratory failure', 'Sepsis']
            }
        }

    def load_treatment_database(self):
        """Load treatment recommendations for common conditions with detailed medicines"""
        return {
            'Common Cold': {
                'description': 'Viral infection of upper respiratory tract',
                'self_care': [
                    'Rest and sleep',
                    'Drink plenty of fluids (water, herbal tea, broth)',
                    'Use saline nasal spray or drops',
                    'Gargle with warm salt water',
                    'Use a humidifier or take steamy showers',
                    'Eat chicken soup or warm liquids',
                    'Avoid smoking and secondhand smoke'
                ],
                'medications': {
                    'Pain/Fever Relievers': ['Paracetamol (Acetaminophen) 500mg every 4-6 hours', 'Ibuprofen 200-400mg every 6 hours'],
                    'Decongestants': ['Pseudoephedrine 30-60mg every 4-6 hours', 'Phenylephrine nasal spray'],
                    'Antihistamines': ['Chlorpheniramine 4mg every 4-6 hours', 'Diphenhydramine 25-50mg at bedtime'],
                    'Cough Suppressants': ['Dextromethorphan 10-20mg every 4 hours'],
                    'Expectorants': ['Guaifenesin 200-400mg every 4 hours']
                },
                'precautions': [
                    'Wash hands frequently with soap and water',
                    'Use tissues when coughing or sneezing',
                    'Avoid close contact with others',
                    'Disinfect surfaces regularly',
                    'Stay home if you have fever',
                    'Get adequate sleep',
                    'Avoid sharing utensils or personal items'
                ],
                'dietary_advice': [
                    'Vitamin C rich foods (oranges, lemons, bell peppers)',
                    'Zinc supplements (lozenges)',
                    'Honey for cough (not for children under 1)',
                    'Ginger tea for sore throat',
                    'Garlic for immune support',
                    'Stay hydrated with water and electrolyte drinks'
                ],
                'when_to_see_doctor': 'If symptoms last more than 10 days, high fever >102°F, difficulty breathing, or severe headache',
                'follow_up': 'Usually resolves within 7-10 days. Follow-up if symptoms worsen.',
                'emergency_signs': [
                    'Difficulty breathing',
                    'Chest pain or pressure',
                    'Sudden dizziness',
                    'Severe vomiting',
                    'Confusion'
                ]
            },
            'Influenza (Flu)': {
                'description': 'Viral respiratory illness more severe than common cold',
                'self_care': [
                    'Bed rest for at least 24 hours after fever subsides',
                    'Drink plenty of fluids to prevent dehydration',
                    'Use humidifier to ease breathing',
                    'Warm baths to reduce body aches',
                    'Apply warm compress for sinus pain',
                    'Take steamy showers for congestion'
                ],
                'medications': {
                    'Antiviral Drugs': ['Oseltamivir (Tamiflu) 75mg twice daily for 5 days', 'Zanamivir (Relenza) inhaled'],
                    'Pain/Fever Relievers': ['Paracetamol 500-1000mg every 6 hours', 'Ibuprofen 400-600mg every 6 hours'],
                    'Decongestants': ['Pseudoephedrine 60mg every 4-6 hours'],
                    'Cough Medicine': ['Dextromethorphan 10-20mg every 4 hours', 'Guaifenesin for productive cough']
                },
                'precautions': [
                    'Annual flu vaccination',
                    'Stay home for at least 24 hours after fever is gone',
                    'Wear mask in public if symptomatic',
                    'Practice respiratory hygiene',
                    'Avoid touching eyes, nose, mouth',
                    'Clean and disinfect surfaces'
                ],
                'dietary_advice': [
                    'Chicken soup or bone broth',
                    'Elderberry syrup',
                    'Green tea with honey',
                    'Probiotic foods (yogurt, kefir)',
                    'Garlic and onion in meals',
                    'Stay well-hydrated'
                ],
                'when_to_see_doctor': 'Difficulty breathing, chest pain, persistent high fever, confusion, severe weakness',
                'follow_up': 'Most recover within 1-2 weeks. Elderly and high-risk patients need monitoring.',
                'emergency_signs': [
                    'Shortness of breath or difficulty breathing',
                    'Persistent chest pain',
                    'Severe dehydration',
                    'High fever not responding to medication',
                    'Bluish lips or face'
                ]
            },
            'Migraine': {
                'description': 'Neurological condition causing severe headaches',
                'self_care': [
                    'Rest in dark, quiet room',
                    'Apply cold compress to forehead or neck',
                    'Stay hydrated',
                    'Practice relaxation techniques',
                    'Avoid trigger foods',
                    'Maintain regular sleep schedule'
                ],
                'medications': {
                    'Acute Treatment': ['Sumatriptan 50-100mg at onset', 'Rizatriptan 10mg', 'Eletriptan 40mg'],
                    'Pain Relievers': ['Ibuprofen 400-800mg', 'Naproxen 500-550mg', 'Aspirin 900-1000mg'],
                    'Anti-nausea': ['Metoclopramide 10mg', 'Prochlorperazine 5-10mg'],
                    'Preventive': ['Propranolol 40-240mg daily', 'Topiramate 25-100mg daily', 'Amitriptyline 10-150mg daily']
                },
                'precautions': [
                    'Identify and avoid triggers (stress, certain foods, lack of sleep)',
                    'Maintain headache diary',
                    'Regular exercise but not during attacks',
                    'Stay hydrated',
                    'Avoid skipping meals',
                    'Manage stress through meditation/yoga'
                ],
                'dietary_advice': [
                    'Avoid aged cheeses, processed meats, chocolate',
                    'Limit caffeine and alcohol',
                    'Magnesium-rich foods (spinach, almonds, avocado)',
                    'Riboflavin (vitamin B2) supplements',
                    'Coenzyme Q10 supplements',
                    'Regular meals to prevent low blood sugar'
                ],
                'when_to_see_doctor': 'Severe headache with fever/stiff neck, sudden severe headache, headache after head injury',
                'follow_up': 'Keep headache diary. Regular follow-up if chronic.',
                'emergency_signs': [
                    'Worst headache of your life',
                    'Headache with fever, stiff neck, confusion',
                    'Headache after head injury',
                    'Sudden severe headache',
                    'Headache with weakness or numbness'
                ]
            },
            'Gastroenteritis': {
                'description': 'Inflammation of stomach and intestines from infection',
                'self_care': [
                    'Clear liquid diet for first 24 hours',
                    'Oral rehydration solutions (ORS)',
                    'Rest',
                    'Avoid dairy, fatty, spicy foods',
                    'Gradually reintroduce bland foods'
                ],
                'medications': {
                    'Rehydration': ['Oral Rehydration Salts (ORS) as directed'],
                    'Anti-diarrheal': ['Loperamide 4mg initially then 2mg after each loose stool (max 16mg/day)', 'Bismuth subsalicylate 524mg every 30-60 minutes'],
                    'Anti-nausea': ['Ondansetron 4-8mg as needed'],
                    'Probiotics': ['Lactobacillus GG, Saccharomyces boulardii'],
                    'Antibiotics': ['Only if bacterial cause confirmed by doctor']
                },
                'precautions': [
                    'Wash hands thoroughly with soap',
                    'Disinfect surfaces',
                    'Avoid preparing food for others while sick',
                    'Practice safe food handling',
                    'Drink bottled water when traveling',
                    'Cook food thoroughly'
                ],
                'dietary_advice': [
                    'BRAT diet (bananas, rice, applesauce, toast)',
                    'Clear broths and soups',
                    'Crackers and plain toast',
                    'Avoid dairy, caffeine, alcohol',
                    'Small, frequent meals',
                    'Probiotic yogurt once recovered'
                ],
                'when_to_see_doctor': 'Blood in stool, high fever >102°F, signs of dehydration, symptoms lasting >3 days',
                'follow_up': 'Gradually resume normal diet. Follow-up if symptoms persist.',
                'emergency_signs': [
                    'Severe dehydration (dry mouth, no urine, dizziness)',
                    'Bloody stools',
                    'High fever with stiff neck',
                    'Severe abdominal pain',
                    'Vomiting preventing fluid intake'
                ]
            },
            'Hypertension': {
                'description': 'High blood pressure requiring long-term management',
                'self_care': [
                    'Regular exercise (30 minutes most days)',
                    'Reduce sodium intake',
                    'Maintain healthy weight',
                    'Limit alcohol consumption',
                    'Stress management techniques',
                    'Quit smoking',
                    'Monitor blood pressure regularly'
                ],
                'medications': {
                    'ACE Inhibitors': ['Lisinopril 10-40mg daily', 'Enalapril 5-40mg daily'],
                    'ARBs': ['Losartan 25-100mg daily', 'Valsartan 80-320mg daily'],
                    'Beta Blockers': ['Atenolol 25-100mg daily', 'Metoprolol 50-200mg daily'],
                    'Diuretics': ['Hydrochlorothiazide 12.5-50mg daily', 'Chlorthalidone 12.5-25mg daily'],
                    'Calcium Channel Blockers': ['Amlodipine 5-10mg daily', 'Diltiazem 120-360mg daily']
                },
                'precautions': [
                    'Regular blood pressure monitoring',
                    'Take medications as prescribed',
                    'Regular doctor visits',
                    'Avoid NSAIDs unless prescribed',
                    'Limit caffeine',
                    'Check labels for sodium content'
                ],
                'dietary_advice': [
                    'DASH diet (fruits, vegetables, whole grains)',
                    'Limit sodium to <2300mg daily',
                    'Potassium-rich foods (bananas, spinach, sweet potatoes)',
                    'Magnesium-rich foods (nuts, seeds, legumes)',
                    'Limit processed foods',
                    'Reduce saturated fats'
                ],
                'when_to_see_doctor': 'Regular monitoring needed. Immediate if BP >180/120, severe headache, chest pain, vision changes',
                'follow_up': 'Regular blood pressure checks. Adjust medication as needed.',
                'emergency_signs': [
                    'Blood pressure >180/120',
                    'Severe headache',
                    'Chest pain',
                    'Difficulty breathing',
                    'Blurred vision',
                    'Nausea/vomiting'
                ]
            },
            'Diabetes': {
                'description': 'Chronic condition affecting blood sugar control',
                'self_care': [
                    'Monitor blood sugar regularly',
                    'Follow diabetic diet plan',
                    'Regular physical activity',
                    'Foot care daily',
                    'Medication adherence',
                    'Regular eye exams',
                    'Stress management'
                ],
                'medications': {
                    'Type 1 Diabetes': ['Insulin (multiple types and regimens)'],
                    'Type 2 Oral Medications': ['Metformin 500-2000mg daily', 'Sulfonylureas (Glipizide, Glyburide)', 'DPP-4 inhibitors (Sitagliptin)'],
                    'Injectable': ['GLP-1 receptor agonists (Liraglutide, Semaglutide)', 'Insulin as needed'],
                    'Other': ['SGLT2 inhibitors (Empagliflozin, Canagliflozin)']
                },
                'precautions': [
                    'Carry glucose tablets or candy',
                    'Wear medical alert bracelet',
                    'Regular HbA1c testing',
                    'Foot examination daily',
                    'Never skip meals',
                    'Carry insulin if prescribed'
                ],
                'dietary_advice': [
                    'Consistent carbohydrate intake',
                    'High fiber foods',
                    'Lean proteins',
                    'Healthy fats',
                    'Limit sugary foods and drinks',
                    'Portion control',
                    'Regular meal times'
                ],
                'when_to_see_doctor': 'Regular check-ups. Immediate if blood sugar >300mg/dL or <70mg/dL with symptoms.',
                'follow_up': 'Regular A1C tests (every 3-6 months). Annual eye and foot exams.',
                'emergency_signs': [
                    'Blood sugar >300mg/dL persistently',
                    'Blood sugar <70mg/dL with confusion',
                    'Ketones in urine with high blood sugar',
                    'Extreme thirst and frequent urination',
                    'Nausea, vomiting, abdominal pain'
                ]
            },
            'Asthma': {
                'description': 'Chronic inflammatory disease of airways',
                'self_care': [
                    'Identify and avoid triggers',
                    'Use air purifier',
                    'Practice breathing exercises',
                    'Stay hydrated',
                    'Maintain healthy weight',
                    'Get flu and pneumonia vaccines'
                ],
                'medications': {
                    'Quick Relief (Rescue)': ['Albuterol inhaler 1-2 puffs every 4-6 hours as needed', 'Levalbuterol inhaler'],
                    'Controller Medications': ['Inhaled corticosteroids (Fluticasone, Budesonide)', 'Combination inhalers (Advair, Symbicort)'],
                    'Oral Medications': ['Montelukast 10mg daily', 'Prednisone for exacerbations'],
                    'Biologics': ['Omalizumab for severe allergic asthma']
                },
                'precautions': [
                    'Always carry rescue inhaler',
                    'Use spacer with inhaler',
                    'Follow asthma action plan',
                    'Monitor peak flow regularly',
                    'Avoid smoke and strong odors',
                    'Manage allergies'
                ],
                'dietary_advice': [
                    'Anti-inflammatory foods (fruits, vegetables)',
                    'Omega-3 fatty acids (fish, flaxseed)',
                    'Vitamin D rich foods',
                    'Magnesium-rich foods',
                    'Avoid sulfites in dried fruits and wine',
                    'Stay well-hydrated'
                ],
                'when_to_see_doctor': 'Difficulty breathing despite medication, frequent attacks, waking up at night with symptoms',
                'follow_up': 'Regular pulmonary function tests. Adjust medication as needed.',
                'emergency_signs': [
                    'Rescue inhaler not helping',
                    'Severe shortness of breath',
                    'Difficulty speaking in full sentences',
                    'Lips or fingernails turning blue',
                    'Rapid worsening of symptoms'
                ]
            },
            'COVID-19': {
                'description': 'Viral respiratory disease',
                'self_care': [
                    'Isolate from others for at least 5 days',
                    'Rest',
                    'Stay hydrated',
                    'Monitor oxygen levels with pulse oximeter',
                    'Use humidifier',
                    'Get plenty of sleep'
                ],
                'medications': {
                    'Antiviral': ['Paxlovid (nirmatrelvir/ritonavir)', 'Molnupiravir', 'Remdesivir (for hospitalized)'],
                    'Symptom Relief': ['Paracetamol for fever', 'Ibuprofen for body aches', 'Dextromethorphan for cough'],
                    'Other': ['Corticosteroids for severe cases', 'Anticoagulants for high-risk patients']
                },
                'precautions': [
                    'Vaccination and boosters',
                    'Wear mask in crowded places',
                    'Practice good hand hygiene',
                    'Improve ventilation indoors',
                    'Isolate when symptomatic',
                    'Test if exposed or symptomatic'
                ],
                'dietary_advice': [
                    'Stay well-hydrated',
                    'Vitamin C and D supplements',
                    'Zinc lozenges',
                    'Protein-rich foods',
                    'Warm fluids (tea, soup)',
                    'Easily digestible foods if nauseous'
                ],
                'when_to_see_doctor': 'Difficulty breathing, persistent chest pain, confusion, oxygen saturation <94%',
                'follow_up': 'Follow quarantine guidelines. Monitor for long COVID symptoms.',
                'emergency_signs': [
                    'Trouble breathing',
                    'Persistent chest pain or pressure',
                    'New confusion',
                    'Inability to wake or stay awake',
                    'Pale, gray, or blue-colored skin, lips, or nail beds'
                ]
            }
        }

    def train_model(self):
        """Train the symptom-classification model"""
        symptoms = []
        specialties = []
        for specialty, symptom_list in self.symptom_database.items():
            symptoms.extend(symptom_list)
            specialties.extend([specialty] * len(symptom_list))
        
        self.symptom_corpus = symptoms
        self.specialty_labels = specialties
        self.tfidf_matrix = self.vectorizer.fit_transform(symptoms)

    def analyze_symptoms(self, symptom_text):
        """AI-powered symptom analysis and specialty recommendation"""
        if not symptom_text.strip():
            return "General Medicine", 0.0, []
        
        # Transform input symptoms
        symptom_vec = self.vectorizer.transform([symptom_text.lower()])
        
        # Calculate similarity with known symptoms
        similarities = cosine_similarity(symptom_vec, self.tfidf_matrix)
        
        if similarities.max() < 0.1:  # Low confidence threshold
            return "General Medicine", similarities.max(), []
        
        # Get most similar specialty
        best_match_idx = similarities.argmax()
        predicted_specialty = self.specialty_labels[best_match_idx]
        confidence = similarities.max()
        
        # Extract key symptoms
        extracted_symptoms = []
        symptom_words = symptom_text.lower().split()
        for symptom in self.symptom_corpus:
            if any(word in symptom for word in symptom_words):
                extracted_symptoms.append(symptom)
        
        return predicted_specialty, confidence, extracted_symptoms

    def get_urgency_level(self, symptoms):
        """AI urgency assessment"""
        emergency_keywords = [
            'severe', 'unconscious', 'bleeding', 'chest pain', 'difficulty breathing',
            'sudden', 'sharp pain', 'emergency', 'critical', 'stroke', 'heart attack',
            'paralysis', 'severe headache', 'high fever', 'poisoning', 'major injury'
        ]
        
        symptom_lower = symptoms.lower()
        emergency_count = sum(1 for keyword in emergency_keywords if keyword in symptom_lower)
        
        if emergency_count >= 2:
            return "HIGH", "🚨 Seek immediate medical attention - This could be life-threatening"
        elif emergency_count >= 1:
            return "MEDIUM", "⚠️ Consult doctor within 24 hours"
        else:
            return "LOW", "✅ Schedule routine appointment within a week"

    def get_gemini_analysis(self, symptom_text, patient_info=None):
        """Enhanced AI analysis using Gemini API"""
        if not self.gemini_model or not self.gemini_available:
            return None
        
        try:
            # Create comprehensive prompt for medical analysis
            prompt = f"""You are an AI medical assistant with expertise in symptom analysis. 
            Analyze these symptoms and provide structured, helpful insights:

PATIENT SYMPTOMS:
{symptom_text}

ADDITIONAL PATIENT INFO:
{patient_info if patient_info else 'No additional information provided'}

Please provide a clear, concise analysis in the following format:

KEY FINDINGS:
- [List 3-4 key observations about the symptoms]

POSSIBLE CONDITIONS (in order of likelihood):
1. [Most likely condition] - [Brief reason]
2. [Second likely condition] - [Brief reason]
3. [Third likely condition] - [Brief reason]

URGENCY LEVEL:
[LOW/MEDIUM/HIGH] - [Brief explanation]

RECOMMENDED ACTIONS:
1. [Immediate action to take]
2. [Follow-up action]
3. [When to seek medical care]

IMPORTANT NOTES:
- [Any warnings or precautions]
- [Red flags to watch for]
- [General health advice]

Focus on:
- Medical accuracy and safety
- Clear, actionable advice
- Appropriate urgency assessment
- When to seek professional medical help
- Avoid providing definitive diagnoses (suggest possibilities instead)
"""
            
            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                return None
                
        except Exception as e:
            print(f"Gemini API generation error: {str(e)[:200]}")
            return None

    def enhanced_symptom_analysis(self, symptom_text, patient_info=None):
        """Combine local ML with Gemini AI for best results"""
        
        # Run local analysis first
        local_analysis = self.symptom_analysis_agent(symptom_text)
        
        # Get Gemini AI analysis if available
        gemini_analysis = None
        if self.gemini_model and symptom_text.strip():
            gemini_analysis = self.get_gemini_analysis(symptom_text, patient_info)
        
        # Combine results
        combined_analysis = {
            "local_analysis": local_analysis,
            "ai_analysis": gemini_analysis,
            "has_gemini": gemini_analysis is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        return combined_analysis

    def symptom_analysis_agent(self, symptom_text):
        """Symptom Analysis Agent: Extracts and categorizes symptoms"""
        analysis = {
            'symptoms_found': [],
            'body_systems_affected': [],
            'severity_assessment': 'unknown',
            'duration_pattern': 'unknown',
            'trigger_factors': []
        }
        
        if not symptom_text or not symptom_text.strip():
            return analysis
        
        symptom_lower = symptom_text.lower()
        
        # Extract symptoms from database
        for specialty, symptoms in self.symptom_database.items():
            for symptom in symptoms:
                if symptom in symptom_lower or any(word in symptom_lower for word in symptom.split()):
                    if symptom not in analysis['symptoms_found']:
                        analysis['symptoms_found'].append(symptom)
                    if specialty not in analysis['body_systems_affected']:
                        analysis['body_systems_affected'].append(specialty)
        
        # Severity detection
        severity_keywords = {
            'mild': ['mild', 'slight', 'minor', 'manageable', 'low', 'bearable', 'tolerable'],
            'moderate': ['moderate', 'medium', 'bothersome', 'uncomfortable', 'annoying', 'troublesome'],
            'severe': ['severe', 'intense', 'unbearable', 'excruciating', 'extreme', 'sharp', 'stabbing', 
                      'throbbing', 'debilitating', 'crippling', 'agonizing']
        }
        
        # Check for severity keywords in the text
        for level, keywords in severity_keywords.items():
            for keyword in keywords:
                if keyword in symptom_lower:
                    analysis['severity_assessment'] = level
                    break
            if analysis['severity_assessment'] != 'unknown':
                break
        
        # If no severity found, infer from symptoms
        if analysis['severity_assessment'] == 'unknown':
            severe_symptoms = ['chest pain', 'shortness of breath', 'severe headache', 'paralysis', 
                             'unconscious', 'bleeding', 'high fever', 'vomiting blood']
            moderate_symptoms = ['fever', 'cough', 'diarrhea', 'vomiting', 'headache', 'body aches']
            
            # Check for severe symptoms
            if any(severe_symptom in symptom_lower for severe_symptom in severe_symptoms):
                analysis['severity_assessment'] = 'severe'
            # Check for moderate symptoms
            elif any(moderate_symptom in symptom_lower for moderate_symptom in moderate_symptoms):
                analysis['severity_assessment'] = 'moderate'
            elif analysis['symptoms_found']:
                analysis['severity_assessment'] = 'mild'
        
        # Duration pattern detection
        time_patterns = {
            'morning': ['morning', 'wake up', 'upon waking', 'after sleep', 'dawn', 'breakfast'],
            'evening': ['evening', 'night', 'bedtime', 'after work', 'dinner', 'sunset'],
            'afternoon': ['afternoon', 'midday', 'lunch', 'post-lunch'],
            'nocturnal': ['night', 'sleep', 'midnight', 'nocturnal', 'wake up at night', 'can\'t sleep'],
            'constant': ['constant', 'all the time', 'continuous', 'non-stop', 'persistent', 'ongoing'],
            'intermittent': ['intermittent', 'comes and goes', 'on and off', 'occasional', 'sporadic'],
            'postprandial': ['after eating', 'after meal', 'postprandial', 'after food', 'after dinner'],
            'with_activity': ['with activity', 'when walking', 'during exercise', 'when moving', 'physical exertion']
        }
        
        for pattern, keywords in time_patterns.items():
            for keyword in keywords:
                if keyword in symptom_lower:
                    analysis['duration_pattern'] = pattern
                    break
            if analysis['duration_pattern'] != 'unknown':
                break
        
        # Additional trigger detection
        triggers = {
            'stress': ['stress', 'anxiety', 'worry', 'tension', 'pressure'],
            'exercise': ['exercise', 'physical activity', 'workout', 'running', 'walking'],
            'cold_weather': ['cold', 'weather', 'winter', 'chilly', 'temperature'],
            'allergens': ['allergy', 'pollen', 'dust', 'mold', 'pet dander', 'allergic'],
            'food': ['after eating', 'food', 'meal', 'spicy', 'dairy', 'gluten', 'certain foods'],
            'position': ['lying down', 'sitting', 'standing', 'bending', 'position']
        }
        
        for trigger, keywords in triggers.items():
            for keyword in keywords:
                if keyword in symptom_lower:
                    analysis['trigger_factors'].append(trigger)
                    break
        
        return analysis

    def diagnosis_agent(self, symptoms, symptom_analysis, body_systems_affected):
        """Diagnosis Agent: Suggests possible diagnoses that match affected body systems"""
        possible_diagnoses = []
        
        # First, filter diagnoses by affected body systems
        system_relevant_diagnoses = []
        for condition, data in self.diagnosis_database.items():
            condition_systems = data.get('body_systems', [])
            # Check if condition matches ANY affected body system
            system_match = False
            for system in body_systems_affected:
                if system in condition_systems:
                    system_match = True
                    break
            
            # Only include diagnoses that match at least one affected body system
            if system_match:
                system_relevant_diagnoses.append(condition)
        
        # If no system matches, show only the diagnoses with highest symptom match
        if not system_relevant_diagnoses:
            # Fall back to all diagnoses but with lower priority
            system_relevant_diagnoses = list(self.diagnosis_database.keys())
        
        # Now calculate matches for relevant diagnoses
        for condition in system_relevant_diagnoses:
            data = self.diagnosis_database[condition]
            symptom_match_count = 0
            total_symptoms = len(data['symptoms'])
            
            for symptom in data['symptoms']:
                if symptom in ' '.join(symptoms).lower() or any(symptom in s for s in symptoms):
                    symptom_match_count += 1
            
            # Calculate match percentage
            if symptom_match_count > 0:
                match_percentage = (symptom_match_count / total_symptoms) * 100
                
                # Check if condition matches affected body systems
                condition_systems = data.get('body_systems', [])
                system_match_boost = 0
                for system in body_systems_affected:
                    if system in condition_systems:
                        system_match_boost += 25  # 25% boost for each matching system
                
                adjusted_match_percentage = min(100, match_percentage + system_match_boost)
                
                # Only show diagnoses with significant match
                if adjusted_match_percentage >= 40:  # Threshold for consideration
                    possible_diagnoses.append({
                        'condition': condition,
                        'match_percentage': adjusted_match_percentage,
                        'original_match': match_percentage,
                        'system_boost': system_match_boost,
                        'urgency': data['urgency'],
                        'specialty': data['specialty'],
                        'body_systems': condition_systems,
                        'matching_symptoms': symptom_match_count,
                        'total_symptoms': total_symptoms,
                        'matches_affected_systems': system_match_boost > 0,
                        'description': data.get('description', ''),
                        'common_in': data.get('common_in', ''),
                        'complications': data.get('complications', [])
                    })
        
        # Sort by match percentage
        possible_diagnoses.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        # Add confidence levels
        for diagnosis in possible_diagnoses:
            if diagnosis['match_percentage'] >= 80:
                diagnosis['confidence'] = 'High'
            elif diagnosis['match_percentage'] >= 60:
                diagnosis['confidence'] = 'Medium'
            else:
                diagnosis['confidence'] = 'Low'
        
        return possible_diagnoses

    def get_diagnosis_explanation(self, diagnosis, body_systems_affected, symptoms_found):
        """Generate explanation for why a diagnosis was selected"""
        explanation = []
        
        # Match with affected body systems
        matching_systems = set(diagnosis['body_systems']) & set(body_systems_affected)
        if matching_systems:
            explanation.append(f"✅ Matches affected body systems: {', '.join(matching_systems)}")
        else:
            explanation.append("⚠️ Does not match affected body systems (showing due to high symptom match)")
        
        # Match with symptoms
        explanation.append(f"✅ {diagnosis['matching_symptoms']} out of {diagnosis['total_symptoms']} key symptoms match")
        
        # System boost explanation
        if diagnosis['system_boost'] > 0:
            explanation.append(f"📈 +{diagnosis['system_boost']}% match boost for system alignment")
        
        # Urgency level
        explanation.append(f"⚠️ Urgency level: {diagnosis['urgency'].upper()}")
        
        return explanation

    def get_enhanced_symptom_based_medicines(self, symptoms, severity_level="moderate"):
        """Get personalized medicine recommendations based on specific symptoms with severity"""
        symptom_recommendations = []
        symptom_categories = {}
        
        # Group symptoms by category
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            found_category = None
            
            # Check each symptom category
            for category in self.symptom_medicine_database.keys():
                # Better matching logic
                if category in symptom_lower or symptom_lower in category:
                    found_category = category
                    break
                # Also check if symptom contains category keywords
                elif any(word in symptom_lower for word in category.split()):
                    found_category = category
                    break
            
            if found_category:
                if found_category not in symptom_categories:
                    symptom_categories[found_category] = []
                if symptom not in symptom_categories[found_category]:
                    symptom_categories[found_category].append(symptom)
        
        # Generate recommendations for each symptom category
        for category, matching_symptoms in symptom_categories.items():
            if category in self.symptom_medicine_database:
                category_data = self.symptom_medicine_database[category]
                
                # Determine severity level for this category
                if severity_level in category_data:
                    medicines = category_data[severity_level]
                elif 'moderate' in category_data:
                    medicines = category_data['moderate']
                elif 'mild' in category_data:
                    medicines = category_data['mild']
                elif 'severe' in category_data:
                    medicines = category_data['severe']
                else:
                    continue
                
                # Format recommendations for each medicine
                for medicine in medicines:
                    # Show which symptoms this medicine addresses
                    symptom_str = ", ".join([s.title() for s in matching_symptoms[:2]])
                    if len(matching_symptoms) > 2:
                        symptom_str += f" and {len(matching_symptoms) - 2} more"
                    
                    symptom_recommendations.append({
                        'symptoms': matching_symptoms,
                        'category': category,
                        'medicine': medicine,
                        'display': f"**{category.title()}** ({symptom_str}): {medicine}"
                    })
        
        # Sort by number of symptoms addressed (most first)
        symptom_recommendations.sort(key=lambda x: len(x['symptoms']), reverse=True)
        
        # Convert to display format
        display_recommendations = [rec['display'] for rec in symptom_recommendations[:15]]
        
        # Get unique symptoms list
        unique_symptoms = []
        for rec in symptom_recommendations:
            for symptom in rec['symptoms']:
                if symptom.lower() not in unique_symptoms:
                    unique_symptoms.append(symptom.lower())
        
        return display_recommendations, unique_symptoms

    def treatment_advisor_agent(self, diagnosis, patient_profile=None):
        """Treatment Advisor Agent: Provides treatment recommendations"""
        if diagnosis not in self.treatment_database:
            return None
        
        treatment_plan = self.treatment_database[diagnosis].copy()
        
        # Personalize based on patient profile if available
        if patient_profile:
            if patient_profile.get('age', 0) < 18:
                if 'medications' in treatment_plan:
                    # Add pediatric dosing notes
                    for category, meds in treatment_plan['medications'].items():
                        treatment_plan['medications'][category] = [f"{med} (consult pediatrician for dosage)" for med in meds]
            
            if patient_profile.get('has_allergies', False):
                treatment_plan['precautions'].append("Inform doctor about all allergies before taking any medication")
            
            # Get symptom-based medicines separately
            if patient_profile.get('symptoms'):
                symptoms = patient_profile['symptoms']
                severity = patient_profile.get('severity', 'moderate')
                symptom_based_medicines, unique_symptoms = self.get_enhanced_symptom_based_medicines(symptoms, severity)
                
                if symptom_based_medicines:
                    treatment_plan['symptom_specific_medicines'] = symptom_based_medicines
                    treatment_plan['detected_symptoms'] = [s.title() for s in unique_symptoms]
                    treatment_plan['symptom_severity'] = severity
    
        # Add general advice
        if 'general_advice' not in treatment_plan:
            treatment_plan['general_advice'] = [
                "Follow medication schedule strictly",
                "Monitor symptoms daily",
                "Keep a symptom diary",
                "Stay hydrated",
                "Get adequate rest"
            ]
        
        return treatment_plan

class MultiLanguageSupport:
    def __init__(self):
        self.languages = {
            'english': self.english_translations(),
            'telugu': self.telugu_translations(),
            'hindi': self.hindi_translations()
        }
    
    def english_translations(self):
        return {
            'welcome': '🏥 AI Healthcare Network',
            'login': 'Login',
            'search_hospitals': 'Search Hospitals',
            'symptom_checker': 'AI Symptom Checker',
            'emergency': 'Emergency',
            'schedule_visit': 'Schedule Visit',
            'patient_reviews': 'Patient Reviews',
            'logout': 'Logout',
            'symptom_analysis': 'Symptom Analysis',
            'diagnosis': 'Diagnosis',
            'treatment': 'Treatment'
        }
    
    def telugu_translations(self):
        return {
            'welcome': '🏥 AI ఆరోగ్య సేవా వలయం',
            'login': 'లాగిన్',
            'search_hospitals': 'హాస్పిటల్స్ వెతకండి',
            'symptom_checker': 'AI లక్షణాల చెకర్',
            'emergency': 'అత్యవసర',
            'schedule_visit': 'విజిట్ షెడ్యూల్ చేయండి',
            'patient_reviews': 'రోగుల సమీక్షలు',
            'logout': 'లాగ్అవట్',
            'symptom_analysis': 'లక్షణాల విశ్లేషణ',
            'diagnosis': 'నిర్ధారణ',
            'treatment': 'చికిత్స'
        }
    
    def hindi_translations(self):
        return {
            'welcome': '🏥 AI स्वास्थ्य सेवा नेटवर्क',
            'login': 'लॉगिन',
            'search_hospitals': 'अस्पताल खोजें',
            'symptom_checker': 'AI लक्षण जांच',
            'emergency': 'आपातकाल',
            'schedule_visit': 'यात्रा निर्धारित करें',
            'patient_reviews': 'मरीज़ों की समीक्षाएं',
            'logout': 'लॉगआउट',
            'symptom_analysis': 'लक्षण विश्लेषण',
            'diagnosis': 'निदान',
            'treatment': 'उपचार'
        }
    
    def get_translation(self, language, key):
        return self.languages.get(language, self.english_translations()).get(key, key)

class PatientReviewSystem:
    def __init__(self):
        self.reviews_file = "patient_reviews.json"
        self.reviews = self.load_reviews()
    
    def load_reviews(self):
        """Load patient reviews from file"""
        try:
            with open(self.reviews_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def save_reviews(self):
        """Save reviews to file"""
        try:
            with open(self.reviews_file, 'w') as f:
                json.dump(self.reviews, f, indent=2)
        except Exception as e:
            print(f"Error saving reviews: {e}")
    
    def add_review(self, hospital_id, patient_name, rating, comment):
        """Add a new patient review with sentiment analysis"""
        if hospital_id not in self.reviews:
            self.reviews[hospital_id] = []
        
        # AI Sentiment Analysis
        sentiment_score = self.analyze_sentiment(comment)
        sentiment = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        
        review = {
            'id': secrets.token_hex(8),
            'patient_name': patient_name,
            'rating': rating,
            'comment': comment,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.reviews[hospital_id].append(review)
        self.save_reviews()
        return review
    
    def analyze_sentiment(self, text):
        """AI-powered sentiment analysis"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def get_hospital_reviews(self, hospital_id):
        """Get all reviews for a hospital"""
        return self.reviews.get(hospital_id, [])
    
    def get_hospital_rating_stats(self, hospital_id):
        """Calculate rating statistics for a hospital"""
        reviews = self.get_hospital_reviews(hospital_id)
        if not reviews:
            return 0.0, 0
        
        ratings = [review['rating'] for review in reviews]
        return np.mean(ratings), len(reviews)
    
    def get_all_reviews(self):
        """Get all reviews from all hospitals"""
        all_reviews = []
        for hospital_id, reviews_list in self.reviews.items():
            for review in reviews_list:
                review_with_hospital = review.copy()
                review_with_hospital['hospital_id'] = hospital_id
                all_reviews.append(review_with_hospital)
        
        # Sort by timestamp, newest first
        return sorted(all_reviews, key=lambda x: x['timestamp'], reverse=True)

# Security features
class SecurityManager:
    def __init__(self):
        self.session_timeout = 1800  # 30 minutes
        
    def validate_input(self, input_string, input_type='text'):
        """Prevent SQL injection and XSS attacks"""
        if input_type == 'phone':
            return bool(re.match(r'^[\d\s\-\+\(\)]{10,15}$', input_string))
        elif input_type == 'email':
            return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', input_string))
        else:
            # Remove potentially dangerous characters
            cleaned = re.sub(r'[<>"\']', '', input_string)
            return cleaned
    
    def generate_session_token(self):
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def hash_password(self, password):
        """Hash password for security"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return self.hash_password(password) == hashed

class HealthcareSystem:
    def __init__(self):
        self.security = SecurityManager()
        self.ai_symptom_checker = AISymptomChecker()
        self.language_support = MultiLanguageSupport()
        self.review_system = PatientReviewSystem()
        self.hospitals = self.create_telangana_hospital_data()
        self.user_sessions = {}
        # API Keys
        self.gemini_api_key = GEMINI_API_KEY
        self.openai_api_key = OPENAI_API_KEY
        self.google_maps_api_key = GOOGLE_MAPS_API_KEY
        self.weather_api_key = WEATHER_API_KEY
        self.twilio_api_key = TWILIO_API_KEY
        
        # Predefined user credentials for demo
        self.users = {
            'patient': {'password': self.security.hash_password('patient123'), 'role': 'patient'},
            'doctor': {'password': self.security.hash_password('doctor123'), 'role': 'doctor'},
            'admin': {'password': self.security.hash_password('admin123'), 'role': 'admin'}
        }
    
    def create_telangana_hospital_data(self):
        """Create comprehensive Telangana hospital data with proper coordinates"""
        hospitals = [
            # Hyderabad Hospitals
            {
                'id': 'HYP001',
                'name': 'Yashoda Hospitals - Somajiguda',
                'address': 'Raj Bhavan Road, Somajiguda',
                'city': 'Hyderabad',
                'state': 'Telangana',
                'phone': '+91-40-4567 4567',
                'emergency_phone': '+91-40-4567 4567',
                'email': 'info@yashodahospitals.com',
                'website': 'https://www.yashodahospitals.com',
                'specialties': 'Cardiology, Cardiac Surgery, Neurology, Oncology, Orthopedics, Emergency Medicine, ICU, General Checkup, Pediatrics, Surgery',
                'beds_total': 400,
                'beds_available': 67,
                'icu_beds': 45,
                'ventilators': 35,
                'oxygen_beds': 120,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.6,
                'wait_time_emergency': 15,
                'consultation_fee': 800,
                'insurance_accepted': ['ICICI', 'HDFC', 'Star Health', 'Apollo Munich'],
                'latitude': 17.4254,
                'longitude': 78.4494,
                'timings': '24/7',
                'doctors_count': 85,
                'accreditation': 'NABH, NABL',
                'parking_available': True,
                'cafeteria': True
            },
            {
                'id': 'HYP002',
                'name': 'Apollo Hospitals - Jubilee Hills',
                'address': 'Film Nagar, Jubilee Hills',
                'city': 'Hyderabad',
                'state': 'Telangana',
                'phone': '+91-40-2360 7777',
                'emergency_phone': '+91-40-2360 7777',
                'email': 'info.hyderabad@apollohospitals.com',
                'website': 'https://www.apollohospitals.com',
                'specialties': 'Organ Transplant, Neurosurgery, Cardiology, Oncology, Pediatrics, General Checkup, Emergency Care, Surgery, ICU Care',
                'beds_total': 350,
                'beds_available': 42,
                'icu_beds': 38,
                'ventilators': 30,
                'oxygen_beds': 95,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.7,
                'wait_time_emergency': 12,
                'consultation_fee': 1000,
                'insurance_accepted': ['All major insurers'],
                'latitude': 17.4332,
                'longitude': 78.4073,
                'timings': '24/7',
                'doctors_count': 92,
                'accreditation': 'JCI, NABH',
                'parking_available': True,
                'cafeteria': True
            },
            {
                'id': 'HYP003',
                'name': 'KIMS Hospitals - Secunderabad',
                'address': '1-8-31/1, Minister Road, Secunderabad',
                'city': 'Hyderabad',
                'state': 'Telangana',
                'phone': '+91-40-4488 5000',
                'emergency_phone': '+91-40-4488 5000',
                'email': 'info@kimshospitals.com',
                'website': 'https://www.kimshospitals.com',
                'specialties': 'Cardiac Sciences, Neurology, Gastroenterology, Nephrology, Urology, General Checkup, Emergency Care, Orthopedics, Surgery',
                'beds_total': 300,
                'beds_available': 38,
                'icu_beds': 32,
                'ventilators': 25,
                'oxygen_beds': 80,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.5,
                'wait_time_emergency': 18,
                'consultation_fee': 700,
                'insurance_accepted': ['ICICI', 'Bajaj', 'Star Health'],
                'latitude': 17.4399,
                'longitude': 78.4983,
                'timings': '24/7',
                'doctors_count': 78,
                'accreditation': 'NABH',
                'parking_available': True,
                'cafeteria': True
            },
            # Warangal Hospitals
            {
                'id': 'WGL001',
                'name': 'Medicover Hospitals - Warangal',
                'address': 'Hanamkonda, Warangal',
                'city': 'Warangal',
                'state': 'Telangana',
                'phone': '+91-870-2567 890',
                'emergency_phone': '+91-870-2567 891',
                'email': 'warangal@medicoverhospitals.com',
                'website': 'https://www.medicoverhospitals.com',
                'specialties': 'Multi-specialty, Emergency Care, ICU, Surgery, Cardiology, Neurology, Orthopedics, Pediatrics, General Checkup, Oncology',
                'beds_total': 150,
                'beds_available': 25,
                'icu_beds': 15,
                'ventilators': 12,
                'oxygen_beds': 45,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.2,
                'wait_time_emergency': 25,
                'consultation_fee': 500,
                'insurance_accepted': ['Star Health', 'HDFC', 'ICICI'],
                'latitude': 17.9784,
                'longitude': 79.5941,
                'timings': '24/7',
                'doctors_count': 35,
                'accreditation': 'NABH',
                'parking_available': True,
                'cafeteria': True
            },
            {
                'id': 'WGL002',
                'name': 'Kakatiya Medical College - Warangal',
                'address': 'MGM Hospital Campus, Warangal',
                'city': 'Warangal',
                'state': 'Telangana',
                'phone': '+91-870-2456 789',
                'emergency_phone': '+91-870-2456 790',
                'email': 'kmcwgl@telangana.gov.in',
                'website': 'https://www.kmcwgl.gov.in',
                'specialties': 'Multi-specialty, Medical College, Super Specialty, Research, Emergency Medicine, Surgery, Pediatrics, Orthopedics, Neurology, Cardiology, ICU Care, Oncology',
                'beds_total': 800,
                'beds_available': 120,
                'icu_beds': 45,
                'ventilators': 30,
                'oxygen_beds': 200,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.3,
                'wait_time_emergency': 20,
                'consultation_fee': 100,
                'insurance_accepted': ['All Government Schemes', 'Star Health'],
                'latitude': 17.9924,
                'longitude': 79.5930,
                'timings': '24/7',
                'doctors_count': 150,
                'accreditation': 'NABH, MCI',
                'parking_available': True,
                'cafeteria': True
            },
            # Khammam Hospitals
            {
                'id': 'KHM001',
                'name': 'Malla Reddy Hospital - Khammam',
                'address': 'Vidya Nagar, Khammam',
                'city': 'Khammam',
                'state': 'Telangana',
                'phone': '+91-874-2256 789',
                'emergency_phone': '+91-874-2256 790',
                'email': 'khammam@mallareddyhospital.com',
                'website': 'https://www.mallareddyhospital.com',
                'specialties': 'General Medicine, Surgery, Pediatrics, Gynecology, Orthopedics, Emergency Care, General Checkup',
                'beds_total': 120,
                'beds_available': 18,
                'icu_beds': 10,
                'ventilators': 8,
                'oxygen_beds': 35,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': False,
                'rating': 4.0,
                'wait_time_emergency': 30,
                'consultation_fee': 300,
                'insurance_accepted': ['Star Health', 'HDFC'],
                'latitude': 17.2473,
                'longitude': 80.1514,
                'timings': '24/7',
                'doctors_count': 22,
                'accreditation': 'State Accreditation',
                'parking_available': True,
                'cafeteria': False
            },
            {
                'id': 'KHM002',
                'name': 'Khammam Government General Hospital',
                'address': 'Kothagudem Road, Khammam',
                'city': 'Khammam',
                'state': 'Telangana',
                'phone': '+91-874-2222 123',
                'emergency_phone': '+91-874-2222 124',
                'email': 'khammam.gh@telangana.gov.in',
                'website': 'https://www.telanganahealth.gov.in',
                'specialties': 'Multi-specialty, Emergency Care, Surgery, Maternity, Pediatrics, ICU, General Checkup, Orthopedics, Neurology',
                'beds_total': 250,
                'beds_available': 45,
                'icu_beds': 20,
                'ventilators': 15,
                'oxygen_beds': 80,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.1,
                'wait_time_emergency': 25,
                'consultation_fee': 150,
                'insurance_accepted': ['All Government Schemes', 'Star Health'],
                'latitude': 17.2500,
                'longitude': 80.1500,
                'timings': '24/7',
                'doctors_count': 55,
                'accreditation': 'NABH',
                'parking_available': True,
                'cafeteria': True
            },
            # Nizamabad Hospitals
            {
                'id': 'NIZ001',
                'name': 'Nizamabad Government Hospital',
                'address': 'Hyderabad Road, Nizamabad',
                'city': 'Nizamabad',
                'state': 'Telangana',
                'phone': '+91-846-2245 456',
                'emergency_phone': '+91-846-2245 457',
                'email': 'nizamabad.gh@telangana.gov.in',
                'website': 'https://www.telanganahealth.gov.in',
                'specialties': 'Multi-specialty, Emergency Care, Surgery, Maternity, Pediatrics, General Checkup, Orthopedics, ICU Care',
                'beds_total': 180,
                'beds_available': 32,
                'icu_beds': 12,
                'ventilators': 8,
                'oxygen_beds': 50,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.0,
                'wait_time_emergency': 28,
                'consultation_fee': 280,
                'insurance_accepted': ['All Government Schemes', 'Star Health'],
                'latitude': 18.6735,
                'longitude': 78.0940,
                'timings': '24/7',
                'doctors_count': 38,
                'accreditation': 'NABH',
                'parking_available': True,
                'cafeteria': True
            },
            {
                'id': 'NIZ002',
                'name': 'Nizamabad Multi-Specialty Center',
                'address': 'Armoor Road, Nizamabad',
                'city': 'Nizamabad',
                'state': 'Telangana',
                'phone': '+91-846-2256 123',
                'emergency_phone': '+91-846-2256 124',
                'email': 'info@nizamabadhospital.com',
                'website': 'https://www.nizamabadhospital.com',
                'specialties': 'General Medicine, Surgery, Orthopedics, Pediatrics, Emergency Care, General Checkup, Neurology',
                'beds_total': 120,
                'beds_available': 25,
                'icu_beds': 8,
                'ventilators': 5,
                'oxygen_beds': 35,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': False,
                'rating': 3.9,
                'wait_time_emergency': 30,
                'consultation_fee': 350,
                'insurance_accepted': ['Star Health', 'HDFC'],
                'latitude': 18.6750,
                'longitude': 78.0900,
                'timings': '24/7',
                'doctors_count': 25,
                'accreditation': 'State Accreditation',
                'parking_available': True,
                'cafeteria': False
            },
            # Karimnagar Hospitals
            {
                'id': 'KAR001',
                'name': 'Karimnagar Government Hospital',
                'address': 'Collectorate Road, Karimnagar',
                'city': 'Karimnagar',
                'state': 'Telangana',
                'phone': '+91-878-2245 123',
                'emergency_phone': '+91-878-2245 124',
                'email': 'karimnagar.gh@telangana.gov.in',
                'website': 'https://www.telanganahealth.gov.in',
                'specialties': 'Multi-specialty, Emergency Care, Surgery, Pediatrics, Gynecology, General Checkup, Orthopedics, ICU Care',
                'beds_total': 200,
                'beds_available': 35,
                'icu_beds': 15,
                'ventilators': 10,
                'oxygen_beds': 60,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.1,
                'wait_time_emergency': 25,
                'consultation_fee': 300,
                'insurance_accepted': ['All Government Schemes', 'Star Health', 'ICICI'],
                'latitude': 18.4386,
                'longitude': 79.1288,
                'timings': '24/7',
                'doctors_count': 45,
                'accreditation': 'NABH',
                'parking_available': True,
                'cafeteria': True
            },
            {
                'id': 'KAR002',
                'name': 'Apollo Clinic - Karimnagar',
                'address': 'Huzurabad Road, Karimnagar',
                'city': 'Karimnagar',
                'state': 'Telangana',
                'phone': '+91-878-2256 789',
                'emergency_phone': '+91-878-2256 790',
                'email': 'karimnagar@apolloclinic.com',
                'website': 'https://www.apolloclinic.com',
                'specialties': 'General Medicine, Pediatrics, Cardiology, Orthopedics, Emergency Care, General Checkup, Neurology',
                'beds_total': 80,
                'beds_available': 15,
                'icu_beds': 6,
                'ventilators': 4,
                'oxygen_beds': 25,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': False,
                'rating': 4.2,
                'wait_time_emergency': 20,
                'consultation_fee': 500,
                'insurance_accepted': ['All major insurers'],
                'latitude': 18.4350,
                'longitude': 79.1250,
                'timings': '24/7',
                'doctors_count': 22,
                'accreditation': 'Apollo Standards',
                'parking_available': True,
                'cafeteria': True
            },
            # Jangaon Hospitals
            {
                'id': 'JAN001',
                'name': 'Jangaon District Hospital',
                'address': 'Station Road, Jangaon',
                'city': 'Jangaon',
                'state': 'Telangana',
                'phone': '+91-871-2245 678',
                'emergency_phone': '+91-871-2245 679',
                'email': 'jangaon.dh@telangana.gov.in',
                'website': 'https://www.telanganahealth.gov.in',
                'specialties': 'Multi-specialty, Emergency Care, Surgery, Maternity, Pediatrics, General Checkup, Orthopedics, ICU Care, Cardiology, Neurology',
                'beds_total': 150,
                'beds_available': 28,
                'icu_beds': 10,
                'ventilators': 6,
                'oxygen_beds': 45,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': True,
                'rating': 4.0,
                'wait_time_emergency': 30,
                'consultation_fee': 250,
                'insurance_accepted': ['All Government Schemes', 'Star Health'],
                'latitude': 17.7226,
                'longitude': 79.1800,
                'timings': '24/7',
                'doctors_count': 32,
                'accreditation': 'NABH',
                'parking_available': True,
                'cafeteria': True
            },
            {
                'id': 'JAN002',
                'name': 'Sri Sai Multi-Specialty Hospital - Jangaon',
                'address': 'Kazipet Road, Jangaon',
                'city': 'Jangaon',
                'state': 'Telangana',
                'phone': '+91-871-2256 789',
                'emergency_phone': '+91-871-2256 790',
                'email': 'srisai.jangaon@gmail.com',
                'website': 'https://www.srisaihospitals.com',
                'specialties': 'General Surgery, Orthopedics, Pediatrics, Gynecology, Emergency Care, General Checkup, Cardiology, Neurology',
                'beds_total': 100,
                'beds_available': 20,
                'icu_beds': 8,
                'ventilators': 5,
                'oxygen_beds': 30,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': False,
                'rating': 3.9,
                'wait_time_emergency': 28,
                'consultation_fee': 350,
                'insurance_accepted': ['Star Health', 'ICICI'],
                'latitude': 17.7280,
                'longitude': 79.1750,
                'timings': '24/7',
                'doctors_count': 18,
                'accreditation': 'State Accreditation',
                'parking_available': True,
                'cafeteria': False
            },
            # Gatkesar Hospitals
            {
                'id': 'GAT001',
                'name': 'Gatkesar Community Health Center',
                'address': 'Gatkesar Main Road, Near Bus Stand',
                'city': 'Gatkesar',
                'state': 'Telangana',
                'phone': '+91-841-2234 567',
                'emergency_phone': '+91-841-2234 568',
                'email': 'gatkesar.chc@telangana.gov.in',
                'website': 'https://www.telanganahealth.gov.in',
                'specialties': 'Primary Care, Emergency Medicine, Pediatrics, General Surgery, General Checkup, Emergency Care, Cardiology, Neurology',
                'beds_total': 80,
                'beds_available': 15,
                'icu_beds': 6,
                'ventilators': 4,
                'oxygen_beds': 25,
                'emergency_services': True,
                'ambulance_services': True,
                'blood_bank': False,
                'rating': 3.8,
                'wait_time_emergency': 35,
                'consultation_fee': 200,
                'insurance_accepted': ['Government Schemes'],
                'latitude': 17.3650,
                'longitude': 78.5850,
                'timings': '24/7',
                'doctors_count': 15,
                'accreditation': 'Government Certified',
                'parking_available': True,
                'cafeteria': False
            }
        ]
        return pd.DataFrame(hospitals)
    
    def get_city_coordinates(self, city_name):
        """Get default coordinates for each city"""
        city_coordinates = {
            'Hyderabad': (17.3850, 78.4867),
            'Warangal': (17.9784, 79.5941),
            'Khammam': (17.2473, 80.1514),
            'Nizamabad': (18.6735, 78.0940),
            'Karimnagar': (18.4386, 79.1288),
            'Jangaon': (17.7226, 79.1800),
            'Gatkesar': (17.3650, 78.5850)
        }
        return city_coordinates.get(city_name, (17.3850, 78.4867))
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate approximate distance between coordinates"""
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111
    
    def find_nearby_hospitals(self, user_lat, user_lng, radius_km=50):
        """Find hospitals within specified radius"""
        nearby_hospitals = []
        for _, hospital in self.hospitals.iterrows():
            distance = self.calculate_distance(
                user_lat, user_lng, 
                hospital['latitude'], hospital['longitude']
            )
            if distance <= radius_km:
                hospital_data = hospital.copy()
                hospital_data['distance_km'] = round(distance, 2)
                nearby_hospitals.append(hospital_data)
        
        nearby_hospitals.sort(key=lambda x: x['distance_km'])
        return nearby_hospitals
    
    def get_hospitals_by_city(self, city_name):
        """Get hospitals by city name"""
        return self.hospitals[self.hospitals['city'] == city_name]
    
    def get_hospital_by_id(self, hospital_id):
        """Get hospital details by ID"""
        hospital = self.hospitals[self.hospitals['id'] == hospital_id]
        return hospital.iloc[0] if not hospital.empty else None

    def filter_hospitals_by_specialty(self, specialty, city=None):
        """Filter hospitals by medical specialty - ENSURES HOSPITALS ARE ALWAYS FOUND"""
        try:
            if city and city != "All Cities":
                hospitals = self.get_hospitals_by_city(city)
            else:
                hospitals = self.hospitals.copy()
            
            # Map specialty keywords to search terms
            specialty_mapping = {
                'General Checkup': ['General Checkup', 'General Medicine', 'Primary Care', 'Multi-specialty'],
                'Emergency Care': ['Emergency Care', 'Emergency Medicine', 'Emergency'],
                'Cardiology': ['Cardiology', 'Cardiac', 'Heart'],
                'Neurology': ['Neurology', 'Neuro', 'Brain'],
                'Orthopedics': ['Orthopedics', 'Ortho', 'Bone'],
                'Pediatrics': ['Pediatrics', 'Children', 'Child'],
                'Oncology': ['Oncology', 'Cancer', 'Tumor'],
                'Surgery': ['Surgery', 'Surgical'],
                'ICU Care': ['ICU', 'Intensive Care', 'Critical Care']
            }
            
            search_terms = specialty_mapping.get(specialty, [specialty])
            
            filtered_hospitals = []
            for _, hospital in hospitals.iterrows():
                hospital_specialties = hospital['specialties']
                
                # COMPREHENSIVE NULL CHECK
                if pd.isna(hospital_specialties) or hospital_specialties is None or hospital_specialties == '':
                    continue
                    
                # Convert to string and handle any remaining None values
                specialties_str = str(hospital_specialties) if hospital_specialties is not None else ""
                
                for term in search_terms:
                    if term.lower() in specialties_str.lower():
                        filtered_hospitals.append(hospital)
                        break
            
            # If no hospitals found for specialty, return ALL hospitals in the city
            if len(filtered_hospitals) == 0:
                st.info(f"💡 **Note**: No hospitals found specifically for {specialty} in {city}. Showing all available hospitals in {city}.")
                if city and city != "All Cities":
                    return self.get_hospitals_by_city(city)
                else:
                    return self.hospitals.copy()
            
            return pd.DataFrame(filtered_hospitals)
        except Exception as e:
            st.error(f"Error filtering hospitals: {str(e)}")
            # Return all hospitals as fallback
            if city and city != "All Cities":
                return self.get_hospitals_by_city(city)
            else:
                return self.hospitals.copy()

def display_hospital_details(hospital, show_full_details=True, highlight=False):
    """Display comprehensive hospital information with optional highlighting"""
    if show_full_details:
        with st.container():
            # Highlight border if this is the target hospital
            border_color = "#ff4444" if highlight else "#2e86ab"
            border_width = "8px" if highlight else "5px"
            
            # Create facilities string
            facilities = []
            if hospital['ambulance_services']:
                facilities.append('🚑 Ambulance')
            if hospital['blood_bank']:
                facilities.append('💉 Blood Bank')
            if hospital['parking_available']:
                facilities.append('🅿️ Parking')
            if hospital['cafeteria']:
                facilities.append('☕ Cafeteria')
            facilities_str = ' '.join(facilities) if facilities else 'No additional facilities'
            
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: {border_width} solid {border_color}; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='color: #ff0000; margin-bottom: 1rem;'>{hospital['name']}</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; color: #000000;'>
                    <div style='color: #000000;'>
                        <strong style='color: #000000;'>📍 Address:</strong> <span style='color: #000000;'>{hospital['address']}, {hospital['city']}</span><br>
                        <strong style='color: #000000;'>📞 Main Phone:</strong> <span style='color: #000000;'>{hospital['phone']}</span><br>
                        <strong style='color: #000000;'>🚨 Emergency:</strong> <span style='color: #000000;'>{hospital['emergency_phone']}</span><br>
                        <strong style='color: #000000;'>⭐ Rating:</strong> <span style='color: #000000;'>{hospital['rating']}/5.0</span><br>
                        <strong style='color: #000000;'>🛏️ Available Beds:</strong> <span style='color: #000000;'>{hospital['beds_available']}</span><br>
                        <strong style='color: #000000;'>💨 Oxygen Beds:</strong> <span style='color: #000000;'>{hospital['oxygen_beds']}</span>
                    </div>
                    <div style='color: #000000;'>
                        <strong style='color: #000000;'>🏥 ICU Beds:</strong> <span style='color: #000000;'>{hospital['icu_beds']}</span><br>
                        <strong style='color: #000000;'>🫁 Ventilators:</strong> <span style='color: #000000;'>{hospital['ventilators']}</span><br>
                        <strong style='color: #000000;'>⏱️ ER Wait Time:</strong> <span style='color: #000000;'>{hospital['wait_time_emergency']} min</span><br>
                        <strong style='color: #000000;'>💰 Consultation Fee:</strong> <span style='color: #000000;'>₹{hospital['consultation_fee']}</span><br>
                        <strong style='color: #000000;'>🕒 Timings:</strong> <span style='color: #000000;'>{hospital['timings']}</span><br>
                        <strong style='color: #000000;'>🎯 Specialties:</strong> <span style='color: #000000;'>{hospital['specialties']}</span>
                    </div>
                </div>
                <div style='margin-top: 1rem; color: #000000;'>
                    <strong style='color: #000000;'>🏅 Accreditation:</strong> <span style='color: #000000;'>{hospital['accreditation']}</span><br>
                    <strong style='color: #000000;'>🩺 Doctors:</strong> <span style='color: #000000;'>{hospital['doctors_count']}</span><br>
                    <strong style='color: #000000;'>🏢 Facilities:</strong> <span style='color: #000000;'>{facilities_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_directions_map(hospital, user_lat, user_lng):
    """Show interactive directions map"""
    st.subheader(f"🗺️ Directions to {hospital['name']}")
    
    # Create a simple directions interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"""
        **Route Information:**
        - From: Your Location ({user_lat:.4f}, {user_lng:.4f})
        - To: {hospital['name']} ({hospital['latitude']:.4f}, {hospital['longitude']:.4f})
        - Distance: {hospital.get('distance_km', 'N/A')} km
        - Estimated Travel Time: {int(hospital.get('distance_km', 10) * 2)} minutes
        """)
        
        # Display coordinates
        st.write("**Coordinates:**")
        st.write(f"Destination: {hospital['latitude']:.6f}, {hospital['longitude']:.6f}")
        
    with col2:
        # Navigation options
        st.subheader("Navigation Options")
        
        # Google Maps link
        google_maps_url = f"https://www.google.com/maps/dir/{user_lat},{user_lng}/{hospital['latitude']},{hospital['longitude']}"
        st.markdown(f"""
        <a href="{google_maps_url}" target="_blank">
            <button style="background-color: #4285F4; color: white; padding: 10px 20px; border: none; border-radius: 5px; width: 100%; margin: 5px 0;">
            🗺️ Open Google Maps
            </button>
        </a>
        """, unsafe_allow_html=True)

def show_symptom_analysis_agent(healthcare_system, active_tab="Symptom Analysis"):
    """Symptom Analysis Agent Interface with Gemini AI"""
    st.subheader("🔍 Symptom Analysis Agent")
    
    with st.expander("📝 Enter Symptoms", expanded=True):
        # Get stored symptoms if available
        default_symptoms = st.session_state.get('symptom_text', '')
        
        symptoms = st.text_area(
            "Describe your symptoms in detail:",
            value=default_symptoms,
            placeholder="Example: I have had chest pain and shortness of breath since morning, worse with exercise...",
            height=150,
            key="symptom_analysis_input"
        )
        
        # Store symptoms in session
        if symptoms != default_symptoms:
            st.session_state.symptom_text = symptoms
        
        # Additional patient information for better analysis
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, 
                                 value=st.session_state.get('patient_age', 30),
                                 key="symptom_age")
            gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"],
                                 index=st.session_state.get('patient_gender_idx', 0),
                                 key="symptom_gender")
        
        with col2:
            symptom_duration = st.selectbox("Symptom Duration", 
                                           ["Select", "Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"],
                                           index=st.session_state.get('symptom_duration_idx', 0),
                                           key="symptom_duration")
            severity = st.selectbox("Severity", ["Select", "Mild", "Moderate", "Severe"],
                                   index=st.session_state.get('symptom_severity_idx', 0),
                                   key="symptom_severity")
        
        # Store patient info
        st.session_state.patient_age = age
        st.session_state.patient_gender_idx = ["Select", "Male", "Female", "Other"].index(gender)
        st.session_state.symptom_duration_idx = ["Select", "Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"].index(symptom_duration)
        st.session_state.symptom_severity_idx = ["Select", "Mild", "Moderate", "Severe"].index(severity)
        
        # Gemini AI Toggle
        use_gemini = st.checkbox("🤖 Enable Advanced AI Analysis (Gemini)", 
                                value=True,
                                help="Uses Google's Gemini AI for enhanced symptom analysis")
    
    # Only show the analyze button if this is the active tab
    if active_tab == "Symptom Analysis":
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔬 Analyze Symptoms with AI Agent", type="primary", key="analyze_symptoms_btn", use_container_width=True):
                if symptoms:
                    with st.spinner("🔍 Symptom Analysis Agent is analyzing..."):
                        # Create patient info dictionary for Gemini
                        patient_info = {
                            'age': age,
                            'gender': gender if gender != "Select" else "Not specified",
                            'duration': symptom_duration if symptom_duration != "Select" else "Not specified",
                            'severity': severity if severity != "Select" else "Not specified"
                        }
                        
                        # Run enhanced analysis with Gemini if enabled
                        if use_gemini and healthcare_system.ai_symptom_checker.gemini_available:
                            analysis = healthcare_system.ai_symptom_checker.enhanced_symptom_analysis(symptoms, patient_info)
                        else:
                            analysis = healthcare_system.ai_symptom_checker.symptom_analysis_agent(symptoms)
                            analysis = {
                                'local_analysis': analysis,
                                'ai_analysis': None,
                                'has_gemini': False
                            }
                        
                        # Store analysis results with body systems
                        st.session_state.symptom_analysis_data = {
                            'symptoms_text': symptoms,
                            'analysis': analysis,
                            'patient_info': patient_info,
                            'body_systems_affected': analysis['local_analysis']['body_systems_affected'] if 'local_analysis' in analysis else []
                        }
                        
                        # Also run diagnosis agent to get possible diagnoses - WITH BODY SYSTEMS
                        predicted_specialty, confidence, extracted_symptoms = healthcare_system.ai_symptom_checker.analyze_symptoms(symptoms)
                        possible_diagnoses = healthcare_system.ai_symptom_checker.diagnosis_agent(
                            extracted_symptoms, 
                            analysis['local_analysis'] if 'local_analysis' in analysis else {},
                            analysis['local_analysis']['body_systems_affected'] if 'local_analysis' in analysis else []
                        )
                        
                        # Store diagnosis data
                        st.session_state.diagnosis_data = {
                            'predicted_specialty': predicted_specialty,
                            'confidence': confidence,
                            'extracted_symptoms': extracted_symptoms,
                            'possible_diagnoses': possible_diagnoses
                        }
                        
                        # Set the top diagnosis automatically
                        if possible_diagnoses:
                            st.session_state.top_diagnosis = possible_diagnoses[0]['condition']
                            st.session_state.top_diagnosis_details = possible_diagnoses[0]
                        
                        # Display analysis results
                        st.success("✅ Symptom Analysis Complete!")
                        
                        # Display Gemini AI analysis if available
                        if analysis.get('ai_analysis'):
                            st.markdown("---")
                            st.subheader("🤖 Advanced AI Analysis (Gemini)")
                            with st.expander("View Detailed AI Analysis", expanded=True):
                                st.markdown(analysis['ai_analysis'])
                        
                        # Results in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Detected Symptoms")
                            local_analysis = analysis['local_analysis'] if 'local_analysis' in analysis else analysis
                            if local_analysis['symptoms_found']:
                                # Get unique symptoms (remove duplicates)
                                unique_symptoms = []
                                for symptom in local_analysis['symptoms_found']:
                                    if symptom not in unique_symptoms:
                                        unique_symptoms.append(symptom)
                                
                                for symptom in unique_symptoms:
                                    st.write(f"• {symptom.title()}")
                            else:
                                # Better symptom extraction from text
                                words = symptoms.lower().split()
                                common_symptoms = ['pain', 'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting', 'diarrhea', 'dizziness']
                                detected = [word for word in words if word in common_symptoms]
                                if detected:
                                    st.info("Found these common symptoms in your description:")
                                    for symptom in set(detected):
                                        st.write(f"• {symptom.title()}")
                                else:
                                    st.info("No specific symptoms detected in database. Try being more descriptive.")
                            
                            st.subheader("🧬 Body Systems Affected")
                            if local_analysis['body_systems_affected']:
                                for system in local_analysis['body_systems_affected']:
                                    st.write(f"• {system.title()} ⭐")
                            else:
                                st.info("Could not identify affected body systems")
                        
                        with col2:
                            st.subheader("📊 Analysis Details")
                            st.write(f"**Severity Assessment:** {local_analysis['severity_assessment'].title()}")
                            st.write(f"**Duration Pattern:** {local_analysis['duration_pattern'].title()}")
                            
                            if local_analysis.get('trigger_factors'):
                                st.subheader("🎯 Trigger Factors")
                                for trigger in local_analysis['trigger_factors']:
                                    st.write(f"• {trigger.replace('_', ' ').title()}")
                        
                        # Show preliminary diagnosis hint
                        if possible_diagnoses:
                            top_diag = possible_diagnoses[0]
                            system_matches = set(top_diag['body_systems']) & set(local_analysis['body_systems_affected'])
                            
                            st.markdown("---")
                            if system_matches:
                                st.info(f"**🔍 Preliminary Finding:** The top matching diagnosis ({top_diag['condition']}) aligns with {len(system_matches)} of your affected body systems")
                            else:
                                st.warning(f"**🔍 Preliminary Finding:** Top diagnosis ({top_diag['condition']}) doesn't match your affected body systems - showing due to high symptom match")
                        
                        # Navigation to Diagnosis Agent
                        st.markdown("---")
                        st.success("🔜 **Next Step:** Switch to the '🩺 Diagnosis Agent' tab to see detailed diagnosis analysis based on your symptoms!")
                else:
                    st.warning("Please describe your symptoms for analysis")
        
        with col2:
            if st.button("🔄 Clear Analysis", type="secondary", key="clear_analysis_btn", use_container_width=True):
                st.session_state.symptom_analysis_data = None
                st.session_state.diagnosis_data = None
                st.session_state.top_diagnosis = None
                st.session_state.symptom_text = ''
                st.success("Analysis cleared!")

def show_diagnosis_agent(healthcare_system, active_tab="Diagnosis"):
    """Enhanced Diagnosis Agent Interface with detailed disease information"""
    st.subheader("🩺 Diagnosis Agent - Disease Identification")
    
    # Check if we have symptom data from previous analysis
    if 'symptom_analysis_data' in st.session_state and st.session_state.symptom_analysis_data is not None:
        # Use stored symptoms data
        symptoms_data = st.session_state.symptom_analysis_data
        diagnosis_data = st.session_state.get('diagnosis_data', {})
        
        st.info("📋 **Using symptoms from Symptom Analysis Agent**")
        
        with st.expander("📝 Review Symptoms Analysis", expanded=True):
            st.write(f"**Patient Symptoms:** {symptoms_data['symptoms_text']}")
            st.write(f"**Age:** {symptoms_data['patient_info']['age']} | **Gender:** {symptoms_data['patient_info']['gender']}")
            st.write(f"**Duration:** {symptoms_data['patient_info']['duration']} | **Severity:** {symptoms_data['patient_info']['severity']}")
            
            # Show affected body systems from symptom analysis
            if symptoms_data and 'body_systems_affected' in symptoms_data:
                if symptoms_data['body_systems_affected']:
                    st.write("**Affected Body Systems Identified:**")
                    for system in symptoms_data['body_systems_affected']:
                        st.write(f"• {system.title()}")
        
        # Show diagnosis results
        if diagnosis_data and diagnosis_data.get('possible_diagnoses'):
            # Only process if this is the active tab
            if active_tab == "Diagnosis":
                st.success("✅ Diagnosis Analysis Complete")
                
                # Show top diagnosis prominently with detailed information
                top_diagnosis = diagnosis_data['possible_diagnoses'][0]
                
                # Get affected body systems from symptom analysis
                body_systems_affected = []
                if symptoms_data and 'body_systems_affected' in symptoms_data:
                    body_systems_affected = symptoms_data['body_systems_affected']
                
                # Generate explanation
                explanation = healthcare_system.ai_symptom_checker.get_diagnosis_explanation(
                    top_diagnosis, 
                    body_systems_affected,
                    symptoms_data.get('analysis', {}).get('local_analysis', {}).get('symptoms_found', [])
                )
                
                # Display top diagnosis with detailed information
                system_matches = set(top_diagnosis['body_systems']) & set(body_systems_affected)
                
                # Disease Information Card
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
                    <h2 style='color: white; text-align: center; margin-bottom: 1rem;'>🏆 TOP DIAGNOSIS IDENTIFIED</h2>
                    <h1 style='color: white; text-align: center; margin: 0;'>{top_diagnosis['condition']}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed Disease Information
                with st.expander("📋 Detailed Disease Information", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🩺 Disease Overview")
                        if top_diagnosis.get('description'):
                            st.info(f"**Description:** {top_diagnosis['description']}")
                        
                        if top_diagnosis.get('common_in'):
                            st.write(f"**Common in:** {top_diagnosis['common_in']}")
                        
                        st.write(f"**Medical Specialty:** {top_diagnosis['specialty'].title()}")
                        
                        if top_diagnosis.get('complications'):
                            st.warning("**Possible Complications:**")
                            for comp in top_diagnosis['complications']:
                                st.write(f"• {comp}")
                    
                    with col2:
                        st.subheader("📊 Diagnosis Statistics")
                        st.write(f"**Overall Match:** {top_diagnosis['match_percentage']:.1f}%")
                        st.write(f"**Symptom Match:** {top_diagnosis['matching_symptoms']}/{top_diagnosis['total_symptoms']} symptoms")
                        st.write(f"**Confidence Level:** {top_diagnosis['confidence']}")
                        st.write(f"**Urgency Level:** {top_diagnosis['urgency'].upper()}")
                        
                        if system_matches:
                            st.success(f"✅ **System Alignment:** Matches {len(system_matches)} affected body system(s)")
                            for system in system_matches:
                                st.write(f"  • {system.title()}")
                        else:
                            st.warning("⚠️ **Note:** This diagnosis doesn't match identified body systems")
                
                # Diagnosis Explanation
                with st.expander("🔍 Why this diagnosis was selected", expanded=False):
                    for line in explanation:
                        st.write(line)
                
                # Store top diagnosis for treatment advisor
                st.session_state.top_diagnosis = top_diagnosis['condition']
                st.session_state.top_diagnosis_details = top_diagnosis
                
                # Select Button for Treatment Advisor
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("💊 Get Treatment for This Diagnosis", type="primary", use_container_width=True):
                        st.session_state.active_ai_agent = "Treatment"
                        st.rerun()
                
                # Show other possible diagnoses
                st.subheader("🔍 Other Possible Diagnoses")
                for idx, diagnosis in enumerate(diagnosis_data['possible_diagnoses'][1:4], 1):  # Show next 3
                    # Check system alignment
                    diag_system_matches = set(diagnosis['body_systems']) & set(body_systems_affected)
                    has_system_match = len(diag_system_matches) > 0
                    
                    # Color based on system match
                    if has_system_match:
                        match_color = "#4CAF50"  # Green for matching systems
                    elif diagnosis['match_percentage'] > 70:
                        match_color = "#FF9800"  # Orange for high match without system alignment
                    elif diagnosis['match_percentage'] > 50:
                        match_color = "#FFC107"  # Yellow for medium match
                    else:
                        match_color = "#F44336"  # Red for low match
                    
                    with st.expander(f"#{idx+1}: {diagnosis['condition']} ({diagnosis['match_percentage']:.1f}% match)", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Confidence:** {diagnosis['confidence']}")
                            st.write(f"**Urgency:** {diagnosis['urgency'].upper()}")
                            st.write(f"**Specialty:** {diagnosis['specialty'].title()}")
                        
                        with col2:
                            if diagnosis.get('description'):
                                st.write(f"**Description:** {diagnosis['description'][:100]}...")
                        
                        if has_system_match:
                            st.success(f"✅ Matches {len(diag_system_matches)} affected body system(s)")
                        else:
                            st.warning("⚠️ Doesn't match identified body systems")
                        
                        if st.button(f"Select {diagnosis['condition']}", key=f"select_diag_{idx}"):
                            st.session_state.top_diagnosis = diagnosis['condition']
                            st.session_state.top_diagnosis_details = diagnosis
                            st.session_state.active_ai_agent = "Treatment"
                            st.rerun()
                
                # Show recommended specialty
                st.subheader("🎯 Recommended Medical Specialty")
                
                # Get specialty based on affected body systems
                if body_systems_affected:
                    if 'neurology' in body_systems_affected:
                        recommended_specialty = 'Neurology'
                    elif 'cardiology' in body_systems_affected:
                        recommended_specialty = 'Cardiology'
                    elif 'respiratory' in body_systems_affected:
                        recommended_specialty = 'Pulmonology'
                    elif 'pediatrics' in body_systems_affected:
                        recommended_specialty = 'Pediatrics'
                    else:
                        recommended_specialty = diagnosis_data['predicted_specialty'].title()
                else:
                    recommended_specialty = diagnosis_data['predicted_specialty'].title()
                
                st.info(f"Based on your affected body systems, AI recommends consulting a **{recommended_specialty}** specialist")
                
                # Navigation to Treatment Advisor
                st.markdown("---")
                st.success("🔜 **Next Step:** Click the 'Get Treatment' button above or switch to the '💊 Treatment Advisor' tab to get detailed treatment recommendations!")
                
        else:
            # Manual diagnosis input option
            st.warning("No diagnosis data available. Please run Symptom Analysis first.")
            
            # Only show manual input if this is the active tab
            if active_tab == "Diagnosis":
                with st.expander("📝 Enter Symptoms Manually", expanded=True):
                    manual_input = st.text_area(
                        "Describe your symptoms for diagnosis:",
                        placeholder="Example: Fever, cough, body aches, fatigue...",
                        height=150,
                        key="manual_diagnosis_input"
                    )
                    
                    if st.button("🩺 Get AI Diagnosis", type="primary", key="manual_diagnosis_btn"):
                        if manual_input:
                            with st.spinner("🩺 Diagnosis Agent is analyzing..."):
                                time.sleep(2)
                                
                                # Extract symptoms
                                predicted_specialty, confidence, extracted_symptoms = healthcare_system.ai_symptom_checker.analyze_symptoms(manual_input)
                                
                                # Run Symptom Analysis first
                                symptom_analysis = healthcare_system.ai_symptom_checker.symptom_analysis_agent(manual_input)
                                
                                # Run Diagnosis Agent with body systems
                                possible_diagnoses = healthcare_system.ai_symptom_checker.diagnosis_agent(
                                    extracted_symptoms, 
                                    symptom_analysis,
                                    symptom_analysis['body_systems_affected']
                                )
                                
                                # Store diagnosis data
                                st.session_state.diagnosis_data = {
                                    'predicted_specialty': predicted_specialty,
                                    'confidence': confidence,
                                    'extracted_symptoms': extracted_symptoms,
                                    'possible_diagnoses': possible_diagnoses
                                }
                                
                                # Set the top diagnosis automatically
                                if possible_diagnoses:
                                    st.session_state.top_diagnosis = possible_diagnoses[0]['condition']
                                    st.session_state.top_diagnosis_details = possible_diagnoses[0]
                                
                                st.rerun()
    else:
        # Manual diagnosis input
        st.warning("⚠️ No symptom data available. Please use the Symptom Analysis Agent first.")
        
        # Only show manual input if this is the active tab
        if active_tab == "Diagnosis":
            with st.expander("📝 Enter Symptoms for Diagnosis", expanded=True):
                diagnosis_input = st.text_area(
                    "Describe your symptoms for diagnosis:",
                    placeholder="Example: Fever, cough, body aches, fatigue...",
                    height=150,
                    key="diagnosis_input"
                )
            
            if st.button("🩺 Get AI Diagnosis", type="primary", key="diagnosis_btn"):
                if diagnosis_input:
                    with st.spinner("🩺 Diagnosis Agent is analyzing..."):
                        time.sleep(2)
                        
                        # Extract symptoms
                        predicted_specialty, confidence, extracted_symptoms = healthcare_system.ai_symptom_checker.analyze_symptoms(diagnosis_input)
                        
                        # Run Symptom Analysis first
                        symptom_analysis = healthcare_system.ai_symptom_checker.symptom_analysis_agent(diagnosis_input)
                        
                        # Run Diagnosis Agent with body systems
                        possible_diagnoses = healthcare_system.ai_symptom_checker.diagnosis_agent(
                            extracted_symptoms, 
                            symptom_analysis,
                            symptom_analysis['body_systems_affected']
                        )
                        
                        # Store diagnosis data
                        st.session_state.diagnosis_data = {
                            'predicted_specialty': predicted_specialty,
                            'confidence': confidence,
                            'extracted_symptoms': extracted_symptoms,
                            'possible_diagnoses': possible_diagnoses
                        }
                        
                        # Set the top diagnosis automatically
                        if possible_diagnoses:
                            st.session_state.top_diagnosis = possible_diagnoses[0]['condition']
                            st.session_state.top_diagnosis_details = possible_diagnoses[0]
                        
                        st.rerun()
                else:
                    st.warning("Please describe your symptoms for diagnosis")

def show_treatment_advisor_agent(healthcare_system, active_tab="Treatment"):
    """Enhanced Treatment Advisor Agent Interface with comprehensive symptom-based medicines"""
    st.subheader("💊 Treatment Advisor Agent")
    
    # Check if we have top diagnosis from previous analysis
    if ('top_diagnosis' in st.session_state and st.session_state.top_diagnosis is not None) or \
       ('diagnosis_data' in st.session_state and st.session_state.diagnosis_data is not None and 
        st.session_state.diagnosis_data.get('possible_diagnoses')):
        
        # Determine which diagnosis to use
        if 'top_diagnosis' in st.session_state and st.session_state.top_diagnosis is not None:
            # Use the stored top diagnosis
            selected_diagnosis = st.session_state.top_diagnosis
            st.success(f"🎯 **Diagnosis from your symptom analysis:** {selected_diagnosis}")
        else:
            # Use the first diagnosis from diagnosis data
            diagnosis_data = st.session_state.diagnosis_data
            if diagnosis_data and diagnosis_data.get('possible_diagnoses'):
                selected_diagnosis = diagnosis_data['possible_diagnoses'][0]['condition']
                st.session_state.top_diagnosis = selected_diagnosis
                st.success(f"🎯 **Diagnosis identified:** {selected_diagnosis}")
            else:
                st.warning("No diagnosis data found. Please use the Diagnosis Agent first.")
                return
        
        # Patient profile from stored data
        patient_profile = {}
        symptom_list = []  # Store symptoms for medicine customization
        
        if 'symptom_analysis_data' in st.session_state and st.session_state.symptom_analysis_data is not None:
            patient_info = st.session_state.symptom_analysis_data['patient_info']
            # Extract symptoms from analysis
            symptom_analysis = st.session_state.symptom_analysis_data.get('analysis', {})
            if 'local_analysis' in symptom_analysis:
                symptom_list = symptom_analysis['local_analysis'].get('symptoms_found', [])
            elif 'symptoms_found' in symptom_analysis:
                symptom_list = symptom_analysis.get('symptoms_found', [])
            
            # Also extract from symptom text if no symptoms found
            if not symptom_list and 'symptoms_text' in st.session_state.symptom_analysis_data:
                symptom_text = st.session_state.symptom_analysis_data['symptoms_text'].lower()
                # Extract common symptoms from text
                common_symptoms = ['headache', 'fever', 'cough', 'pain', 'fatigue', 'nausea', 
                                 'vomiting', 'diarrhea', 'dizziness', 'shortness', 'breath']
                for symptom in common_symptoms:
                    if symptom in symptom_text:
                        symptom_list.append(symptom)
            
            patient_profile = {
                'age': patient_info['age'],
                'gender': patient_info['gender'],
                'duration': patient_info['duration'],
                'severity': patient_info['severity'],
                'symptoms': symptom_list  # Add symptoms to patient profile
            }
        elif 'patient_age' in st.session_state:
            patient_profile = {
                'age': st.session_state.patient_age,
                'gender': ["Select", "Male", "Female", "Other"][st.session_state.patient_gender_idx],
                'duration': ["Select", "Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"][st.session_state.symptom_duration_idx],
                'severity': ["Select", "Mild", "Moderate", "Severe"][st.session_state.symptom_severity_idx]
            }
        
        # Updated Patient Profile
        with st.expander("👤 Review Patient Profile", expanded=True):
            if patient_profile:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Age:** {patient_profile['age']}")
                    st.write(f"**Gender:** {patient_profile['gender']}")
                with col2:
                    st.write(f"**Symptom Duration:** {patient_profile['duration']}")
                    st.write(f"**Symptom Severity:** {patient_profile['severity']}")
                
                # Display detected symptoms (without duplicates)
                if symptom_list:
                    # Remove duplicates from symptom list
                    unique_symptoms = []
                    for symptom in symptom_list:
                        if symptom not in unique_symptoms:
                            unique_symptoms.append(symptom)
                    
                    st.write("**Detected Symptoms for Medicine Customization:**")
                    for symptom in unique_symptoms[:10]:  # Show only unique symptoms, limit to 10
                        st.write(f"• {symptom.title()}")
                else:
                    # Extract symptoms from text if none detected
                    if 'symptom_analysis_data' in st.session_state:
                        symptom_text = st.session_state.symptom_analysis_data.get('symptoms_text', '')
                        if symptom_text:
                            st.info("Extracting symptoms from your description for medicine customization...")
                            # Simple extraction
                            words = symptom_text.lower().split()
                            common_words = ['headache', 'fever', 'cough', 'pain', 'fatigue', 'nausea', 
                                          'vomiting', 'diarrhea', 'dizziness', 'shortness']
                            detected = [word for word in words if any(common in word for common in common_words)]
                            if detected:
                                st.write("**Auto-detected symptoms:**")
                                for symptom in set(detected)[:5]:
                                    st.write(f"• {symptom.title()}")
                                    if symptom not in symptom_list:
                                        symptom_list.append(symptom)
                            else:
                                st.info("No symptoms detected for medicine customization")
            
            # Only allergies checkbox
            has_allergies = st.checkbox("Known Allergies", key="treatment_allergies_auto")
            
            # Update patient profile with allergies
            if patient_profile:
                patient_profile['has_allergies'] = has_allergies
        
        # Get symptom-based medicines BEFORE treatment plan generation
        symptom_based_medicines = []
        if symptom_list:
            # Get severity from patient profile or default to moderate
            severity = patient_profile.get('severity', 'moderate')
            if severity not in ['mild', 'moderate', 'severe']:
                severity = 'moderate'
            
            # Get symptom-based medicines
            symptom_based_medicines, unique_symptoms = healthcare_system.ai_symptom_checker.get_enhanced_symptom_based_medicines(
                symptom_list, severity
            )
            
            if symptom_based_medicines:
                st.success(f"🎯 **AI detected {len(unique_symptoms)} symptoms for personalized medicine selection**")
                
                # Show symptom-medicine mapping
                with st.expander("🔍 View Symptom-Medicine Mapping", expanded=False):
                    for med in symptom_based_medicines:
                        st.markdown(med)
        
        # Show the "Get Treatment Advice for Selected Diagnosis" button
        if st.button("💊 Get Treatment Advice for Selected Diagnosis", type="primary", key="treatment_btn_auto"):
            with st.spinner("💊 Treatment Advisor Agent is generating recommendations..."):
                time.sleep(2)
                
                # Get treatment recommendations with symptom-based customization
                treatment_plan = healthcare_system.ai_symptom_checker.treatment_advisor_agent(selected_diagnosis, patient_profile)
                
                if treatment_plan:
                    # Add symptom-based medicines to treatment plan if available
                    if symptom_based_medicines:
                        treatment_plan['symptom_specific_medicines'] = symptom_based_medicines
                    
                    # Store treatment data
                    st.session_state.treatment_data = {
                        'diagnosis': selected_diagnosis,
                        'treatment_plan': treatment_plan,
                        'patient_profile': patient_profile,
                        'symptom_based_medicines': symptom_based_medicines
                    }
                    
                    # Display comprehensive treatment plan with symptom-specific medicines
                    display_comprehensive_treatment_plan(selected_diagnosis, treatment_plan, symptom_list)
                else:
                    st.error("No treatment information available for this diagnosis")
    else:
        # Manual diagnosis selection
        st.warning("⚠️ No diagnosis selected. Please use the Diagnosis Agent first.")
        
        # Show the button even when manually selecting a diagnosis
        diagnoses = list(healthcare_system.ai_symptom_checker.treatment_database.keys())
        selected_diagnosis = st.selectbox(
            "Select Diagnosis for Treatment Advice:",
            ["Select a diagnosis"] + diagnoses,
            key="manual_diagnosis_select"
        )
        
        # Patient profile
        with st.expander("👤 Patient Profile (Optional)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30, key="treatment_age")
                patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="treatment_gender")
            with col2:
                has_allergies = st.checkbox("Known Allergies", key="treatment_allergies")
            
            # Symptoms input for medicine customization
            symptoms_input = st.text_input("Main Symptoms (for medicine selection)", 
                                         placeholder="e.g., fever, headache, cough",
                                         key="treatment_symptoms")
            
            # Severity selection
            symptom_severity = st.selectbox("Symptom Severity", ["Mild", "Moderate", "Severe"], 
                                          key="treatment_severity")
        
        if st.button("💊 Get Treatment Advice", type="primary", key="treatment_btn") and selected_diagnosis != "Select a diagnosis":
            with st.spinner("💊 Treatment Advisor Agent is generating recommendations..."):
                time.sleep(2)
                
                # Create patient profile with symptoms
                symptoms_list = [s.strip().lower() for s in symptoms_input.split(',')] if symptoms_input else []
                
                patient_profile = {
                    'age': patient_age,
                    'gender': patient_gender,
                    'has_allergies': has_allergies,
                    'symptoms': symptoms_list,
                    'severity': symptom_severity.lower()
                }
                
                # Get symptom-based medicines
                symptom_based_medicines = []
                if symptoms_list:
                    symptom_based_medicines, unique_symptoms = healthcare_system.ai_symptom_checker.get_enhanced_symptom_based_medicines(
                        symptoms_list, symptom_severity.lower()
                    )
                
                # Get treatment recommendations
                treatment_plan = healthcare_system.ai_symptom_checker.treatment_advisor_agent(selected_diagnosis, patient_profile)
                
                if treatment_plan:
                    # Add symptom-based medicines to treatment plan if available
                    if symptom_based_medicines:
                        treatment_plan['symptom_specific_medicines'] = symptom_based_medicines
                    
                    # Store treatment data
                    st.session_state.treatment_data = {
                        'diagnosis': selected_diagnosis,
                        'treatment_plan': treatment_plan,
                        'patient_profile': patient_profile,
                        'symptom_based_medicines': symptom_based_medicines
                    }
                    
                    # Display comprehensive treatment plan
                    display_comprehensive_treatment_plan(selected_diagnosis, treatment_plan, symptoms_list)
                else:
                    st.error("No treatment information available for this diagnosis")

def display_comprehensive_treatment_plan(diagnosis, treatment_plan, symptom_list=None):
    """Display comprehensive treatment plan with symptom-specific medicines"""
    st.success(f"✅ Treatment Plan for {diagnosis}")
    
    # Treatment plan header
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #FF9800, #F57C00); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h2 style='color: white; text-align: center; margin: 0;'>💊 TREATMENT PLAN</h2>
        <h3 style='color: white; text-align: center; margin: 0.5rem 0;'>{diagnosis}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 Self-Care", "🎯 Symptom Medicines", "💊 General Medicines", "⚠️ Precautions", "🥗 Diet", "🏥 Doctor Visit", "🚨 Emergency"])
    
    with tab1:
        st.subheader("🏠 Self-Care Recommendations")
        if 'self_care' in treatment_plan and treatment_plan['self_care']:
            for i, care in enumerate(treatment_plan['self_care'], 1):
                st.write(f"{i}. {care}")
        else:
            st.info("No specific self-care recommendations available. Consult your doctor.")
    
    with tab2:
        st.subheader("🎯 Symptom-Specific Medicines")
        
        # Display symptom-specific medicines if available
        if 'symptom_specific_medicines' in treatment_plan and treatment_plan['symptom_specific_medicines']:
            st.info("💡 **Personalized based on your specific symptoms**")
            for med in treatment_plan['symptom_specific_medicines']:
                st.markdown(f"• {med}")
            
            # Show medicine mapping explanation
            if 'detected_symptoms' in treatment_plan:
                st.markdown("---")
                st.subheader("🔍 Symptom-Medicine Mapping:")
                for symptom in treatment_plan['detected_symptoms'][:5]:
                    st.write(f"**{symptom}:** Medicines selected based on severity and type")
        else:
            st.info("No specific symptom medicines identified. See general medicines tab.")
    
    with tab3:
        st.subheader("💊 General Medicines for this Condition")
        if 'medications' in treatment_plan and treatment_plan['medications']:
            for category, meds in treatment_plan['medications'].items():
                st.markdown(f"**{category}:**")
                # Show unique medicines only
                unique_meds = []
                for med in meds:
                    if med not in unique_meds:
                        unique_meds.append(med)
                
                for med in unique_meds:
                    st.write(f"• {med}")
                st.write("")  # Add spacing
        else:
            st.info("Consult doctor for medication recommendations")
    
    with tab4:
        st.subheader("⚠️ Important Precautions")
        if 'precautions' in treatment_plan and treatment_plan['precautions']:
            for i, precaution in enumerate(treatment_plan['precautions'], 1):
                st.write(f"{i}. {precaution}")
        else:
            st.info("Follow general health precautions")
    
    with tab5:
        st.subheader("🥗 Dietary Advice")
        if 'dietary_advice' in treatment_plan and treatment_plan['dietary_advice']:
            for i, advice in enumerate(treatment_plan['dietary_advice'], 1):
                st.write(f"{i}. {advice}")
        else:
            st.info("Maintain a balanced, nutritious diet")
    
    with tab6:
        st.subheader("🏥 When to See a Doctor")
        if 'when_to_see_doctor' in treatment_plan:
            st.info(treatment_plan['when_to_see_doctor'])
        else:
            st.info("Consult a doctor for proper medical advice")
        
        if 'follow_up' in treatment_plan:
            st.subheader("📅 Follow-up Instructions")
            st.info(treatment_plan['follow_up'])
    
    with tab7:
        st.subheader("🚨 Emergency Warning Signs")
        if 'emergency_signs' in treatment_plan and treatment_plan['emergency_signs']:
            for i, sign in enumerate(treatment_plan['emergency_signs'], 1):
                st.error(f"{i}. {sign}")
        else:
            st.warning("Seek emergency care for severe symptoms like difficulty breathing, chest pain, or loss of consciousness")
    
    # General advice
    st.markdown("---")
    st.subheader("📋 General Treatment Advice")
    if 'general_advice' in treatment_plan and treatment_plan['general_advice']:
        for advice in treatment_plan['general_advice']:
            st.write(f"• {advice}")
    
    # Download treatment plan as PDF
    st.markdown("---")
    st.subheader("📥 Download Treatment Plan")
    
    # Create a simple text version for download
    treatment_text = f"""TREATMENT PLAN FOR {diagnosis.upper()}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SELF-CARE RECOMMENDATIONS:
{chr(10).join(f"- {care}" for care in treatment_plan.get('self_care', ['No specific recommendations']))}

SYMPTOM-SPECIFIC MEDICINES:
{chr(10).join(f"- {med}" for med in treatment_plan.get('symptom_specific_medicines', ['No symptom-specific medicines']))}

GENERAL MEDICINES:
{chr(10).join(f"- {category}: {', '.join(meds)}" for category, meds in treatment_plan.get('medications', {}).items())}

PRECAUTIONS:
{chr(10).join(f"- {precaution}" for precaution in treatment_plan.get('precautions', ['No specific precautions']))}

DIETARY ADVICE:
{chr(10).join(f"- {advice}" for advice in treatment_plan.get('dietary_advice', ['No specific dietary advice']))}

WHEN TO SEE A DOCTOR:
{treatment_plan.get('when_to_see_doctor', 'Consult a healthcare professional')}

EMERGENCY WARNING SIGNS:
{chr(10).join(f"- {sign}" for sign in treatment_plan.get('emergency_signs', ['Difficulty breathing', 'Chest pain', 'Loss of consciousness']))}

IMPORTANT: This is for informational purposes only. Always consult a healthcare professional for medical advice.
"""
    
    # Create download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download as Text File",
            data=treatment_text,
            file_name=f"treatment_plan_{diagnosis.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Create JSON version
        treatment_json = json.dumps({
            'diagnosis': diagnosis,
            'treatment_plan': treatment_plan,
            'generated_date': datetime.now().isoformat(),
            'disclaimer': 'For informational purposes only. Consult a healthcare professional.'
        }, indent=2)
        
        st.download_button(
            label="📊 Download as JSON",
            data=treatment_json,
            file_name=f"treatment_plan_{diagnosis.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Important disclaimer
    st.markdown("---")
    st.error("""
    **⚠️ IMPORTANT MEDICAL DISCLAIMER:**
    
    This AI-generated treatment advice is for **INFORMATIONAL PURPOSES ONLY** and is **NOT A SUBSTITUTE** for professional medical advice, diagnosis, or treatment.
    
    **ALWAYS:**
    - Consult a qualified healthcare professional for proper diagnosis and treatment
    - Follow your doctor's prescribed treatment plan
    - Do not disregard professional medical advice
    - Do not delay seeking treatment because of information provided here
    - In case of emergency, call 108 immediately
    """)
    
    # Navigation to Full Report
    st.markdown("---")
    st.success("🔜 **Final Step:** Switch to the '📊 Full AI Report' tab to see your complete medical analysis!")

def show_full_ai_report(healthcare_system, active_tab="Full Report"):
    """Comprehensive AI Medical Report Interface with all collected data"""
    st.subheader("📊 AI Medical Analysis Report")
    
    # Check if we have data from previous agents
    has_symptom_data = 'symptom_analysis_data' in st.session_state and st.session_state.symptom_analysis_data is not None
    has_diagnosis_data = 'diagnosis_data' in st.session_state and st.session_state.diagnosis_data is not None
    has_treatment_data = 'treatment_data' in st.session_state and st.session_state.treatment_data is not None
    
    # Display data availability status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if has_symptom_data:
            st.markdown("""
            <div style='background: #4CAF50; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Symptom Analysis</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>✅ Available</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #f44336; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Symptom Analysis</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>❌ Missing</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if has_diagnosis_data:
            st.markdown("""
            <div style='background: #4CAF50; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Diagnosis Data</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>✅ Available</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #f44336; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Diagnosis Data</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>❌ Missing</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if has_treatment_data:
            treatment_data = st.session_state.treatment_data
            if treatment_data and 'treatment_plan' in treatment_data and treatment_data['treatment_plan'] is not None:
                st.markdown("""
                <div style='background: #4CAF50; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>Treatment Plan</h4>
                    <h3 style='color: white; margin: 0.5rem 0;'>✅ Available</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #FF9800; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>Treatment Plan</h4>
                    <h3 style='color: white; margin: 0.5rem 0;'>⚠️ Partial</h3>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #f44336; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Treatment Plan</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>❌ Missing</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if has_symptom_data and has_diagnosis_data and has_treatment_data:
            treatment_data = st.session_state.treatment_data
            if treatment_data and 'treatment_plan' in treatment_data and treatment_data['treatment_plan'] is not None:
                st.markdown("""
                <div style='background: #2196F3; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>Report Status</h4>
                    <h3 style='color: white; margin: 0.5rem 0;'>📊 Complete</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #FF9800; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>Report Status</h4>
                    <h3 style='color: white; margin: 0.5rem 0;'>📝 Partial</h3>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #f44336; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Report Status</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>❌ Incomplete</h3>
            </div>
            """, unsafe_allow_html=True)
    
    st.info("✅ **Data from previous agents will be included in your report**")
    
    # Only show the generate report button if this is the active tab
    if active_tab == "Full Report":
        if st.button("📊 Generate Comprehensive AI Report", type="primary", key="full_report_btn", use_container_width=True):
            with st.spinner("🤖 AI Agents collaborating to generate comprehensive report..."):
                time.sleep(3)
                
                # Get urgency level
                symptom_text = ""
                if has_symptom_data:
                    symptom_text = st.session_state.symptom_analysis_data['symptoms_text']
                
                urgency_level, urgency_advice = healthcare_system.ai_symptom_checker.get_urgency_level(symptom_text)
                
                # Compile comprehensive report from stored data
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'report_id': secrets.token_hex(8).upper(),
                    'ai_agents_used': [],
                    'report_version': '1.0'
                }
                
                # Add symptom analysis data
                if has_symptom_data:
                    report['symptom_analysis'] = st.session_state.symptom_analysis_data
                    report['ai_agents_used'].append('Symptom Analysis Agent')
                
                # Add diagnosis data
                if has_diagnosis_data:
                    report['diagnosis_analysis'] = st.session_state.diagnosis_data
                    report['ai_agents_used'].append('Diagnosis Agent')
                
                # Add treatment data
                if has_treatment_data:
                    treatment_data = st.session_state.treatment_data
                    if treatment_data and 'treatment_plan' in treatment_data and treatment_data['treatment_plan'] is not None:
                        report['treatment_analysis'] = treatment_data
                        report['ai_agents_used'].append('Treatment Advisor Agent')
                    else:
                        st.warning("⚠️ Treatment data exists but treatment plan is incomplete")
                        # Add partial treatment data
                        if treatment_data:
                            report['treatment_analysis'] = {
                                'diagnosis': treatment_data.get('diagnosis'),
                                'patient_profile': treatment_data.get('patient_profile'),
                                'status': 'incomplete'
                            }
                            report['ai_agents_used'].append('Treatment Advisor Agent (Partial)')
                
                # Add urgency assessment
                report['urgency_assessment'] = {
                    'level': urgency_level,
                    'advice': urgency_advice
                }
                
                # Add report summary
                report['summary'] = {
                    'total_agents_used': len(report['ai_agents_used']),
                    'report_complete': len(report['ai_agents_used']) == 3,
                    'generated_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Store the complete report
                st.session_state.full_ai_report = report
                
                # Display the comprehensive report
                display_comprehensive_report(report)
    else:
        # Show missing data guidance
        missing_data = []
        if not has_symptom_data:
            missing_data.append("Symptom Analysis")
        if not has_diagnosis_data:
            missing_data.append("Diagnosis")
        if not has_treatment_data:
            missing_data.append("Treatment Plan")
        
        if missing_data:
            st.warning(f"⚠️ **Missing data:** {', '.join(missing_data)}")
            st.info("💡 **To generate a complete report:**")
            st.write("1. Use the **Symptom Analysis Agent** to analyze your symptoms")
            st.write("2. Use the **Diagnosis Agent** to get possible diagnoses")
            st.write("3. Use the **Treatment Advisor Agent** to get treatment recommendations")
            st.write("4. Return here to generate the complete AI Medical Report")
            
            # Quick navigation buttons
            st.markdown("---")
            st.subheader("🚀 Quick Navigation to Missing Data")
            
            cols = st.columns(3)
            
            if not has_symptom_data:
                with cols[0]:
                    if st.button("🔍 Go to Symptom Analysis", use_container_width=True):
                        st.session_state.active_ai_agent = "Symptom Analysis"
                        st.session_state.show_symptom_checker = True
                        st.rerun()
            
            if not has_diagnosis_data:
                with cols[1]:
                    if st.button("🩺 Go to Diagnosis Agent", use_container_width=True):
                        st.session_state.active_ai_agent = "Diagnosis"
                        st.session_state.show_symptom_checker = True
                        st.rerun()
            
            if not has_treatment_data:
                with cols[2]:
                    if st.button("💊 Go to Treatment Advisor", use_container_width=True):
                        st.session_state.active_ai_agent = "Treatment"
                        st.session_state.show_symptom_checker = True
                        st.rerun()

def display_comprehensive_report(report):
    """Display the comprehensive AI medical report WITH ENHANCED DOWNLOAD OPTIONS"""
    st.success("✅ AI Medical Analysis Report Generated")
    
    # Report header
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a237e 0%, #283593 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0; text-align: center;'>🏥 AI Medical Analysis Report</h2>
        <p style='text-align: center; margin: 0.5rem 0 0 0;'>Report ID: {report['report_id']}</p>
        <p style='text-align: center; margin: 0.5rem 0 0 0;'>Generated on {datetime.fromisoformat(report['timestamp']).strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    # Get body systems from symptom analysis
    body_systems_affected = []
    if 'symptom_analysis' in report:
        body_systems_affected = report['symptom_analysis'].get('body_systems_affected', [])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        urgency_color = {"HIGH": "#f44336", "MEDIUM": "#ff9800", "LOW": "#4caf50"}
        urgency_level = report['urgency_assessment']['level']
        st.markdown(f"""
        <div style='background: {urgency_color.get(urgency_level, "#666")}; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: white; margin: 0;'>Urgency Level</h4>
            <h3 style='color: white; margin: 0.5rem 0;'>{urgency_level}</h3>
            <small>{report['urgency_assessment']['advice']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'diagnosis_analysis' in report:
            specialty = report['diagnosis_analysis']['predicted_specialty']
            st.markdown(f"""
            <div style='background: #2196f3; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Recommended Specialty</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>{specialty.title()}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'diagnosis_analysis' in report:
            confidence = report['diagnosis_analysis']['confidence']
            confidence_color = "#4caf50" if confidence > 0.7 else "#ff9800" if confidence > 0.4 else "#f44336"
            st.markdown(f"""
            <div style='background: {confidence_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>AI Confidence</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>{confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if body_systems_affected:
            st.markdown(f"""
            <div style='background: #9c27b0; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                <h4 style='color: white; margin: 0;'>Affected Systems</h4>
                <h3 style='color: white; margin: 0.5rem 0;'>{len(body_systems_affected)}</h3>
                <small>{', '.join([s.title() for s in body_systems_affected[:2]])}{'...' if len(body_systems_affected) > 2 else ''}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Sections
    if 'symptom_analysis' in report:
        with st.expander("🔍 Symptom Analysis (Agent 1)", expanded=True):
            symptom_data = report['symptom_analysis']
            st.write(f"**Patient Symptoms:** {symptom_data['symptoms_text']}")
            
            if 'patient_info' in symptom_data:
                st.write("**Patient Information:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"• Age: {symptom_data['patient_info'].get('age', 'N/A')}")
                    st.write(f"• Gender: {symptom_data['patient_info'].get('gender', 'N/A')}")
                with col2:
                    st.write(f"• Duration: {symptom_data['patient_info'].get('duration', 'N/A')}")
                    st.write(f"• Severity: {symptom_data['patient_info'].get('severity', 'N/A')}")
            
            if 'analysis' in symptom_data:
                analysis = symptom_data['analysis']
                st.write("**Detected Symptoms:**")
                # Show unique symptoms only
                unique_symptoms = []
                if 'local_analysis' in analysis:
                    for symptom in analysis['local_analysis'].get('symptoms_found', []):
                        if symptom not in unique_symptoms:
                            unique_symptoms.append(symptom)
                elif 'symptoms_found' in analysis:
                    for symptom in analysis.get('symptoms_found', []):
                        if symptom not in unique_symptoms:
                            unique_symptoms.append(symptom)
                
                for symptom in unique_symptoms:
                    st.write(f"• {symptom.title()}")
                
                st.write("**Body Systems Affected:**")
                if 'local_analysis' in analysis:
                    for system in analysis['local_analysis'].get('body_systems_affected', []):
                        st.write(f"• {system.title()} ⭐")
                elif 'body_systems_affected' in analysis:
                    for system in analysis.get('body_systems_affected', []):
                        st.write(f"• {system.title()} ⭐")
                
                if 'local_analysis' in analysis:
                    st.write(f"**Severity Assessment:** {analysis['local_analysis'].get('severity_assessment', 'N/A').title()}")
                    st.write(f"**Duration Pattern:** {analysis['local_analysis'].get('duration_pattern', 'N/A').title()}")
                    
                    if analysis['local_analysis'].get('trigger_factors'):
                        st.write("**Trigger Factors:**")
                        for trigger in analysis['local_analysis']['trigger_factors']:
                            st.write(f"• {trigger.replace('_', ' ').title()}")
    
    if 'diagnosis_analysis' in report:
        with st.expander("🩺 Diagnosis Analysis (Agent 2)", expanded=True):
            diag_data = report['diagnosis_analysis']
            
            if diag_data.get('possible_diagnoses'):
                st.write("**Top Diagnosis:**")
                top_diag = diag_data['possible_diagnoses'][0]
                
                # Check system alignment
                system_alignment = "Good" if top_diag.get('system_boost', 0) > 0 else "Moderate"
                alignment_color = "#4CAF50" if system_alignment == "Good" else "#FF9800"
                
                st.markdown(f"""
                <div style='background: #e8f5e8; padding: 1.5rem; border-radius: 8px; border-left: 5px solid {alignment_color}; margin: 0.5rem 0;'>
                    <h4 style='margin: 0; color: #2e7d32;'>{top_diag['condition']}</h4>
                    <div style='display: flex; justify-content: between; align-items: center; margin-top: 0.5rem;'>
                        <span>Overall Match: <strong>{top_diag['match_percentage']:.1f}%</strong></span>
                        <span>Confidence: <strong>{top_diag['confidence']}</strong></span>
                        <span>Urgency: <strong>{top_diag['urgency']}</strong></span>
                    </div>
                    <div style='margin-top: 0.5rem; color: #666;'>
                        <small>Symptom Match: {top_diag.get('original_match', top_diag['match_percentage']):.1f}% | System Alignment: {system_alignment}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show system alignment details
                if 'symptom_analysis' in report:
                    affected_systems = report['symptom_analysis'].get('body_systems_affected', [])
                    diag_systems = top_diag.get('body_systems', [])
                    matching_systems = set(affected_systems) & set(diag_systems)
                    
                    if matching_systems:
                        st.success(f"🎯 **System Alignment:** Diagnosis matches {len(matching_systems)} affected body system(s): {', '.join(matching_systems)}")
                
                st.write("**Other Possible Diagnoses:**")
                for diag in diag_data['possible_diagnoses'][1:]:
                    # Check system alignment
                    has_system_match = diag.get('system_boost', 0) > 0
                    match_indicator = " 🎯" if has_system_match else ""
                    
                    st.write(f"• {diag['condition']}{match_indicator} ({diag['match_percentage']:.1f}% match, {diag['confidence']} confidence)")
            else:
                st.info("No specific diagnoses identified")
    
    if 'treatment_analysis' in report and report['treatment_analysis'] is not None:
        with st.expander("💊 Treatment Recommendations (Agent 3)", expanded=True):
            treat_data = report['treatment_analysis']
            
            # Check if treatment data is complete
            if treat_data.get('status') == 'incomplete':
                st.warning("⚠️ Treatment data is incomplete. Please use the Treatment Advisor Agent to generate a complete treatment plan.")
                if treat_data.get('diagnosis'):
                    st.write(f"**Diagnosis:** {treat_data['diagnosis']}")
                if treat_data.get('patient_profile'):
                    st.write("**Partial patient profile available**")
            else:
                # Complete treatment data
                if treat_data.get('diagnosis'):
                    st.write(f"**Diagnosis:** {treat_data['diagnosis']}")
                
                if treat_data.get('treatment_plan'):
                    treatment = treat_data['treatment_plan']
                    
                    st.write("**Self-Care Recommendations:**")
                    for care in treatment.get('self_care', []):
                        st.write(f"• {care}")
                    
                    # Show symptom-specific medicines if available
                    if 'symptom_specific_medicines' in treatment and treatment['symptom_specific_medicines']:
                        st.write("**🎯 Symptom-Specific Medicines:**")
                        for med in treatment['symptom_specific_medicines']:
                            st.write(f"• {med}")
                    
                    st.write("**General Medication Suggestions:**")
                    if 'medications' in treatment and treatment['medications']:
                        for category, meds in treatment['medications'].items():
                            st.markdown(f"**{category}:**")
                            # Show unique medicines only
                            unique_meds = []
                            for med in meds:
                                if med not in unique_meds:
                                    unique_meds.append(med)
                            
                            for med in unique_meds:
                                st.write(f"• {med}")
                    else:
                        st.info("Consult doctor for medication recommendations")
                    
                    st.write("**Warning Signs to Watch For:**")
                    for sign in treatment.get('emergency_signs', []):
                        st.write(f"• {sign}")
                    
                    st.write("**When to See a Doctor:**")
                    st.info(treatment.get('when_to_see_doctor', 'Consult a healthcare professional'))
                else:
                    st.info("Consult healthcare provider for treatment recommendations")
    else:
        st.warning("Treatment data not available. Please use the Treatment Advisor agent first.")
    
    # AI Agents Used
    st.info(f"**🤖 AI Agents Used:** {', '.join(report['ai_agents_used'])}")
    
    # ✅ ENHANCED DOWNLOAD SECTION WITH PDF OPTION
    st.markdown("---")
    st.subheader("📥 Download Complete Report")
    
    # Create report content for download
    report_content = generate_report_content(report)
    
    # Create multiple download formats in tabs
    download_tab1, download_tab2, download_tab3, download_tab4 = st.tabs(["📄 PDF Report", "📊 JSON Data", "📝 Text Report", "💊 Treatment Only"])
    
    with download_tab1:
        # PDF Report (Using HTML to simulate PDF)
        st.write("**Professional PDF Report**")
        st.info("Download a professionally formatted PDF version of your complete medical analysis.")
        
        # Create HTML for PDF (simplified version)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Medical Analysis Report - {report['report_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #1a237e 0%, #283593 100%); 
                         color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 5px solid #2196F3; background: #f5f5f5; }}
                .urgent {{ color: #f44336; font-weight: bold; }}
                .info {{ color: #2196F3; }}
                .warning {{ color: #ff9800; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 AI Medical Analysis Report</h1>
                <h3>Report ID: {report['report_id']}</h3>
                <p>Generated on {datetime.fromisoformat(report['timestamp']).strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            
            <h2>📋 Executive Summary</h2>
            <p><strong>Urgency Level:</strong> <span class="urgent">{report['urgency_assessment']['level']}</span></p>
            <p><strong>Urgency Advice:</strong> {report['urgency_assessment']['advice']}</p>
            <p><strong>AI Agents Used:</strong> {', '.join(report['ai_agents_used'])}</p>
            
            <div class="section">
                <h3>🔍 Symptom Analysis</h3>
                <p><strong>Patient Symptoms:</strong> {report.get('symptom_analysis', {}).get('symptoms_text', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h3>🩺 Diagnosis Analysis</h3>
        """
        
        if 'diagnosis_analysis' in report:
            diag_data = report['diagnosis_analysis']
            if diag_data.get('possible_diagnoses'):
                top_diag = diag_data['possible_diagnoses'][0]
                html_content += f"""
                <p><strong>Top Diagnosis:</strong> {top_diag['condition']}</p>
                <p><strong>Match Percentage:</strong> {top_diag['match_percentage']:.1f}%</p>
                <p><strong>Confidence:</strong> {top_diag['confidence']}</p>
                """
        
        html_content += """
            </div>
            
            <div class="section">
                <h3>⚠️ Important Medical Disclaimer</h3>
                <p>This AI-generated report is for informational and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
                <p class="urgent">In case of emergency, call 108 immediately.</p>
            </div>
            
            <div class="section">
                <p><em>Report generated by AI Healthcare Network - For informational purposes only</em></p>
            </div>
        </body>
        </html>
        """
        
        # Create PDF download button (HTML as fallback, but you could use a PDF generator here)
        import base64
        b64_html = base64.b64encode(html_content.encode()).decode()
        
        st.markdown(f"""
        <a href="data:text/html;base64,{b64_html}" download="AI_Medical_Report_{report['report_id']}.html">
            <button style="background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; width: 100%;">
                📄 Download as HTML/PDF Report
            </button>
        </a>
        """, unsafe_allow_html=True)
        
        st.caption("Note: HTML file can be converted to PDF using your browser's print function")
    
    with download_tab2:
        # JSON Data
        st.write("**Complete JSON Data**")
        st.info("Download all raw data in JSON format for integration with other systems.")
        
        report_json = json.dumps(report, indent=2, default=str)
        
        st.download_button(
            label="📊 Download Complete JSON",
            data=report_json,
            file_name=f"ai_medical_report_{report['report_id']}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Show preview of JSON
        with st.expander("Preview JSON Structure"):
            st.code(report_json[:500] + "..." if len(report_json) > 500 else report_json)
    
    with download_tab3:
        # Text Report
        st.write("**Detailed Text Report**")
        st.info("Comprehensive text report suitable for printing or sharing with healthcare providers.")
        
        st.download_button(
            label="📝 Download Text Report",
            data=report_content,
            file_name=f"ai_medical_report_{report['report_id']}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Show preview of text report
        with st.expander("Preview Text Report"):
            st.text(report_content[:500] + "..." if len(report_content) > 500 else report_content)
    
    with download_tab4:
        # Treatment Plan Only
        st.write("**Treatment Plan Only**")
        st.info("Download just the treatment recommendations separately.")
        
        if 'treatment_analysis' in report and report['treatment_analysis'] and report['treatment_analysis'].get('treatment_plan'):
            treatment_plan = report['treatment_analysis']['treatment_plan']
            diagnosis = report['treatment_analysis'].get('diagnosis', 'Unknown Diagnosis')
            
            treatment_text = f"""TREATMENT PLAN - {diagnosis.upper()}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {report['report_id']}

SELF-CARE RECOMMENDATIONS:
{chr(10).join(f"- {care}" for care in treatment_plan.get('self_care', ['No specific recommendations']))}

SYMPTOM-SPECIFIC MEDICINES:
{chr(10).join(f"- {med}" for med in treatment_plan.get('symptom_specific_medicines', ['No symptom-specific medicines']))}

GENERAL MEDICATIONS:
{chr(10).join(f"- {category}: {', '.join(meds)}" for category, meds in treatment_plan.get('medications', {}).items())}

PRECAUTIONS:
{chr(10).join(f"- {precaution}" for precaution in treatment_plan.get('precautions', ['No specific precautions']))}

DIETARY ADVICE:
{chr(10).join(f"- {advice}" for advice in treatment_plan.get('dietary_advice', ['No specific dietary advice']))}

WHEN TO SEE A DOCTOR:
{treatment_plan.get('when_to_see_doctor', 'Consult a healthcare professional')}

EMERGENCY WARNING SIGNS:
{chr(10).join(f"- {sign}" for sign in treatment_plan.get('emergency_signs', ['Difficulty breathing', 'Chest pain', 'Loss of consciousness']))}

FOLLOW-UP INSTRUCTIONS:
{treatment_plan.get('follow_up', 'Follow-up as recommended by your healthcare provider')}

IMPORTANT DISCLAIMER:
This treatment plan is generated by AI for informational purposes only. 
Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
Do not self-medicate based on this information.
In case of emergency, call 108 immediately.
"""
            
            st.download_button(
                label="💊 Download Treatment Plan",
                data=treatment_text,
                file_name=f"treatment_plan_{diagnosis.lower().replace(' ', '_')}_{report['report_id']}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            with st.expander("Preview Treatment Plan"):
                st.text(treatment_text[:300] + "..." if len(treatment_text) > 300 else treatment_text)
        else:
            st.warning("No treatment plan available for download")
            st.info("Complete the Treatment Advisor Agent to generate a treatment plan")
    
    # Email Report Option
    st.markdown("---")
    st.subheader("📧 Share Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_address = st.text_input("Email address to send report", placeholder="example@email.com")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("📤 Send Report via Email", use_container_width=True):
            if email_address and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email_address):
                # Simulate email sending
                with st.spinner(f"Sending report to {email_address}..."):
                    time.sleep(2)
                st.success(f"✅ Report sent to {email_address}")
            else:
                st.error("Please enter a valid email address")
    
    # Print Report Option
    st.markdown("---")
    if st.button("🖨️ Print Report", use_container_width=True):
        # Create a print-friendly version
        st.markdown("""
        <script>
        window.print();
        </script>
        """, unsafe_allow_html=True)
        st.info("Opening print dialog...")
    
    # Manage Report Section with Clear AI Session Data Button
    st.markdown("---")
    st.subheader("🔄 Manage Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear AI Session Data", use_container_width=True,
                    help="Clear all data from AI agents and start fresh"):
            st.session_state.symptom_analysis_data = None
            st.session_state.diagnosis_data = None
            st.session_state.top_diagnosis = None
            st.session_state.top_diagnosis_details = None
            st.session_state.treatment_data = None
            st.session_state.full_ai_report = None
            st.session_state.symptom_text = ''
            st.session_state.active_ai_agent = None
            st.success("✅ AI session data cleared!")
            st.rerun()
    
    with col2:
        if st.button("📋 Start New Analysis", use_container_width=True,
                    help="Start a new medical analysis from scratch"):
            st.session_state.symptom_analysis_data = None
            st.session_state.diagnosis_data = None
            st.session_state.top_diagnosis = None
            st.session_state.top_diagnosis_details = None
            st.session_state.treatment_data = None
            st.session_state.full_ai_report = None
            st.session_state.symptom_text = ''
            st.session_state.active_ai_agent = "Symptom Analysis"
            st.rerun()
    
    # Disclaimer
    st.markdown("---")
    st.error("""
    **⚠️ MEDICAL DISCLAIMER:**
    This AI-generated report is for informational and educational purposes only. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    In case of emergency, call 108 immediately.
    """)

def generate_report_content(report):
    """Generate text content for the report download"""
    content = f"""
    {'='*80}
    {' ' * 25}AI MEDICAL ANALYSIS REPORT
    {'='*80}
    
    Report ID: {report['report_id']}
    Generated: {datetime.fromisoformat(report['timestamp']).strftime('%B %d, %Y at %I:%M %p')}
    AI Agents Used: {', '.join(report['ai_agents_used'])}
    Report Version: {report.get('report_version', '1.0')}
    
    {'='*80}
    EXECUTIVE SUMMARY
    {'='*80}
    
    Urgency Level: {report['urgency_assessment']['level']}
    Urgency Advice: {report['urgency_assessment']['advice']}
    
    Report Completeness: {'Complete' if report['summary'].get('report_complete', False) else 'Partial'}
    Total AI Agents Used: {report['summary'].get('total_agents_used', 0)}/3
    
    """
    
    # Symptom Analysis
    if 'symptom_analysis' in report:
        content += f"""
        {'='*80}
        SYMPTOM ANALYSIS (Agent 1)
        {'='*80}
        
        PATIENT SYMPTOMS:
        {report['symptom_analysis']['symptoms_text']}
        
        PATIENT INFORMATION:
        - Age: {report['symptom_analysis']['patient_info'].get('age', 'N/A')}
        - Gender: {report['symptom_analysis']['patient_info'].get('gender', 'N/A')}
        - Duration: {report['symptom_analysis']['patient_info'].get('duration', 'N/A')}
        - Severity: {report['symptom_analysis']['patient_info'].get('severity', 'N/A')}
        
        DETECTED SYMPTOMS:
        """
        symptom_data = report['symptom_analysis']
        analysis = symptom_data.get('analysis', {})
        
        # Get symptoms from analysis
        symptoms_list = []
        if 'local_analysis' in analysis:
            symptoms_list = analysis['local_analysis'].get('symptoms_found', [])
        elif 'symptoms_found' in analysis:
            symptoms_list = analysis.get('symptoms_found', [])
        
        # Remove duplicates
        unique_symptoms = []
        for symptom in symptoms_list:
            if symptom not in unique_symptoms:
                unique_symptoms.append(symptom)
        
        for symptom in unique_symptoms:
            content += f"- {symptom.title()}\n"
        
        content += f"""
        BODY SYSTEMS AFFECTED:
        """
        # Get body systems
        body_systems = []
        if 'body_systems_affected' in symptom_data:
            body_systems = symptom_data.get('body_systems_affected', [])
        elif 'local_analysis' in analysis:
            body_systems = analysis['local_analysis'].get('body_systems_affected', [])
        
        if body_systems:
            for system in body_systems:
                content += f"- {system.title()}\n"
        else:
            content += "- Could not identify specific body systems\n"
        
        if 'local_analysis' in analysis:
            local = analysis['local_analysis']
            content += f"""
        SEVERITY ASSESSMENT: {local.get('severity_assessment', 'N/A').title()}
        DURATION PATTERN: {local.get('duration_pattern', 'N/A').title()}
        
        TRIGGER FACTORS:
        """
            if local.get('trigger_factors'):
                for trigger in local['trigger_factors']:
                    content += f"- {trigger.replace('_', ' ').title()}\n"
            else:
                content += "- No specific trigger factors identified\n"
    
    # Diagnosis Analysis
    if 'diagnosis_analysis' in report:
        content += f"""
        {'='*80}
        DIAGNOSIS ANALYSIS (Agent 2)
        {'='*80}
        
        RECOMMENDED SPECIALTY: {report['diagnosis_analysis']['predicted_specialty'].title()}
        AI CONFIDENCE SCORE: {report['diagnosis_analysis']['confidence']:.1%}
        
        TOP DIAGNOSIS IDENTIFIED:
        """
        if report['diagnosis_analysis'].get('possible_diagnoses'):
            top_diag = report['diagnosis_analysis']['possible_diagnoses'][0]
            content += f"""
            - CONDITION: {top_diag['condition']}
            - OVERALL MATCH: {top_diag['match_percentage']:.1f}%
            - CONFIDENCE LEVEL: {top_diag['confidence']}
            - URGENCY LEVEL: {top_diag['urgency'].upper()}
            - MEDICAL SPECIALTY: {top_diag['specialty'].title()}
            - SYMPTOM MATCH: {top_diag['matching_symptoms']}/{top_diag['total_symptoms']} key symptoms
            """
            
            if top_diag.get('description'):
                content += f"""
            - DESCRIPTION: {top_diag['description']}
            """
            
            if top_diag.get('common_in'):
                content += f"""
            - COMMON IN: {top_diag['common_in']}
            """
            
            if top_diag.get('complications'):
                content += f"""
            - POSSIBLE COMPLICATIONS:
            """
                for comp in top_diag['complications']:
                    content += f"              * {comp}\n"
        
        content += f"""
        OTHER POSSIBLE DIAGNOSES:
        """
        if report['diagnosis_analysis'].get('possible_diagnoses'):
            for idx, diag in enumerate(report['diagnosis_analysis']['possible_diagnoses'][1:4], 2):
                content += f"""
        {idx}. {diag['condition']}
           - Match: {diag['match_percentage']:.1f}%
           - Confidence: {diag['confidence']}
           - Urgency: {diag['urgency'].upper()}
        """
    
    # Treatment Analysis
    if 'treatment_analysis' in report and report['treatment_analysis']:
        content += f"""
        {'='*80}
        TREATMENT RECOMMENDATIONS (Agent 3)
        {'='*80}
        
        DIAGNOSIS: {report['treatment_analysis'].get('diagnosis', 'N/A')}
        
        TREATMENT PLAN:
        """
        if report['treatment_analysis'].get('treatment_plan'):
            treatment = report['treatment_analysis']['treatment_plan']
            
            content += """
        SELF-CARE RECOMMENDATIONS:
        """
            for care in treatment.get('self_care', []):
                content += f"- {care}\n"
            
            if 'symptom_specific_medicines' in treatment and treatment['symptom_specific_medicines']:
                content += """
        SYMPTOM-SPECIFIC MEDICINES:
        """
                for med in treatment['symptom_specific_medicines']:
                    content += f"- {med}\n"
            
            content += """
        MEDICATION SUGGESTIONS:
        """
            for category, meds in treatment.get('medications', {}).items():
                content += f"- {category}:\n"
                for med in meds:
                    content += f"  * {med}\n"
            
            content += """
        IMPORTANT PRECAUTIONS:
        """
            for precaution in treatment.get('precautions', []):
                content += f"- {precaution}\n"
            
            content += """
        DIETARY ADVICE:
        """
            for advice in treatment.get('dietary_advice', []):
                content += f"- {advice}\n"
            
            content += f"""
        WHEN TO SEE A DOCTOR:
        {treatment.get('when_to_see_doctor', 'Consult a healthcare professional')}
        
        FOLLOW-UP INSTRUCTIONS:
        {treatment.get('follow_up', 'Follow-up as recommended by your healthcare provider')}
        
        EMERGENCY WARNING SIGNS:
        """
            for sign in treatment.get('emergency_signs', []):
                content += f"- {sign}\n"
        
        content += """
        
        IMPORTANT NOTES:
        1. This report is for informational purposes only
        2. Always consult a qualified healthcare professional
        3. Do not self-medicate based on this report
        4. Follow medication schedules only as prescribed by a doctor
        5. Monitor symptoms and seek medical help if they worsen
        6. In case of emergency, call 108 immediately
        
        REPORT GENERATED BY:
        AI Healthcare Network - Telangana
        For informational and educational purposes only
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Report ID: {report['report_id']}
        """.format()
    
    return content

def show_patient_reviews(healthcare_system, hospital_id, hospital_name):
    """Patient Reviews and Rating System for individual hospital"""
    st.subheader(f"💬 Patient Reviews & Ratings - {hospital_name}")
    
    # Add new review
    with st.form(f"add_review_form_{hospital_id}"):
        st.write("**Share Your Experience**")
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Your Name*", key=f"name_{hospital_id}")
            rating = st.slider("Rating*", 1, 5, 5, key=f"rating_{hospital_id}")
        
        with col2:
            comment = st.text_area("Your Review*", height=100, key=f"comment_{hospital_id}")
        
        submitted = st.form_submit_button("Submit Review")
        
        if submitted:
            if patient_name and comment:
                review = healthcare_system.review_system.add_review(
                    hospital_id, patient_name, rating, comment
                )
                st.success("✅ Thank you for your review! AI sentiment analysis completed.")
                
                # Show sentiment analysis result
                sentiment_icon = "😊" if review['sentiment'] == 'positive' else "😐" if review['sentiment'] == 'neutral' else "😞"
                st.info(f"**AI Sentiment Analysis:** {sentiment_icon} {review['sentiment'].title()} (Score: {review['sentiment_score']:.2f})")
            else:
                st.error("Please fill all required fields.")
    
    # Display existing reviews
    reviews = healthcare_system.review_system.get_hospital_reviews(hospital_id)
    if reviews:
        st.write("---")
        st.write(f"**Recent Reviews ({len(reviews)})**")
        
        for review in sorted(reviews, key=lambda x: x['timestamp'], reverse=True)[:5]:
            sentiment_color = {
                'positive': '#4CAF50',
                'neutral': '#FF9800', 
                'negative': '#F44336'
            }
            
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {sentiment_color[review['sentiment']]};'>
                <div style='display: flex; justify-content: between; align-items: center;'>
                    <strong>{review['patient_name']}</strong>
                    <div>
                        {'⭐' * review['rating']}{'☆' * (5 - review['rating'])}
                        <span style='color: {sentiment_color[review['sentiment']]}; font-size: 0.8em;'>
                            ({review['sentiment']})
                        </span>
                    </div>
                </div>
                <p style='margin: 0.5rem 0 0 0;'>{review['comment']}</p>
                <small style='color: #666;'>{datetime.fromisoformat(review['timestamp']).strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No reviews yet. Be the first to share your experience!")

def show_patient_reviews_hub(healthcare_system):
    """Enhanced Patient Reviews Hub - Shows all reviews from all hospitals"""
    st.subheader("💬 Patient Reviews Hub - All Hospital Reviews")
    
    # Get all reviews from all hospitals
    all_reviews = healthcare_system.review_system.get_all_reviews()
    
    if not all_reviews:
        st.info("🌟 No reviews yet! Be the first to share your experience by visiting a hospital page and leaving a review.")
        return
    
    # Statistics
    total_reviews = len(all_reviews)
    positive_reviews = sum(1 for review in all_reviews if review['sentiment'] == 'positive')
    negative_reviews = sum(1 for review in all_reviews if review['sentiment'] == 'negative')
    neutral_reviews = sum(1 for review in all_reviews if review['sentiment'] == 'neutral')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", total_reviews)
    with col2:
        st.metric("Positive Reviews", f"{positive_reviews} ({positive_reviews/total_reviews*100:.1f}%)")
    with col3:
        st.metric("Neutral Reviews", f"{neutral_reviews} ({neutral_reviews/total_reviews*100:.1f}%)")
    with col4:
        st.metric("Negative Reviews", f"{negative_reviews} ({negative_reviews/total_reviews*100:.1f}%)")
    
    # Filter options
    st.subheader("🔍 Filter Reviews")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by hospital
        hospital_ids = list(healthcare_system.review_system.reviews.keys())
        hospital_names = []
        for hospital_id in hospital_ids:
            hospital = healthcare_system.get_hospital_by_id(hospital_id)
            if hospital is not None:
                hospital_names.append(hospital['name'])
        
        selected_hospital = st.selectbox(
            "Filter by Hospital",
            ["All Hospitals"] + hospital_names
        )
    
    with col2:
        # Filter by sentiment
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            ["All Sentiments", "Positive", "Neutral", "Negative"]
        )
    
    with col3:
        # Filter by rating
        rating_filter = st.selectbox(
            "Filter by Rating",
            ["All Ratings", "5 Stars", "4 Stars", "3 Stars", "2 Stars", "1 Star"]
        )
    
    # Apply filters
    filtered_reviews = all_reviews
    
    if selected_hospital != "All Hospitals":
        # Find hospital ID from name
        hospital_id = None
        for hid, hname in zip(hospital_ids, hospital_names):
            if hname == selected_hospital:
                hospital_id = hid
                break
        if hospital_id:
            filtered_reviews = [r for r in filtered_reviews if r['hospital_id'] == hospital_id]
    
    if sentiment_filter != "All Sentiments":
        filtered_reviews = [r for r in filtered_reviews if r['sentiment'].lower() == sentiment_filter.lower()]
    
    if rating_filter != "All Ratings":
        rating_value = int(rating_filter[0])  # Extract number from "5 Stars", "4 Stars", etc.
        filtered_reviews = [r for r in filtered_reviews if r['rating'] == rating_value]
    
    st.subheader(f"📋 Showing {len(filtered_reviews)} Reviews")
    
    if filtered_reviews:
        # Sort options
        sort_option = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First", "Highest Rating", "Lowest Rating"]
        )
        
        if sort_option == "Newest First":
            filtered_reviews.sort(key=lambda x: x['timestamp'], reverse=True)
        elif sort_option == "Oldest First":
            filtered_reviews.sort(key=lambda x: x['timestamp'])
        elif sort_option == "Highest Rating":
            filtered_reviews.sort(key=lambda x: x['rating'], reverse=True)
        elif sort_option == "Lowest Rating":
            filtered_reviews.sort(key=lambda x: x['rating'])
        
        # Display filtered reviews
        for review in filtered_reviews:
            hospital = healthcare_system.get_hospital_by_id(review['hospital_id'])
            hospital_name = hospital['name'] if hospital is not None else "Unknown Hospital"
            
            sentiment_color = {
                'positive': '#4CAF50',
                'neutral': '#FF9800', 
                'negative': '#F44336'
            }
            
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid {sentiment_color[review['sentiment']]}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <div>
                        <strong style='font-size: 1.1em;'>{review['patient_name']}</strong>
                        <span style='color: #666; margin-left: 1rem;'>at {hospital_name}</span>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 1.2em;'>
                            {'⭐' * review['rating']}{'☆' * (5 - review['rating'])}
                        </div>
                        <span style='color: {sentiment_color[review['sentiment']]}; font-size: 0.9em; font-weight: bold;'>
                            {review['sentiment'].upper()} 
                            <small>(Score: {review['sentiment_score']:.2f})</small>
                        </span>
                    </div>
                </div>
                <p style='margin: 1rem 0 0.5rem 0; font-size: 1.1em; line-height: 1.5;'>{review['comment']}</p>
                <small style='color: #888;'>{datetime.fromisoformat(review['timestamp']).strftime('%B %d, %Y at %I:%M %p')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No reviews match your selected filters.")
    
    # Quick hospital navigation
    st.subheader("🏥 Quick Hospital Navigation")
    st.write("Want to leave a review? Visit any hospital page to share your experience!")
    
    # Show hospital cards for quick access
    hospitals_with_reviews = []
    for hospital_id in hospital_ids:
        hospital = healthcare_system.get_hospital_by_id(hospital_id)
        if hospital is not None:
            reviews_count = len(healthcare_system.review_system.get_hospital_reviews(hospital_id))
            hospitals_with_reviews.append((hospital, reviews_count))
    
    # Sort by number of reviews (most reviewed first)
    hospitals_with_reviews.sort(key=lambda x: x[1], reverse=True)
    
    if hospitals_with_reviews:
        st.write("**Most Reviewed Hospitals:**")
        cols = st.columns(3)
        for idx, (hospital, review_count) in enumerate(hospitals_with_reviews[:6]):
            with cols[idx % 3]:
                # Create a unique key for each button
                button_key = f"nav_{hospital['id']}_{idx}"
                
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; margin: 0.5rem 0;'>
                    <h4 style='margin: 0 0 0.5rem 0; color: #ff0000;'>{hospital['name']}</h4>
                    <p style='margin: 0.2rem 0; color: #666; font-size: 0.9em;'>{hospital['city']}</p>
                    <p style='margin: 0.2rem 0; color: #666; font-size: 0.9em;'>📝 {review_count} reviews</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add button to navigate to hospital page
                if st.button("Visit Hospital Page →", key=button_key, use_container_width=True):
                    st.session_state.search_triggered = True
                    st.session_state.show_reviews = False
                    st.session_state.search_filters = {
                        'city': hospital['city'],
                        'condition': 'General Checkup',  # Default to general checkup
                        'min_rating': 3.0,
                        'max_distance': 100,
                        'require_icu': False,
                        'require_ambulance': False,
                        'user_lat': healthcare_system.get_city_coordinates(hospital['city'])[0],
                        'user_lng': healthcare_system.get_city_coordinates(hospital['city'])[1]
                    }
                    # Store the specific hospital to highlight
                    st.session_state.highlight_hospital_id = hospital['id']
                    st.rerun()

def show_login_form(healthcare_system):
    """Enhanced login form with multi-language support"""
    # Language selector
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        selected_language = st.selectbox(
            "🌐 Language",
            ["english", "telugu", "hindi"],
            format_func=lambda x: x.title()
        )
    
    translations = healthcare_system.language_support
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #ff0000 0%, #990000 50%, #000000 100%); padding: 3rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.3);'>
        <h1 style='color: white; text-align: center; margin-bottom: 1rem; font-size: 3rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            🔐 {translations.get_translation(selected_language, 'login').upper()}
        </h1>
        <p style='color: white; text-align: center; font-size: 1.2rem; opacity: 0.9;'>
            AI-Powered Healthcare Access
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); padding: 2.5rem; border-radius: 15px; border: 2px solid #ff0000; box-shadow: 0 8px 20px rgba(255,0,0,0.2);'>
            """, unsafe_allow_html=True)
            
            # Username field
            st.markdown(f"<h3 style='color: #ff6b6b; margin-bottom: 1rem;'>👤 {translations.get_translation(selected_language, 'login')}</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", key="login_username", label_visibility="collapsed")
            
            # Password field
            st.markdown(f"<h3 style='color: #ff6b6b; margin-top: 1.5rem; margin-bottom: 1rem;'>🔒 Password</h3>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password", label_visibility="collapsed")
            
            # Login button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                login_button = st.form_submit_button(
                    f"🚀 {translations.get_translation(selected_language, 'login').upper()} NOW", 
                    use_container_width=True,
                    type="primary"
                )
            
            st.markdown("""
            </div>
            """, unsafe_allow_html=True)
            
            # Demo credentials info
            with st.expander("🔍 Demo Credentials (Click to View)"):
                st.markdown("""
                <div style='background: #2d2d2d; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff0000;'>
                <h4 style='color: #ff6b6b; margin-bottom: 0.5rem;'>For Testing Purpose:</h4>
                <p style='color: white; margin: 0.2rem 0;'>👤 <strong>Username:</strong> patient | 🔒 <strong>Password:</strong> patient123</p>
                <p style='color: white; margin: 0.2rem 0;'>👤 <strong>Username:</strong> doctor | 🔒 <strong>Password:</strong> doctor123</p>
                <p style='color: white; margin: 0.2rem 0;'>👤 <strong>Username:</strong> admin | 🔒 <strong>Password:</strong> admin123</p>
                </div>
                """, unsafe_allow_html=True)
        
        if login_button:
            if username and password:
                if username in healthcare_system.users:
                    if healthcare_system.security.verify_password(password, healthcare_system.users[username]['password']):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = healthcare_system.users[username]['role']
                        st.success(f"🎉 Welcome back, {username}! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid password! Please try again.")
                else:
                    st.error("❌ User not found! Check the demo credentials.")
            else:
                st.error("⚠️ Please enter both username and password!")

def show_ai_symptom_checker(healthcare_system, active_tab=None):
    """Enhanced AI-Powered Symptom Checker Interface with all agents"""
    st.subheader("🤖 AI Symptom Checker - Multi-Agent System")
    
    # Create tabs for different agent views
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Symptom Analysis", "🩺 Diagnosis Agent", "💊 Treatment Advisor", "📊 Full AI Report"])
    
    # Determine which tab to open based on the active_tab parameter
    tab_to_open = 0  # Default to first tab
    if active_tab == "Symptom Analysis":
        tab_to_open = 0
    elif active_tab == "Diagnosis":
        tab_to_open = 1
    elif active_tab == "Treatment":
        tab_to_open = 2
    elif active_tab == "Full Report":
        tab_to_open = 3
    
    with tab1:
        show_symptom_analysis_agent(healthcare_system, active_tab="Symptom Analysis" if tab_to_open == 0 else None)
    
    with tab2:
        show_diagnosis_agent(healthcare_system, active_tab="Diagnosis" if tab_to_open == 1 else None)
    
    with tab3:
        show_treatment_advisor_agent(healthcare_system, active_tab="Treatment" if tab_to_open == 2 else None)
    
    with tab4:
        show_full_ai_report(healthcare_system, active_tab="Full Report" if tab_to_open == 3 else None)

def show_ai_agents_center():
    """Display AI Agents in the center of the UI (moved from sidebar)"""
    st.markdown("---")
    st.subheader("🤖 AI Medical Agents - Center")
    
    # Create columns for AI agent buttons in the center
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Symptom Analysis", use_container_width=True,
                    help="Analyzes symptoms to identify patterns and affected body systems"):
            st.session_state.show_symptom_checker = True
            st.session_state.active_ai_agent = "Symptom Analysis"
            st.session_state.search_triggered = False
            st.session_state.show_booking = False
            st.session_state.show_directions = False
            st.session_state.show_reviews = False
            st.session_state.highlight_hospital_id = None
            st.rerun()
    
    with col2:
        if st.button("🩺 Diagnosis Agent", use_container_width=True,
                    help="Suggests possible diagnoses based on symptom patterns"):
            st.session_state.show_symptom_checker = True
            st.session_state.active_ai_agent = "Diagnosis"
            st.session_state.search_triggered = False
            st.session_state.show_booking = False
            st.session_state.show_directions = False
            st.session_state.show_reviews = False
            st.session_state.highlight_hospital_id = None
            st.rerun()
    
    with col3:
        if st.button("💊 Treatment Advisor", use_container_width=True,
                    help="Provides treatment recommendations and self-care advice"):
            st.session_state.show_symptom_checker = True
            st.session_state.active_ai_agent = "Treatment"
            st.session_state.search_triggered = False
            st.session_state.show_booking = False
            st.session_state.show_directions = False
            st.session_state.show_reviews = False
            st.session_state.highlight_hospital_id = None
            st.rerun()
    
    with col4:
        if st.button("📊 Full AI Report", use_container_width=True,
                    help="Generates comprehensive medical analysis using all AI agents"):
            st.session_state.show_symptom_checker = True
            st.session_state.active_ai_agent = "Full Report"
            st.session_state.search_triggered = False
            st.session_state.show_booking = False
            st.session_state.show_directions = False
            st.session_state.show_reviews = False
            st.session_state.highlight_hospital_id = None
            st.rerun()

def main():
    # Initialize security and session
    initialize_session()
    
    healthcare = HealthcareSystem()
    
    # Show login form if not logged in
    if not st.session_state.get('logged_in', False):
        show_login_form(healthcare)
        return
    
    # Main application after login
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🏥 AI-Powered Telangana Healthcare Network</h1>
        <p style='color: white; margin: 0.5rem 0 0 0;'>
            Welcome, {st.session_state.username} ({st.session_state.role}) | 
            🌐 {st.session_state.current_language.title()} | 
            <a href="#" onclick="window.location.reload();" style="color: white; text-decoration: none;">🚪 Logout</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check Gemini status
    if hasattr(healthcare.ai_symptom_checker, 'gemini_available'):
        if healthcare.ai_symptom_checker.gemini_available:
            st.success("✅ Gemini AI Connected Successfully! Advanced analysis available.")
        elif GEMINI_AVAILABLE and healthcare.ai_symptom_checker.gemini_api_key:
            st.info("ℹ️ Using local AI models. Gemini API configured but connection failed. Check API key.")
        else:
            st.info("ℹ️ Using local AI models. Add Gemini API key to .env for enhanced analysis.")
    
    # Emergency banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;'>
        <h3 style='margin: 0;'>🚨 Emergency Services: Dial 108 for Ambulance</h3>
        <p style='margin: 0.5rem 0 0 0;'>24/7 Emergency Care Available | Free Government Ambulance Service</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show AI Agents in the center
    show_ai_agents_center()
    
    # **FIXED SIDEBAR WITHOUT GAPS**
    with st.sidebar:
        st.title("🌐 Healthcare Portal")
        
        # Language selection
        selected_language = st.selectbox(
            "Select Language",
            ["english", "telugu", "hindi"],
            index=["english", "telugu", "hindi"].index(st.session_state.current_language),
            key="language_selector"
        )
        st.session_state.current_language = selected_language
        
        # Patient Reviews Hub button
        if st.button("💬 Patient Reviews Hub", use_container_width=True):
            st.session_state.show_reviews = True
            st.session_state.show_symptom_checker = False
            st.session_state.search_triggered = False
            st.session_state.show_booking = False
            st.session_state.show_directions = False
            st.session_state.highlight_hospital_id = None
            st.rerun()
        
        # Main navigation - Hospital Search
        st.subheader("🔍 Find Healthcare")
        
        # Auto-detect location based on city selection
        st.write("📍 Select Your City")
        cities = ["All Cities", "Hyderabad", "Warangal", "Khammam", "Nizamabad", "Karimnagar", "Jangaon", "Gatkesar"]
        selected_city = st.selectbox("Choose City", cities)
        
        # Auto-set coordinates when city is selected
        if selected_city != "All Cities":
            city_lat, city_lng = healthcare.get_city_coordinates(selected_city)
            user_lat = city_lat
            user_lng = city_lng
        else:
            user_lat, user_lng = 17.3850, 78.4867  # Default to Hyderabad
        
        st.write(f"**Coordinates:** {user_lat:.4f}, {user_lng:.4f}")
        
        # Medical needs
        st.write("🎯 Your Medical Needs")
        medical_condition = st.selectbox(
            "Condition/Specialty Needed",
            ["General Checkup", "Emergency Care", "Cardiology", "Neurology", "Orthopedics", 
             "Pediatrics", "Oncology", "Surgery", "ICU Care", "Other"]
        )
        
        # Additional filters
        st.write("⚙️ Additional Filters")
        min_rating = st.slider("Minimum Hospital Rating", 3.0, 5.0, 3.5, 0.1)
        max_distance = st.slider("Maximum Distance (km)", 5, 200, 50)
        require_icu = st.checkbox("Requires ICU Facilities")
        require_ambulance = st.checkbox("Ambulance Service Needed")
        
        if st.button("🔍 Search Hospitals", type="primary", use_container_width=True):
            st.session_state.search_triggered = True
            st.session_state.show_symptom_checker = False
            st.session_state.show_booking = False
            st.session_state.show_directions = False
            st.session_state.show_reviews = False
            st.session_state.highlight_hospital_id = None
            st.session_state.search_filters = {
                'city': selected_city,
                'condition': medical_condition,
                'min_rating': min_rating,
                'max_distance': max_distance,
                'require_icu': require_icu,
                'require_ambulance': require_ambulance,
                'user_lat': user_lat,
                'user_lng': user_lng
            }
            st.rerun()
    
    # AI Symptom Checker View with all agents
    if st.session_state.get('show_symptom_checker', False):
        show_ai_symptom_checker(healthcare, active_tab=st.session_state.active_ai_agent)
        return
    
    # Patient Reviews Hub View
    if st.session_state.get('show_reviews', False):
        show_patient_reviews_hub(healthcare)
        return
    
    # Check if we should show directions
    if st.session_state.get('show_directions', False) and st.session_state.directions_hospital is not None:
        filters = st.session_state.get('search_filters', {})
        user_lat = filters.get('user_lat', 17.3850)
        user_lng = filters.get('user_lng', 78.4867)
        show_directions_map(st.session_state.directions_hospital, user_lat, user_lng)
        
        if st.button("← Back to Hospital List"):
            st.session_state.show_directions = False
            st.session_state.directions_hospital = None
            st.rerun()
        return
    
    # Check if we should show appointment booking
    if st.session_state.get('show_booking', False) and st.session_state.selected_hospital is not None:
        hospital = st.session_state.selected_hospital
        
        st.subheader(f"📅 Schedule Visit at {hospital['name']}")
        
        # Add back button
        if st.button("← Back to Hospital List"):
            st.session_state.show_booking = False
            st.session_state.selected_hospital = None
            st.session_state.appointment_submitted = False
            st.session_state.search_triggered = True
            st.rerun()
        
        # Show appointment form if not submitted
        if not st.session_state.get('appointment_submitted', False):
            with st.form("appointment_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    patient_name = st.text_input("Patient Full Name*")
                    patient_age = st.number_input("Age*", min_value=1, max_value=120, value=30)
                    patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                    patient_phone = st.text_input("Phone Number*", placeholder="+91 XXXXXXXXXX")
                
                with col2:
                    appointment_date = st.date_input("Preferred Date*", min_value=datetime.now().date())
                    appointment_time = st.selectbox("Preferred Time*", 
                                                  ["09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", 
                                                   "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM"])
                    symptoms = st.text_area("Symptoms/Reason for Visit*")
                    insurance_number = st.text_input("Insurance Number (if applicable)")
                
                emergency_contact = st.text_input("Emergency Contact Number")
                additional_notes = st.text_area("Additional Notes")
                
                # Terms and conditions
                agree_terms = st.checkbox("I agree to the terms and conditions*")
                agree_privacy = st.checkbox("I agree to the privacy policy*")
                
                submitted = st.form_submit_button("Confirm Schedule Request", type="primary")
                
                if submitted:
                    if all([patient_name, patient_phone, symptoms, agree_terms, agree_privacy]):
                        # Simulate appointment scheduling
                        with st.spinner("Scheduling your visit..."):
                            time.sleep(2)
                        
                        # Store appointment details
                        st.session_state.appointment_details = {
                            'hospital': hospital['name'],
                            'date': appointment_date,
                            'time': appointment_time,
                            'patient': patient_name,
                            'reference_id': secrets.token_hex(8).upper()
                        }
                        st.session_state.appointment_submitted = True
                        st.rerun()
                    else:
                        st.error("Please fill all required fields (*) and agree to terms.")
        else:
            # Show confirmation message
            appointment = st.session_state.appointment_details
            st.success(f"""
            ✅ Visit Scheduled Successfully!
            
            **Appointment Details:**
            - Hospital: {appointment['hospital']}
            - Date: {appointment['date']} at {appointment['time']}
            - Patient: {appointment['patient']}
            - Reference ID: {appointment['reference_id']}
            
            The hospital will contact you within 30 minutes to confirm.
            """)
            
            if st.button("Book Another Appointment"):
                st.session_state.show_booking = False
                st.session_state.selected_hospital = None
                st.session_state.appointment_submitted = False
                st.rerun()
        return
    
    # Main content area - Hospital search results
    if st.session_state.get('search_triggered', False):
        filters = st.session_state.get('search_filters', {})
        user_lat = filters.get('user_lat', 17.3850)
        user_lng = filters.get('user_lng', 78.4867)
        
        # Set default values for missing filters
        min_rating = filters.get('min_rating', 3.0)
        max_distance = filters.get('max_distance', 50)
        require_icu = filters.get('require_icu', False)
        require_ambulance = filters.get('require_ambulance', False)
        
        # Filter hospitals based on specialty
        specialty = filters.get('condition')
        filtered_hospitals = healthcare.filter_hospitals_by_specialty(specialty, filters.get('city'))
        
        # Apply rating filter
        if not filtered_hospitals.empty and 'rating' in filtered_hospitals.columns:
            filtered_hospitals = filtered_hospitals[filtered_hospitals['rating'] >= min_rating]
        
        # Apply facility filters
        if require_icu:
            filtered_hospitals = filtered_hospitals[filtered_hospitals['icu_beds'] > 0]
        if require_ambulance:
            filtered_hospitals = filtered_hospitals[filtered_hospitals['ambulance_services'] == True]
        
        # Calculate distances and apply distance filter
        nearby_hospitals = []
        
        for _, hospital in filtered_hospitals.iterrows():
            distance = healthcare.calculate_distance(
                user_lat, user_lng, 
                hospital['latitude'], hospital['longitude']
            )
            if distance <= max_distance:
                hospital_data = hospital.copy()
                hospital_data['distance_km'] = round(distance, 2)
                nearby_hospitals.append(hospital_data)
        
        # If no hospitals found after all filters, show all hospitals in the city
        if len(nearby_hospitals) == 0:
            st.info(f"💡 **Note**: No hospitals found matching all your criteria in {filters.get('city', 'All Cities')}. Showing all available hospitals in {filters.get('city', 'All Cities')}.")
            if filters.get('city') and filters.get('city') != "All Cities":
                # Get all hospitals in the city
                nearby_hospitals = []
                city_hospitals = healthcare.get_hospitals_by_city(filters.get('city'))
                for _, hospital in city_hospitals.iterrows():
                    distance = healthcare.calculate_distance(
                        user_lat, user_lng, 
                        hospital['latitude'], hospital['longitude']
                    )
                    hospital_data = hospital.copy()
                    hospital_data['distance_km'] = round(distance, 2)
                    nearby_hospitals.append(hospital_data)
        
        # Display results
        if nearby_hospitals:
            st.success(f"🏥 Found {len(nearby_hospitals)} hospitals matching your criteria")
            
            # Sort by distance
            nearby_hospitals.sort(key=lambda x: x['distance_km'])
            
            # Check if we need to highlight a specific hospital
            highlight_hospital_id = st.session_state.get('highlight_hospital_id')
            
            # Display hospital cards
            for idx, hospital in enumerate(nearby_hospitals, 1):
                highlight = hospital['id'] == highlight_hospital_id
                
                # Create columns for hospital card
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    display_hospital_details(hospital, show_full_details=True, highlight=highlight)
                
                with col2:
                    # Action buttons
                    st.write("")
                    if st.button(f"📅 Schedule Visit", key=f"schedule_{hospital['id']}", use_container_width=True):
                        st.session_state.show_booking = True
                        st.session_state.selected_hospital = hospital
                        st.rerun()
                    
                    if st.button(f"🗺️ Get Directions", key=f"directions_{hospital['id']}", use_container_width=True):
                        st.session_state.show_directions = True
                        st.session_state.directions_hospital = hospital
                        st.rerun()
                    
                    if st.button(f"💬 View Reviews", key=f"reviews_{hospital['id']}", use_container_width=True):
                        # Show reviews inline
                        show_patient_reviews(healthcare, hospital['id'], hospital['name'])
                    
                    # Quick info
                    st.info(f"""
                    **Quick Info:**
                    • Distance: {hospital['distance_km']} km
                    • Wait Time: {hospital['wait_time_emergency']} min
                    • Rating: {hospital['rating']}/5.0
                    """)
                
                # Separator between hospitals
                if idx < len(nearby_hospitals):
                    st.markdown("---")
        else:
            st.warning("No hospitals found matching your criteria. Try adjusting your filters.")
            
            # Show suggestions
            st.info("💡 **Suggestions:**")
            st.write("• Try increasing the maximum distance")
            st.write("• Lower the minimum rating requirement")
            st.write("• Try a different city or 'All Cities' option")
            st.write("• Select 'General Checkup' for maximum options")
            
            if st.button("Show All Hospitals in Telangana"):
                # Reset filters to show all hospitals
                st.session_state.search_filters = {
                    'city': 'All Cities',
                    'condition': 'General Checkup',
                    'min_rating': 3.0,
                    'max_distance': 200,
                    'require_icu': False,
                    'require_ambulance': False,
                    'user_lat': 17.3850,
                    'user_lng': 78.4867
                }
                st.rerun()
    else:
        # Welcome screen - show when no search is triggered
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.3);'>
            <h1 style='color: white; margin-bottom: 1rem; font-size: 3.5rem;'>🚀 Welcome to AI Healthcare Network</h1>
            <p style='font-size: 1.5rem; margin-bottom: 2rem;'>
                Intelligent Healthcare Solutions for Telangana
            </p>
            <div style='display: flex; justify-content: center; gap: 2rem;'>
                <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; width: 200px;'>
                    <h3 style='color: white;'>🔍 Find</h3>
                    <p>Hospitals & Specialists</p>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; width: 200px;'>
                    <h3 style='color: white;'>🤖 AI Agents</h3>
                    <p>Symptom Analysis</p>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; width: 200px;'>
                    <h3 style='color: white;'>💬 Reviews</h3>
                    <p>Patient Experiences</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show AI Agents in the center (already shown above)
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Healthcare Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Hospitals", len(healthcare.hospitals))
        
        with col2:
            total_beds = healthcare.hospitals['beds_total'].sum()
            available_beds = healthcare.hospitals['beds_available'].sum()
            st.metric("Available Beds", f"{available_beds}/{total_beds}")
        
        with col3:
            total_icu = healthcare.hospitals['icu_beds'].sum()
            st.metric("ICU Beds", total_icu)
        
        # Emergency information - CHANGED TO VERTICAL LAYOUT
        st.markdown("---")
        st.subheader("🚨 Emergency Information")
        
        # Changed from horizontal columns to vertical layout
        st.info("""
        **Emergency Numbers:**
        • Ambulance: **108** (Free)
        • Police: **100**
        • Fire: **101**
        • Women Safety: **1091**
        
        **24/7 Emergency Hospitals:**
        • All major hospitals listed
        • Government hospitals: Free emergency care
        • Private hospitals: Insurance accepted
        
        **Important Emergency Tips:**
        • Call 108 immediately for ambulance service
        • Government ambulances are free across Telangana
        • Major hospitals have 24/7 emergency departments
        • Keep emergency contact numbers saved
        """)
        
        # Quick tips - CHANGED TO VERTICAL LAYOUT
        st.markdown("---")
        st.subheader("💡 Quick Tips for Users")
        
        tips = """
        • Use the sidebar to find hospitals by city and specialty
        • Try our AI Symptom Checker for preliminary analysis
        • Read patient reviews before selecting a hospital
        • Check hospital facilities (ICU, ambulance, etc.)
        • Compare consultation fees and wait times
        • Use the directions feature for navigation
        • Schedule appointments in advance when possible
        • Always verify hospital information before visiting
        • Keep your medical records handy
        """
        
        st.info(tips)

if __name__ == "__main__":
    main()