import os
import streamlit as st
import pandas as pd
import docx
from typing import Dict

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain.chains import LLMChain
import json
from datetime import datetime
import openai

# Streamlit Configuration
st.set_page_config(
    page_title="COMPASS - University Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'preferences' not in st.session_state:
    st.session_state.preferences = {}
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'initial_recommendations' not in st.session_state:
    st.session_state.initial_recommendations = None

# Constants
US_REGIONS = {
    "Northeast": ["Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut", "New York", "Pennsylvania", "New Jersey"],
    "Southeast": ["Maryland", "Delaware", "Virginia", "West Virginia", "Kentucky", "Tennessee", "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama", "Mississippi", "Arkansas", "Louisiana"],
    "Midwest": ["Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin", "Minnesota", "Iowa", "Missouri", "North Dakota", "South Dakota", "Nebraska", "Kansas"],
    "Southwest": ["Texas", "Oklahoma", "New Mexico", "Arizona"],
    "West": ["Colorado", "Wyoming", "Montana", "Idaho", "Washington", "Oregon", "Utah", "Nevada", "California", "Alaska", "Hawaii"]
}

# Sidebar Configuration
with st.sidebar:
    st.header("📋 Your Preferences")

    # Initialize default preferences
    default_preferences = {
        "field_of_study": "Computer Science",
        "budget_min": 20000,
        "budget_max": 50000,
        "preferred_locations": [],
        "weather_preference": "Moderate",
    }

    # Use session state to track if preferences are set
    if "preferences_set" not in st.session_state:
        st.session_state.preferences_set = False
        st.session_state.user_preferences = default_preferences

    field_of_study = st.selectbox(
        "Field of Study",
        [
            "Computer Science",
            "Engineering",
            "Business",
            "Sciences",
            "Arts",
            "Other",
        ],
        index=[
            "Computer Science",
            "Engineering",
            "Business",
            "Sciences",
            "Arts",
            "Other",
        ].index(
            st.session_state.user_preferences.get(
                "field_of_study", "Computer Science"
            )
        ),
    )

    budget_range = st.slider(
        "Budget Range (USD/Year)",
        0,
        100000,
        value=(
            st.session_state.user_preferences.get("budget_min", 20000),
            st.session_state.user_preferences.get("budget_max", 50000),
        ),
    )

    preferred_location = st.multiselect(
        "Preferred Locations",
        ["Northeast", "Southeast", "Midwest", "Southwest", "West Coast"],
        default=st.session_state.user_preferences.get("preferred_locations", []),
    )

    weather_preference = st.select_slider(
        "Weather Preference",
        options=["Cold", "Moderate", "Warm", "Hot"],
        value=st.session_state.user_preferences.get(
            "weather_preference", "Moderate"
        ),
    )

    if st.button("💾 Save Preferences"):
        user_prefs = {
            "field_of_study": field_of_study,
            "budget_min": budget_range[0],
            "budget_max": budget_range[1],
            "preferred_locations": preferred_location,
            "weather_preference": weather_preference,
        }

        # Update session state
        st.session_state.user_preferences = user_prefs
        st.session_state.preferences_set = True
        st.success("✅ Preferences saved successfully!")

# Enhanced system prompt
SYSTEM_PROMPT = """You are a highly knowledgeable university advisor for international students. Your role is to provide detailed, actionable advice that helps students make informed decisions about their education in the United States. 

When providing initial recommendations:
1. Focus on the top 3 universities that best match the student's preferences
2. For each university provide:
   - Brief overview of the program strength
   - Specific costs and potential scholarships
   - Location benefits and climate match
   - Notable features or advantages
3. Keep the initial recommendations concise but informative

When answering follow-up questions:
1. Provide detailed, specific information about asked universities
2. Compare and contrast options when relevant
3. Include practical next steps and actionable advice
4. Focus on international student perspective

Remember to:
- Consider the complete student context (field, budget, location preferences)
- Be realistic about challenges and opportunities
- Provide evidence-based recommendations
- Maintain an encouraging and supportive tone"""

# Set API keys from Streamlit secrets
try:
    st.write("🔑 Verifying API Keys...")
    openai_key = st.secrets["open-key"]
    pinecone_key = st.secrets["pinecone_api_key"]
    openweather_key = st.secrets["open-weather"]
    st.write(f"✅ OpenAI API Key Loaded: {openai_key[:5]}...masked")
    st.write(f"✅ Pinecone API Key Loaded: {pinecone_key[:5]}...masked")
except Exception as e:
    st.error(f"❌ Error accessing secrets: {e}")
    raise e

# Initialize OpenAI
openai.api_key = openai_key
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", openai_api_key=openai_key)

# User Authentication Functions
def initialize_user_db():
    """Initialize user database if it doesn't exist"""
    if not os.path.exists('user_data'):
        os.makedirs('user_data')
    if not os.path.exists('user_data/users.json'):
        with open('user_data/users.json', 'w') as f:
            json.dump({}, f)

def load_user_data(user_id):
    """Load user data from JSON file"""
    try:
        with open('user_data/users.json', 'r') as f:
            users = json.load(f)
        return users.get(user_id, None)
    except FileNotFoundError:
        initialize_user_db()
        return None

def save_user_data(user_id, data):
    """Save user data to JSON file"""
    try:
        with open('user_data/users.json', 'r') as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    
    users[user_id] = data
    
    with open('user_data/users.json', 'w') as f:
        json.dump(users, f, indent=4)

def save_current_session():
    """Save current session data for the user"""
    if hasattr(st.session_state, 'user_id') and st.session_state.user_id:
        user_data = {
            'preferences': st.session_state.preferences,
            'chat_history': st.session_state.chat_history[-10:],  # Save last 10 conversations
            'initial_recommendations': st.session_state.initial_recommendations,
            'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_user_data(st.session_state.user_id, user_data)

def login_page():
    """Display login page and handle user authentication"""
    st.write("### Welcome to University Assistant")
    st.write("Please enter your username to continue")
    
    user_id = st.text_input("Username:", key="user_id_input",
                           help="Enter your username to access your saved preferences")
    
    if user_id:
        user_data = load_user_data(user_id)
        
        if user_data:
            st.success(f"Welcome back, {user_id}! 👋")
            # Load user preferences and chat history
            st.session_state.preferences = user_data.get('preferences', {})
            st.session_state.chat_history = user_data.get('chat_history', [])
            st.session_state.initial_recommendations = user_data.get('initial_recommendations', None)
            st.session_state.user_id = user_id
            st.session_state.authenticated = True
            
            # Display last login time
            last_login = user_data.get('last_login', 'First time login')
            st.info(f"Last login: {last_login}")
            
            # Update last login time
            user_data['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_user_data(user_id, user_data)
            
            if st.button("Continue to Assistant"):
                st.experimental_rerun()
        else:
            st.info("New user detected! Let's set up your preferences.")
            st.session_state.user_id = user_id
            st.session_state.authenticated = True
            st.session_state.show_chat = False
            
            if st.button("Set Up Preferences"):
                st.experimental_rerun()
    
    return st.session_state.get('authenticated', False)

# Main Application
def main():
    """Main Streamlit Application."""
    st.title("🎓 COMPASS - University Recommendation System")
    
    # Initialize user database
    initialize_user_db()
    
    # Handle authentication
    if not st.session_state.authenticated:
        if login_page():
            st.experimental_rerun()
        return
    
    # Chat Interface
    st.header("💬 Chat with COMPASS")
    if st.button("🗑 Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Create a container for the chat history
    chat_container = st.container()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat Input
    if prompt := st.chat_input(
        "Ask me about universities, programs, costs, or job prospects...",
        disabled=not st.session_state.preferences_set,
    ):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response with typing indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get bot response
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                ).choices[0].message['content'].strip()
                st.write(response)

                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")