# COMPASS-The-AI-Bot
Comprehensive Master’s Program Assistant for Student Success

The goal of this project is to create an intelligent chatbot that assists international students, 
particularly those pursuing a master’s degree in the United States, in selecting the right university 
and timing their studies. The system will provide tailored recommendations based on factors 
such as cost of living, weather, job market outlook, and university strengths for specific courses, 
helping students make informed decisions. A further aim of the project is to help students 
identify potential scholarship opportunities from both university and external sources, enhancing 
financial support options and facilitating affordability for international students. 

Project Functionality and Use 
The chatbot will operate as a conversational assistant, guiding international students in making 
informed university choices through a user-friendly web interface. Key functionalities include: 

• User-Friendly Web Interface: 

    o The chatbot’s web interface will be designed for ease of use, featuring dropdowns, 
    interactive prompts, and easy navigation to help users enter preferences such as 
    study location, budget, field of study, and climate etc. 
    o Users can explore recommendations, filtering options and refining their inputs as 
    they gain insights from the chatbot’s responses. 
    
• Session-Based Memory and Persistent Context: 

    o Using ChromaDB as a vector database, the chatbot will store user preferences and 
    interaction histories as embeddings. This enables memory persistence across 
    sessions, allowing students to pick up where they left off, even after logging out. 
    o The system’s memory feature is designed to ensure continuity, enabling users to 
    receive consistent, personalized guidance throughout their search. It recalls and 
    applies previous preferences, making refined recommendations as users progress 
    and adjust their input.
    
• Personalized and Context-Aware Recommendations: 

    o Based on user-provided criteria (e.g., location, study field, budget), the chatbot 
    will recommend universities, identify relevant job markets by state, and highlight 
    strengths in specific programs and anticipated industry trends. 
    o Retrieval-Augmented Generation (RAG) is applied, where user queries trigger the 
    retrieval of contextually relevant embeddings from ChromaDB. This method 
    ensures that recommendations adapt to user preferences in real-time, providing 
    responses grounded in relevant data points. 
    o Although job market trends are currently in a work-in-progress phase, periodic 
    updates from static sources will ensure that the chatbot can provide general 
    guidance on sector demands and hiring trends. 
    
• Scholarship Identification: 

    o The chatbot will further assist students by identifying scholarship opportunities 
    both from universities and external sources. By using structured data on eligibility 
    criteria, application deadlines, and funding sources, the chatbot provides 
    suggestions tailored to user qualifications and financial needs.

