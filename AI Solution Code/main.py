import numpy as np
import pandas as pd
import speech_recognition as sr
from gtts import gTTS
import os
import pyaudio
from nltk.stem import WordNetLemmatizer
import pygame
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize lemmatizer
# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Your embedded data with timestamps for time-based indexing
embedded_data = {
   


    "greetings": [
        {"pattern": "hi", "response": "Hi there! How’s your day going so far?", "timestamp": datetime.now() - timedelta(days=1)},
        {"pattern": "hello", "response": "Hello! Ready to tackle today’s challenges?", "timestamp": datetime.now() - timedelta(days=2)},
        {"pattern": "how are you", "response": "Hey, are you crushing your schoolwork today?", "timestamp": datetime.now() - timedelta(days=3)},
        {"pattern": "good morning", "response": "Good morning! Let’s make today productive. What’s on your plate today?", "timestamp": datetime.now() - timedelta(hours=12)},
        {"pattern": "good afternoon", "response": "Good afternoon! Hope everything is going smoothly. What’s your next task?", "timestamp": datetime.now() - timedelta(hours=6)},
        {"pattern": "good evening", "response": "Good evening! Ready to unwind or push through some work? What’s the plan?", "timestamp": datetime.now() - timedelta(hours=4)},
        {"pattern": "hey", "response": "Hey! What’s on the agenda today? Got any big plans?", "timestamp": datetime.now() - timedelta(days=1, hours=3)},
        {"pattern": "what's up", "response": "Not much! What’s up with you? Working on something exciting?", "timestamp": datetime.now() - timedelta(hours=2)},
        {"pattern": "howdy", "response": "Howdy! Hope your day is going great! Got any interesting projects?", "timestamp": datetime.now() - timedelta(days=2, hours=4)},
        {"pattern": "yo", "response": "Yo! What’s happening? Are you tackling anything fun today?", "timestamp": datetime.now() - timedelta(days=3, hours=5)},
        
        # Additional greetings
        {"pattern": "greetings", "response": "Greetings! What’s the latest news from your side?", "timestamp": datetime.now() - timedelta(days=4)},
        {"pattern": "sup", "response": "Sup! How’s everything going for you today?", "timestamp": datetime.now() - timedelta(days=5)},
        {"pattern": "hey there", "response": "Hey there! What’s on your mind today?", "timestamp": datetime.now() - timedelta(hours=8)},
        {"pattern": "yo yo", "response": "Yo yo! How’s it going? Got anything cool going on?", "timestamp": datetime.now() - timedelta(days=1)},
        {"pattern": "hiya", "response": "Hiya! How’s life treating you today?", "timestamp": datetime.now() - timedelta(hours=18)},
        {"pattern": "morning", "response": "Morning! What’s the first task of the day?", "timestamp": datetime.now() - timedelta(days=1, hours=10)},
        {"pattern": "evening", "response": "Evening! What’s been the highlight of your day so far?", "timestamp": datetime.now() - timedelta(hours=5)},
        {"pattern": "good day", "response": "Good day! How are things shaping up for you?", "timestamp": datetime.now() - timedelta(days=6)},
        {"pattern": "yo man", "response": "Yo man! Got anything exciting in store today?", "timestamp": datetime.now() - timedelta(days=7)},
        {"pattern": "hola", "response": "Hola! How’s your day treating you so far?", "timestamp": datetime.now() - timedelta(days=8)},
        {"pattern": "bonjour", "response": "Bonjour! What are you working on today?", "timestamp": datetime.now() - timedelta(days=9)},
        {"pattern": "what’s good", "response": "What’s good? Have you got something fun in the pipeline?", "timestamp": datetime.now() - timedelta(days=10)},
        {"pattern": "how’s it going", "response": "How’s it going? Are you up to anything productive today?", "timestamp": datetime.now() - timedelta(days=11)},
        {"pattern": "what’s new", "response": "What’s new? Anything exciting happening?", "timestamp": datetime.now() - timedelta(days=12)},
        {"pattern": "hi there", "response": "Hi there! How’s the grind going today?", "timestamp": datetime.now() - timedelta(days=13)},
        {"pattern": "what’s poppin", "response": "What’s poppin? Got any cool projects in the works?", "timestamp": datetime.now() - timedelta(days=14)},
        {"pattern": "yo yo yo", "response": "Yo yo yo! How are things moving along today?", "timestamp": datetime.now() - timedelta(hours=30)},
        {"pattern": "holla", "response": "Holla! What’s happening today?", "timestamp": datetime.now() - timedelta(hours=20)},
        {"pattern": "g’day", "response": "G’day! What’s on your schedule?", "timestamp": datetime.now() - timedelta(days=1, hours=12)},
        {"pattern": "cheers", "response": "Cheers! Hope your day is going smoothly.", "timestamp": datetime.now() - timedelta(hours=16)},
        {"pattern": "what’s cracking", "response": "What’s cracking? Working on anything cool today?", "timestamp": datetime.now() - timedelta(days=3, hours=10)},
        {"pattern": "good to see you", "response": "Good to see you! How’s everything going on your end?", "timestamp": datetime.now() - timedelta(days=5, hours=2)},
        {"pattern": "salutations", "response": "Salutations! What’s on your mind today?", "timestamp": datetime.now() - timedelta(days=7, hours=8)}
    ],


    "campus_events": [
        {"pattern": "What events are happening on campus this week?", "response": 
            "Semester 1:\n- M. R. S Graduation ceremony (16 April 2024)\n"
            "- Semester 2:\n- SAFA CAF-D LICENCE Award ceremony (11 August 2024)\n"
            "- Advisory Board Committee meeting (8 October 2024)\n"
            "- M. R. S Mega-conference (1 November 2024)\n"
            "- W&RSETA Retail fit event (VUT & UJ Partnership)", 
            "timestamp": datetime.now() - timedelta(days=5)}
    ],
    "services_queries": [
        # Emergency Medical Services related patterns
        {"pattern": "what emergency medical services are available", 
         "response": "Immediate medical assistance is available on-site, including paramedics during working shifts."},
        {"pattern": "tell me about the emergency medical services", 
         "response": "We provide immediate medical assistance on-site, with paramedics available during working hours."},
        {"pattern": "do you have emergency services at the clinic", 
         "response": "Yes, the clinic offers emergency medical services on-site, including paramedics during work shifts."},
        {"pattern": "is there emergency medical support at the campus", 
         "response": "Yes, we provide emergency medical assistance on-site, including paramedics on duty during working hours."},
        {"pattern": "what do you offer for emergency medical care", 
         "response": "For emergencies, we have medical support available on-site and paramedics on shift to assist."},
        {"pattern": "how can I get emergency medical help at the clinic", 
         "response": "You can receive immediate medical assistance at the clinic, and paramedics are available during shifts to assist with emergencies."}
    ],
    "lost_and_found": [
        {"pattern": "Where is the lost and found on campus?", "response": 
            "The Access and Visitor Control Unit handles lost and found. It's located at Block A115. "
            "Crimes can be reported to the Investigation Unit, which operates 24/7.", 
            "timestamp": datetime.now() - timedelta(days=4)},
        {"pattern": "I lost my student ID, what should I do?", "response": 
            "Apply for a new student card as soon as possible. Access to campus is denied without one. "
            "For help, visit the Access and Visitor Control Unit or contact the Command Centre.", 
            "timestamp": datetime.now() - timedelta(days=6)}
    ],
    "staff_information": [
        {"pattern": "who is the head of department", "response": 
            "The Head Of Department position is currently vacant. Please check back later for updates."},

        {"pattern": "who are the psychologists", "response": 
            "We have several psychologists, including:\n"
            "- Ms. Mamphoreng Mashiloane (Senior Psychologist): Specializes in mental health counseling and therapy.\n"
            "- Dr. Trishana Soni (Senior Educational Psychologist): Focuses on educational assessments and interventions.\n"
            "- Ms. Zandile Shabangu (Educational Psychologist): Provides support for learning difficulties and educational challenges.\n"
            "- Ms. Bonolo Mophosho (Psychologist): Offers counseling services for various mental health issues."},

        {"pattern": "who are the psychometrists", "response": 
            "Our psychometrists include:\n"
            "- Ms. Miemie Taukobong: Specializes in psychological testing and assessments.\n"
            "- Ms. Nomangwane Mbele: Focuses on cognitive and educational assessments."},

        {"pattern": "who are the registered counsellors", "response": 
            "We have:\n"
            "- Ms. Selloane Makau: Provides guidance on emotional and personal challenges.\n"
            "- Ms. Bianca Brits: Offers support and counseling for various life situations."},

        {"pattern": "who is the social worker", "response": 
            "Mr. Siphesihle Maseko is our Social Worker, assisting individuals and families in need of support services."},

        {"pattern": "who is the pastoral counsellor", "response": 
            "Ms. Fihliwe Mvundla serves as our Pastoral Counsellor, offering spiritual and emotional support."},

        {"pattern": "who is the chapel keeper", "response": 
            "Ms. Dieketseng Ndaba is the Chapel Keeper, responsible for maintaining the chapel and assisting with spiritual activities."},

        {"pattern": "what does a psychologist do", "response": 
            "Psychologists help individuals understand their emotions, behaviors, and thoughts. They provide counseling, support mental health, and assist with personal development."},

        {"pattern": "what does a psychometrist do", "response": 
            "A psychometrist administers and interprets psychological tests to evaluate cognitive and emotional functioning."},

        {"pattern": "what does a registered counsellor do", "response": 
            "Registered counsellors provide guidance and support for various personal challenges, including mental health and relationship issues."},

        {"pattern": "what does a social worker do", "response": 
            "A social worker helps individuals and families navigate social services and provides support for mental health, housing, and other social issues."},

        {"pattern": "what does a pastoral counsellor do", "response": 
            "A pastoral counsellor provides guidance on spiritual matters, helping individuals find peace and clarity."},

        {"pattern": "tell me about the staff", "response":
            "Our staff consists of psychologists, psychometrists, registered counsellors, a social worker, and a pastoral counsellor. They provide a range of services including mental health support, counseling, assessments, and spiritual guidance."},

        {"pattern": "what kind of support do you offer", "response":
            "We offer psychological assessments, counseling services, tutoring assistance, group therapy sessions, workshops on mental health awareness, and various resources for personal development."},

        {"pattern": "where can I find the clinic", "response":
            "The clinic is located in Building B, Room 205. You can find it near the main entrance."},

        {"pattern": "where is the clinic", "response":
            "The clinic is located in Building B, Room 205. It’s near the main entrance of the campus."},

        {"pattern": "how do I get to the clinic", "response":
            "To reach the clinic, head to Building B and go to Room 205, located near the main entrance."},

        {"pattern": "can you tell me the location of the clinic", "response":
            "Sure! The clinic can be found in Building B, Room 205, close to the main entrance."},

        {"pattern": "where can I find academic advising", "response": 
        "The academic advising office is located in Building B, Room 204."},

        {"pattern": "how do I find the academic advising office",  "response":
            "You can find the academic advising office in Building B, Room 204, just a short walk from the main courtyard."},

        {"pattern": "where is the gym", "response":
            "The gym is located in the Sports Complex, Building F."},
        
        {"pattern": "how can I get to the gym", "response":
            "To get to the gym, head to Building F, which is the Sports Complex."},
        
        {"pattern":"where do I find the cafeteria","response":
            "The cafeteria is located in Building C, right next to the library."},
        
        {"pattern":"where is the library","response":
            "The library is located on the north side of the campus, near Building C."},
      
        {"pattern":"are there any workshops available","response":
            "We regularly host workshops on topics such as stress management, study skills enhancement, and mental wellness strategies. Please check our events calendar for upcoming sessions."},
      
        {"pattern":"how can I book an appointment with a psychologist","response":
            "To book an appointment with a psychologist, please contact our office at [insert contact number] or visit our website to schedule online."},
      
        {"pattern":"do you offer online counseling","response":
            "Yes! We offer online counseling sessions to accommodate those who prefer remote assistance or have scheduling conflicts."}
    ],
  
    
    
    "location_queries": [
        {"pattern": "where is the clinic", "response": "The clinic is located at block N021."},
        {"pattern": "location of the clinic", "response": "You can find the clinic at block N021."},
        {"pattern": "clinic location", "response": "The clinic is in block N021 on campus."}
    ],
    "services": [
        {"pattern": "Comprehensive Primary Health Care Services", "description": "General health services for students and staff."},
        {"pattern": "Chronic Health Care", "description": "Management and support for chronic health conditions."},
        {"pattern": "Underweight Management", "description": "Support and guidance for individuals underweight."},
        # (Other services...)
    ],
  
    "campus_information": [
        {"pattern": "library hours", "response": "The library is open from 8 AM to 6 PM, Monday to Friday."},
        {"pattern": "where is the library", "response": "The library is located on the north side of the campus, near Building C."},
        {"pattern": "admin office open", "response": "The admin office operates from 9 AM to 5 PM, Monday to Thursday, and 9 AM to 3 PM on Fridays."},
        {"pattern": "where is the admin office", "response": "The admin office is located in Building A, Room 101."},
        {"pattern": "cafeteria hours", "response": "The cafeteria is open from 7 AM to 7 PM every weekday."},
        {"pattern": "where is the gym", "response": "The gym is located in the Sports Complex, Building F."},
        {"pattern": "help with studies", "response": "You can visit the academic advising office in Building B, Room 204. Office hours are from 10 AM to 4 PM."},
        {"pattern": "where is tutoring center", "response": "The tutoring center is located in Building D, Room 102. It's open from 9 AM to 5 PM, Monday to Friday."},
        {"pattern": "exam schedule", "response": "You can check your exam schedule through the student portal or at the registrar's office in Building A."},
        {"pattern": "join study group", "response": "You can join study groups by visiting the academic advising office or signing up through the student portal."},
        {"pattern": "social clubs", "response": "There are various social clubs available, including sports, art, tech, and more. Visit the student center to learn more."},
        {"pattern": "mental health", "response": "The mental health counseling office is in Building E, Room 202. Office hours are 9 AM to 4 PM, Monday to Friday."},
        {"pattern": "career advising", "response": "The career advising office offers help with resumes, internships, and job searches. Visit them in Building C, Room 303."},
        {"pattern": "contact", "response": "You can contact the general student helpdesk at +123 456 789 or email support@studenthelp.com."},
        {"pattern": "campus map", "response": "You can download the campus map from the student portal or pick one up at the admin office."}
    ],
 
    'sport_recreation': [
        {'pattern': 'sport and recreation', 'response': 'VUT Sport is the brand name of the VUT Sport & Recreation Department.'},
        {'pattern': 'sport officer', 'response': 'Contact Mr. Phokoane Mapheto at TT6-8, Tel: 016 950 7645 or Email: phokoanem@vutcloud.onmicrosoft.com.'},
        {'pattern': 'stanley shabalala', 'response': 'Mr. Stanley Shabalala is available at TT6-6, Tel: 016 950 9307, Email: nkanyisos@vut.ac.za.'},
        {'pattern': 'student-athletes', 'response': 'We recruit student-athletes and facilitate their affiliation with sports clubs.'},
        {'pattern': 'sports clubs', 'response': 'Athletes must pay a yearly fee to be recognized by governing bodies like USSA.'},
        {'pattern': 'sports merit bursaries', 'response': 'We offer sports merit bursaries based on academic and sports achievements.'},
        {'pattern': 'intern training', 'response': 'We train PR and Sport Management student interns to develop their skills and experience.'},
        {'pattern': 'sponsorships', 'response': 'We seek sponsorships for our sports teams and handle internal and external marketing.'},
        {'pattern': 'available sports', 'response': 'The available sports at VUT include: Hockey, Karate, Rugby, Cricket, Aerobics, Chess, Athletics, Dance Sport, Softball, and Volleyball.'},
        {'pattern': 'transport for games', 'response': 'We ensure transport is available for teams during games.'},
        {'pattern': 'paramedics at games', 'response': 'Paramedics are present during home games for athlete safety.'},
        {'pattern': 'international competitions', 'response': 'Sport Officers travel with selected athletes to international events.'},
        {'pattern': 'monthly reports', 'response': 'Sport Officers submit monthly and quarterly reports on activities.'},
        {'pattern': 'sports office contact', 'response': 'For further enquiries, contact the Sports Office at TT6-3, Tel: 016 950 7645, or Email: sport.recreation@vut.ac.za.'}
    ],


   
    'student_governance': [
        {'pattern': 'student governance', 'response': 'These initiatives enhance student representation and support effective governance operations.'},
        {'pattern': 'governance structures', 'response': 'We operate through recognized Student Governance Structures.'},
        {'pattern': 'student governance elections', 'response': 'We handle student governance elections and transitional processes.'},
        {'pattern': 'administrative support for src', 'response': 'We provide administrative support for the Student Representative Council (SRC) and its sub-structures.'},
        {'pattern': 'project management support', 'response': 'Our team offers project management support for student initiatives and events.'},
    ],
    'student_life': [
        {'pattern': 'student life', 'response': 'Initiatives promote holistic development, self-expression, and leadership among students.'},
        {'pattern': 'development training', 'response': 'We provide student development training and support to enhance skills.'},
        {'pattern': 'co-curricular development', 'response': 'Our focus includes co-curricular development to enrich student experiences.'},
        {'pattern': 'arts and culture initiatives', 'response': 'We offer various arts and culture programs and initiatives for student engagement.'},
        {'pattern': 'critical conversations', 'response': 'We facilitate critical conversations and dialogues to foster social awareness.'},
    ],
    'vibrant_student_life': [
        {'pattern': 'vibrant student life', 'response': 'We create opportunities for self-expression and engage students in community building.'},
        {'pattern': 'extra-curricular programs', 'response': 'We develop extra-curricular programs that enhance a vibrant student experience.'},
    ],
    'contact_info': [
        {'pattern': 'student governance contact', 'response': 'Office N216. For enquiries, call 016 950 9900 - Tebello Theledi (Administrator).'}
    ]
}
