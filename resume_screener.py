# Resume Screening functionality using Gemini AI

import os
import re
import nltk
import PyPDF2
from google import genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

def download_nltk_resources():
    """Ensure required NLTK resources are available"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

class ResumeScreener:
    def __init__(self):
        load_dotenv()
        self.api_key =  os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in .env or pass it directly.")
        # Ensure NLTK resources are downloaded
        download_nltk_resources()

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Technical skills for software engineers
        self.technical_skills = [
            "Python", "Java", "JavaScript", "C", "Go", 
            "SQL", "NoSQL", "MongoDB", "MySQL", 
            "Docker", "Kubernetes", "AWS", "Azure","REST API",
            "GraphQL", "React", "Node.js", "Django", "Flask",
            "Microservices", "Git", "CI/CD", "Jenkins","RAG","LangChain",
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "Agile", "Scrum", "Data Structures", "Algorithms", "OOP"
        ]
        
        # Initialize stop words
        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_pdf(self, file):
        """Extract text from PDF resume"""
        text = ""
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        except Exception as e:
            text = f"Error extracting text: {str(e)}"
        return text
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        skills = []
        for skill in self.technical_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                skills.append(skill)
        return skills
    
    def extract_experience(self, text):
        """Extract years of experience from resume text"""
        patterns = [
            r'(\d+)(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
            r'experience\s+of\s+(\d+)(?:\+)?\s*(?:years?|yrs?)',
            r'(?:worked|working)\s+for\s+(\d+)(?:\+)?\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[0])
        
        return None
    
    def analyze_education(self, text):
        """Analyze education level from resume text"""
        education_levels = {
            "phd": ["phd", "ph.d", "doctor of philosophy"],
            "masters": ["masters", "ms", "m.s", "master of", "msc", "m.sc", "mba"],
            "bachelors": ["bachelors", "bachelor of", "bs", "b.s", "b.tech", "btech", "be", "b.e"],
            "associate": ["associate", "a.s", "as degree"]
        }
        
        education = []
        for level, keywords in education_levels.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    education.append(level)
                    break
        
        return list(set(education))
    
    def match_job_description(self, resume_text, job_description):
        """Match resume with job description using cosine similarity"""
        vectorizer = CountVectorizer(stop_words='english')
        
        if len(resume_text.split()) < 5 or len(job_description.split()) < 5:
            return 0.0
        
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity
    
    def gemini_evaluation(self, resume_text, job_description, skills_matched, skills_missing):
        """Use Gemini to generate a structured evaluation"""
        prompt = f"""
        You are an HR assistant AI. Evaluate the following resume against the job description.
        
        Task: Compare resumes with job descriptions and give a recommendation. 
        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Skills Matched: {skills_matched}
        Skills Missing: {skills_missing}
        

        Now follow these steps:  
        1. Compare candidate’s skills, experience and degree with job requirements. 
        4. Provide a brief analysis of strengths and weaknesses.
        
        Please return the response in text format with the following fields:
        - recommendation: a short recommendation 
        - strengths: list of strengths
        - weaknesses: list of weaknesses
        
        
        """
        response = self.client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt
        )
        return response.text


    def gemini_scorer(self, resume_text, job_description, skills_matched, skills_missing):
        """Use Gemini to generate a structured evaluation"""
        prompt = f"""
        You are an HR assistant AI. Evaluate the following resume against the job description to generate a match score.
        
        
        Task: Compare resumes with job descriptions and give a match score. 
        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Skills Matched: {skills_matched}
        Skills Missing: {skills_missing}
        

        Now follow these steps:  
        1. Compare candidate’s skills, experience and degree with job requirements. 
        4. Provide a match score between 0-100.
        Please return the response in json format with the following field:
        - match_score: an integer between 0-100
        
        
        """
        response = self.client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt
        )
        return response.text

    def analyze_resume(self, file, job_description):
        """Main function to analyze the resume against a job description"""
        resume_text = self.extract_text_from_pdf(file) if file else ""
        
        if not resume_text:
            return {
                "error": "Could not extract text from the resume",
                "match_score": 0,
                "skills_matched": [],
                "skills_missing": [],
                "experience_years": None,
                "education": [],
                "recommendation": "Cannot analyze - resume text extraction failed"
            }
        
        candidate_skills = self.extract_skills(resume_text)
        required_skills = self.extract_skills(job_description)
        
        skills_matched = [s for s in candidate_skills if s in required_skills]
        skills_missing = [s for s in required_skills if s not in candidate_skills]
        
        experience_years = self.extract_experience(resume_text)
        education = self.analyze_education(resume_text)
          
        # Get Gemini evaluation
        gemini_analysis= self.gemini_evaluation(
            resume_text, job_description, skills_matched, skills_missing
        )
        match_score= self.gemini_scorer(
            resume_text, job_description, skills_matched, skills_missing
        )
        match_score = int(re.search(r"\d+", match_score).group())
        
        return {
            "match_score": match_score,
            "skills_matched": skills_matched,
            "skills_missing": skills_missing,
            "experience_years": experience_years,
            "education": education,
            "gemini_analysis": gemini_analysis
        }
