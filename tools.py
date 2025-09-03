#tools.py
# Gradio UI  

import gradio as gr
import matplotlib.pyplot as plt

# Import our modules
from resume_screener import ResumeScreener
from sentiment_analyzer import EmployeeSentimentAnalyzer

# Create Gradio UI for Resume Screening
def create_resume_screener_ui():

    resume_screener = ResumeScreener()
    
    def process_resume(resume_file, job_description):
        if resume_file is None:
            return "Please upload a resume"
        
        result = resume_screener.analyze_resume(resume_file, job_description)
        
        # Format output for display
        output = f"""
## Resume Analysis Results

### Match Score: {result['match_score']}/100
### Skills Matched:
{', '.join(result['skills_matched']) if result['skills_matched'] else 'None'}

### Skills Missing:
{', '.join(result['skills_missing']) if result['skills_missing'] else 'None'}

### Experience: 
{f"{result['experience_years']} years" if result['experience_years'] else 'Not detected'}

### Highest Qualification: 
{', '.join(result['education']).title() if result['education'] else 'Not detected'}

### Recommendation:
{result.get('gemini_analysis', 'No recommendation available')}
        """
        
        return output
    
    default_job_description = """
    Software Engineer
    
    Requirements:
    1+ years of experience in software development
    Strong proficiency in Python and JavaScript
    Experience with web frameworks like React or Vue (Angular is a plus)
    Knowledge of databases (SQL, NoSQL)
    Familiarity with cloud services (AWS, Azure, or GCP)
    Hands-on experience with AI/ML frameworks (TensorFlow, PyTorch, or Scikit-learn)
    Exposure to LLM/AI tools (LangChain, Hugging Face, or Gemini AI)
    Experience with version control systems like Git
    Understanding of data structures, algorithms, and applied ML techniques
    Bachelor’s degree in Computer Science, AI/ML, or related field
    Experience with Docker, CI/CD pipelines, and MLOps is a plus
    """
    
    with gr.Blocks(title="Resume Screening Tool") as resume_app:
        gr.Markdown("# Resume Screening Tool for Software Engineer Positions")
        
        with gr.Row():
            with gr.Column():
                resume_file = gr.File(label="Upload Resume (PDF)")
                job_desc = gr.Textbox(
                    label="Job Description",
                    placeholder="Enter job description...",
                    value=default_job_description,
                    lines=10
                )
                submit_btn = gr.Button("Analyze Resume")
            
            with gr.Column():
                output = gr.Markdown(label="Analysis Result")
        
        submit_btn.click(
            fn=process_resume,
            inputs=[resume_file, job_desc],
            outputs=output
        )
    
    return resume_app

# Create Gradio UI for Employee Sentiment Analysis
def create_sentiment_analyzer_ui():
    sentiment_analyzer = EmployeeSentimentAnalyzer()
    
    def process_feedback(feedback, past_feedback=None):
        if not past_feedback:
            past_feedback = []
        
        result = sentiment_analyzer.analyze_feedback(feedback, past_feedback)
        
        # Create visualization for sentiment and risk
        fig, ax2 = plt.subplots(figsize=(6, 4)) 
        
        # Sentiment gauge
        
        
        
        
        # Risk gauge
        risk_score = result["risk_score"] / 100.0 if result["risk_score"] > 1 else result["risk_score"]
        risk_colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Green → Yellow → Red
        risk_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("risk", risk_colors)
        risk_norm = plt.Normalize(0, 1)

        ax2.pie(
            [risk_score, 1 - risk_score],
            colors=[risk_cmap(risk_norm(risk_score)), 'lightgray'],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3)
        )
        ax2.text(0, 0, f"{int(risk_score * 100)}%", ha='center', va='center', fontsize=20)
        ax2.set_title("Attrition Risk")
            
       
        
        recommendations_str = "\n".join([f"- {rec}" for rec in result["recommendations"]]) if result["recommendations"] else "None"
        
        output = f"""
## Employee Feedback Analysis Results

### Sentiment: {result["sentiment"].get("sentiment", "Unknown")}

### Attrition Risk: {result["risk_score"]} 
### Recommended Actions:
{recommendations_str}
        """
        
        return output, fig
    
    default_feedback = """
    
    """
    
    with gr.Blocks(title="Employee Sentiment Analysis Tool") as sentiment_app:
        gr.Markdown("# Employee Sentiment Analysis Tool")
        
        with gr.Row():
            with gr.Column():
                feedback_text = gr.Textbox(
                    label="Employee Feedback",
                    placeholder="Enter employee feedback...",
                    value=default_feedback,
                    lines=10
                )
                submit_btn = gr.Button("Analyze Feedback")
            
            with gr.Column():
                output = gr.Markdown(label="Analysis Result")
                plot_output = gr.Plot()
        
        submit_btn.click(
            fn=process_feedback,
            inputs=[feedback_text],
            outputs=[output, plot_output]
        )
    
    return sentiment_app

# Combined app with tabs
def create_hr_ai_tools_app():
    with gr.Blocks(title="HR Automation") as app:
        gr.Markdown("# AI Tools for HR Professionals")
        
        with gr.Tabs():
            with gr.TabItem("Resume Screening"):
                create_resume_screener_ui()
            
            with gr.TabItem("Sentiment Analysis"):
                create_sentiment_analyzer_ui()
    
    return app

# Entry point for the application
def main():
    print("Starting HR AI Tools...")
    
    # Create and launch the app
    app = create_hr_ai_tools_app()
    print("Launching the web interface...")
    app.launch(share=True)  # Set share=True if you want to generate a public link
    
    print("Web interface closed.")

 