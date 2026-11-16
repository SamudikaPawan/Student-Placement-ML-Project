# uses gradio to make a simple ui for the model to use with hugging face
import gradio as gr, pandas as pd, joblib

# import the python objects generated from the notebook
bundle = joblib.load("student_placement_model.pkl")
model, scaler = bundle["model"], bundle["scaler"]
scale_cols, features = bundle["scale_cols"], bundle["features"]

typical = dict(zip(scale_cols, scaler.mean_)) # average value for each feature used when a field is left blank

def predict(gpa, gpa_scale, attendance_percentage, internships_count, projects_count, backlogs,
            linkedin_connections, coding_skill_score, aptitude_score, communication_skill_score,
            logical_reasoning_score, mock_interview_score, extracurricular_score, leadership_score):
    
    cgpa = gpa * 10 / gpa_scale # convert 4.0 GPA to the 10.0 scale of the training data
    
    inputs = {"cgpa": cgpa,
              "attendance_percentage": attendance_percentage,
              "internships_count": internships_count,
              "projects_count": projects_count,
              "backlogs": backlogs,
              "linkedin_connections": linkedin_connections,
              "coding_skill_score": coding_skill_score,
              "aptitude_score": aptitude_score,
              "communication_skill_score": communication_skill_score,
              "logical_reasoning_score": logical_reasoning_score,
              "mock_interview_score": mock_interview_score,
              "extracurricular_score": extracurricular_score,
              "leadership_score": leadership_score}
    
    optional = ["linkedin_connections", "coding_skill_score", "aptitude_score", "communication_skill_score",
                "logical_reasoning_score", "mock_interview_score", "extracurricular_score", "leadership_score"]
    
    for k in optional: # blank optional fields gets filled with its average value
        if not inputs[k]:
            inputs[k] = typical[k]
            
    row = pd.DataFrame([inputs])
    row[["standardized_" + c for c in scale_cols]] = scaler.transform(row[scale_cols])
    p = model.predict_proba(row[features])[0, 1]
    return {"Placed": float(p), "Not Placed": float(1 - p)}

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(3.0, label="CGPA / GPA"),
            gr.Radio([4, 10], value=4, label="GPA scale of your university (4.0 or 10)"),
            gr.Slider(50, 100, 85, label="Attendance %"),
            gr.Slider(0, 8, 0, step=1, label="Internships"),
            gr.Slider(0, 13, 0, step=1, label="Projects"),
            gr.Slider(0, 6, 0, step=1, label="Backlogs"),
            gr.Number(None, label="LinkedIn connections (optional)"),
            gr.Number(None, label="Coding skill score 0-100 (optional)"),
            gr.Number(None, label="Aptitude score 0-100 (optional)"),
            gr.Number(None, label="Communication skill score 0-100 (optional)"),
            gr.Number(None, label="Logical reasoning score 0-100 (optional)"),
            gr.Number(None, label="Mock interview score 0-100 (optional)"),
            gr.Number(None, label="Extracurricular score 0-100 (optional)"),
            gr.Number(None, label="Leadership score 0-100 (optional)")],
    
    outputs=gr.Label(label="Placement prediction"),
    title="Student Placement Predictor")

if __name__ == "__main__":
    demo.launch()