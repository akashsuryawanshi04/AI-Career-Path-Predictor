
def get_career_info(career_name):

    career_data = {

        "AI Engineer": {
            "description": "AI Engineers build intelligent systems using machine learning and deep learning.",
            "skills": [
                "Python",
                "TensorFlow",
                "PyTorch",
                "Deep Learning",
                "NLP",
                "MLOps"
            ]
        },

        "Full Stack Developer": {
            "description": "Full Stack Developers build frontend and backend web applications.",
            "skills": [
                "React",
                "Node.js",
                "MongoDB",
                "REST API",
                "JavaScript",
                "HTML/CSS"
            ]
        },

        "Data Analyst": {
            "description": "Data Analysts analyze data and generate business insights.",
            "skills": [
                "SQL",
                "Power BI",
                "Excel",
                "Python",
                "Data Visualization",
                "Statistics"
            ]
        },

        "Cybersecurity Engineer": {
            "description": "Cybersecurity Engineers protect systems and networks from cyber attacks.",
            "skills": [
                "Networking",
                "Ethical Hacking",
                "Linux",
                "Security Tools",
                "Cloud Security",
                "Penetration Testing"
            ]
        }
    }

    return career_data.get(career_name, {})
