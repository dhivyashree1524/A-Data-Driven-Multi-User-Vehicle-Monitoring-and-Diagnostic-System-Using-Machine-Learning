[README.md](https://github.com/user-attachments/files/25662005/README.md)
Data-Driven Multi-User Vehicle Monitoring and Predictive Diagnostic System using Machine Learning

Project Overview

This project implements a software-based intelligent vehicle monitoring
and analytics dashboard that integrates machine learning, anomaly
detection, role-based access, multilingual voice alerts, and automated
report generation --- without requiring IoT hardware.

 Key Features

Multi-User Role-Based Dashboard

-   Driver View -- Basic safety parameters and alerts\
-   Technician View -- Detailed diagnostic metrics\
-   Manufacturer View -- Complete analytics & performance insights

 Machine Learning Integration

-   Random Forest Classifier for vehicle health prediction\
-   Isolation Forest for anomaly detection\
-   Threshold-based real-time fault alerts

 Multilingual Voice Alerts

-   English voice alerts\
-   Tamil voice alerts (pure Tamil phrases)\
-   Automatic speech without page refresh

 Real-Time Analytics Dashboard

-   Dynamic vehicle selection\
-   ML graphs updated per vehicle\
-   Health prediction visualization

 Automated PDF Report Generation

-   One-click downloadable report\
-   Includes ML prediction, anomaly status, and graphs

System Architecture

Vehicle YAML Dataset
\↓\
Data Processing (Pandas)
\↓\
Diagnostic Engine
- Threshold Analysis
- Random Forest
- Isolation Forest
   \↓\
Role-Based Dashboard (Streamlit)
\↓\
Voice Alert Engine
\↓\
PDF Report Generator

Technologies Used

-   Python\
-   Streamlit\
-   Pandas\
-   Scikit-learn\
-   Matplotlib\
-   PyYAML\
-   ReportLab\
-   HTML & JavaScript (Speech Synthesis API)

 Installation

Install required packages:

pip install streamlit pandas scikit-learn matplotlib pyyaml reportlab

How to Run

1.  Clone the repository\
2.  Navigate to project folder\
3.  Run:

streamlit run app.py

4.  Open in browser:

http://localhost:8501



 Dataset

Structured YAML dataset containing: - Engine Temperature - Fuel Level -
Oil Level - Battery Voltage - Tyre Pressure - Engine RPM - Brake
Health - Driver Score - Location - Timestamp



Future Enhancements

-   Cloud deployment\
-   Real-time streaming integration\
-   Deep learning models\
-   REST API support\
-   Mobile app version


 Author

Dhivyashree\
Engineering Project -- Machine Learning & Intelligent Systems

 License

Developed for academic and research purposes.
