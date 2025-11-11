# 🧠 TrackBot — Personal Finance Manager

**TrackBot** is a Django + MySQL–based personal finance management system that helps users manage, analyze, and predict their financial activities with ease.  
It combines **data analysis**, **AI chatbot**, and **ML-powered forecasting** for a complete expense tracking solution.

---

## 🚀 Features

✅ Add, edit, and view **expenses & income**  
✅ Categorize and visualize **monthly spending trends**  
✅ Real-time **email alerts** when spending exceeds monthly limits  
✅ AI-powered **finance chatbot** (built using `sentence-transformers`)  
✅ Predict **future expenses** via **Linear Regression**  
✅ Detailed **matplotlib**-based graphs for analysis  
✅ MySQL database integration for reliability and scalability  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | Django (Python) |
| **Database** | MySQL (`PyMySQL`) |
| **Frontend** | HTML, CSS, Bootstrap |
| **AI / NLP** | Sentence Transformers |
| **Machine Learning** | Scikit-Learn (Linear Regression) |
| **Data Handling** | Pandas, Matplotlib |
| **Environment** | Python 3.11+ |

---

## Project Structure

TrackBot/
├── manage.py
├── Database.py
├── database.sql
├── personal_finance_manager/        # Django settings and URLs
├── UserApp/                         # Main app (views, chatbot, ML)
├── Templates/                       # HTML files
├── Static/                          # CSS, JS, images
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
