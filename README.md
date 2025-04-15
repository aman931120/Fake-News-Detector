# Fake News Detection Project 📰

This project uses **Logistic Regression** to detect fake news articles based on text classification techniques. It combines a Python backend with a simple web-based frontend for testing news authenticity.

---

## 🔍 Overview

Fake news is a growing problem in the digital age, and this project aims to classify news articles as **Real** or **Fake** using Natural Language Processing (NLP) and machine learning.

---

## 💡 Features

- Text data preprocessing  
- Model training with Logistic Regression  
- Real-time news classification via a web interface  
- Pretrained models included (.pkl files)  

---

## 🗂️ Project Structure

```
Project NEWS/
├── app.py                        # Main backend application
├── model_preprocess.py           # Preprocessing pipeline for text data
├── fake_news_model.pkl           # Initial trained model
├── final_news_model.pkl          # Final trained model
├── final_vectorizer.pkl          # Trained vectorizer for text features
├── Fake.csv                      # Dataset: Fake news articles
├── True.csv                      # Dataset: Real news articles
├── API_Real_News.csv             # Additional news data
└── frontend/
    ├── index.html                # Main HTML page
    ├── news.html                 # News input/output interface
    └── script.js                 # Client-side interaction logic
```
---

## 🚀 How to Run

1. Clone this repository:
    
bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name


2. Install dependencies:
    
bash
    pip install -r requirements.txt


3. Run the application:
    
bash
    python app.py


4. Open your browser and navigate to:
    
http://localhost:5000


---

## ⚙️ Requirements

- Python 3.x  
- Flask  
- scikit-learn  
- pandas  
- joblib  

---

## 📊 Model Info

- **Algorithm:** Logistic Regression  
- **Libraries Used:** scikit-learn, pandas  
- **Input:** News article text  
- **Output:** Prediction (Fake or Real)  

---

## 📸 Demo

Here are some screenshots to demonstrate the application:

### Example of Real News Prediction:
![Real News Prediction](path/to/real_news_image.png)

### Example of Fake News Prediction:
![Fake News Prediction](path/to/fake_news_image.png)

---

## 📌 Note

- Make sure your Fake.csv and True.csv datasets are properly formatted.  
- Pre-trained model files (.pkl) are included, so you can test right away.  

---

## 💻 Author

Your Name  
[GitHub](https://github.com/yourusername)  

---

## 📢 License

This project is open-source under the MIT License.

i want to add 2 pictures for demo 
