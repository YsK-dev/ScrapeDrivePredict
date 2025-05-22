# 🚗 Arabam.com Price Predictor & Scraper 🎯

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)](https://flask.palletsprojects.com/)
[![Selenium](https://img.shields.io/badge/selenium-4.0%2B-orange)](https://www.selenium.dev/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A specialized web application that analyzes listings from Arabam.com to predict accurate vehicle prices. Simply enter a car listing URL from Arabam.com, and the system will estimate its market value using machine learning.

## ⭐ Key Feature: Instant Price Estimation

**✅ URL-Based Price Prediction:** 
- Paste any Arabam.com vehicle listing URL
- Get an instant market value estimation
- See the difference between listed and predicted prices
- Make informed decisions about vehicle purchases or sales

## ✨ Core Capabilities

### 🕷 Web Scraping Engine
- **Targeted Data Collection**: Extracts 25+ vehicle attributes from Arabam.com listings
- **Intelligent Parsing**: Processes Turkish language listings with proper encoding
- **Anti-Detection Measures**: Implements temporary profiles and random delays
- **Error Recovery**: Automatically handles timeouts and unexpected page structures

### 📊 ML-Powered Price Analysis
- **Random Forest Algorithm**: Trained on thousands of Turkish market vehicle listings
- **Market-Specific Features**: 
  - Regional price variations
  - Turkish market value trends
  - Local vehicle preferences
- **Comparison Tools**: Highlights undervalued or overpriced listings

### 🌐 User-Friendly Interface
- **Simple URL Input**: Just paste an Arabam.com listing URL
- **Detailed Results**: View comprehensive analysis of each prediction
- **Data Explorer**: Browse through collected vehicle data
- **Mobile Responsive**: Works on desktop and mobile devices

## 🛠 Technology Stack
- **Backend**: Flask, Python 3.8+
- **Scraping**: Selenium WebDriver, ChromeDriver
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn, joblib
- **Frontend**: HTML5, CSS3, JavaScript

## 🚀 Installation & Usage

### Quick Setup
```bash
git clone [repository-url]
cd ArabamAI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
