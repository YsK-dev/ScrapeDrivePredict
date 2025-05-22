

# 🚗 Car Ads Scraper & Price Predictor 🎯

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)](https://flask.palletsprojects.com/)
[![Selenium](https://img.shields.io/badge/selenium-4.0%2B-orange)](https://www.selenium.dev/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Main Dashboard](https://github.com/user-attachments/assets/bdaa9514-0d21-437a-8f56-cdee202d5c80)

A powerful web scraping and machine learning platform that collects car listings from [arabam.com](https://www.arabam.com), processes them into structured datasets, and predicts vehicle prices using an XGBoost model — all through a responsive web interface.

---

## ✨ Features

### 🚙 Web Scraping

* **Rich Data Extraction**: Captures over 25 attributes per car listing.
* **Headless & Resilient**: Operates in headless mode with randomized user agents to avoid detection.
* **Robust Error Handling**: Implements retry logic and fallbacks for failed scrapes.
* **Duplicate Prevention**: Skips previously collected entries using unique hash matching.
* **Targeted Scraping**: Supports single-listing scraping for specific URLs.

### 📈 Price Prediction

* **Machine Learning Model**: Trained XGBoost model delivers accurate price predictions.
* **Advanced Feature Engineering**: Considers mileage, brand, condition, model year, and more.
* **Price Tiering**: Classifies listings into budget, mid-range, and premium tiers.
* **Brand & Condition Adjustment**: Fine-tunes prices based on specific makes or conditions.

![Prediction Interface](https://github.com/user-attachments/assets/96c0fcf9-fe29-4194-917d-64e0eafc1ba3)
![Data Table Screenshot](https://github.com/user-attachments/assets/b869d4db-e1ba-4c12-b4e9-971186858c9f)

### 🖥 Web Interface

* **Dashboard**: Monitor scraping tasks, memory usage, and uptime.
* **Data Viewer**: Search, filter, and sort collected listings.
* **Live Predictions**: Enter vehicle details and get instant price predictions.
* **Responsive Design**: Fully usable on both mobile and desktop browsers.

---

## 🛠 Technologies Used

| Layer               | Tools & Libraries                                   |
| ------------------- | --------------------------------------------------- |
| **Backend**         | Python 3.8+, Flask, Selenium, BeautifulSoup         |
| **Data Processing** | Pandas, NumPy, scikit-learn, joblib                 |
| **ML Model**        | XGBoost (regression), custom preprocessing pipeline |
| **Frontend**        | HTML5, Bootstrap 5,              |
| **Automation**      | ChromeDriver / GeckoDriver (Firefox), Headless mode |
| **Deployment**      | Docker, *(optional for production)* |

---

## 📋 System Requirements

### 💻 Software

* **Python**: 3.8+
* **Browser**: Chrome or Firefox (latest)
* **WebDriver**: `chromedriver` or `geckodriver` (must match browser version)
* **Optional**: Docker, Git, virtual environment (venv/conda)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/car-ads-scraper.git
cd car-ads-scraper
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Browser & WebDriver

* [Download Chrome](https://www.google.com/chrome/) or [Firefox](https://www.mozilla.org/firefox/)
* [Download matching WebDriver](https://chromedriver.chromium.org/downloads) or [Geckodriver](https://github.com/mozilla/geckodriver/releases)
* Add the driver to your system PATH or specify its path in the code.

### 5. Run the Flask Application

```bash
python app.py
```

### 6. Open the Web Interface

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

---

## 🧩 Troubleshooting

### Common Issues

* **WebDriver not found**:
  Ensure `chromedriver` or `geckodriver` is installed and correctly added to your system's `PATH`.

* **Blank data page**:
  Make sure scraping has completed and `scraped_ads.csv` exists with data.

* **Scraping too slow**:
  Headless browsers introduce delays to avoid bans. You can reduce delays in the code cautiously.

---

## 🤝 Contribution

Contributions and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your fork
5. Create a pull request

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

