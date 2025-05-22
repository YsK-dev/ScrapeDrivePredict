from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import pandas as pd
import os
import numpy as np
import joblib
import logging
import traceback
from webdriver_manager.chrome import ChromeDriverManager
import platform
from datetime import datetime


# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
csv_file_path = os.path.join('scraped_data', "/Users/ysk/Downloads/software-develeopment-all-1--main 2/software develeopment all/car/cleaned_dataset_for_ml.csv")

# Define paths for model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

columns = [
    "ad_Id",
    "ad_date",
    "ad_loc1",
    "ad_loc2",
    "brand",
    "series",
    "model",
    "year",
    "mileage",
    "transmission",
    "fuel_type",
    "body_type",
    "color",
    "engine_capacity",
    "engine_power",
    "drive_type",
    "vehicle_condition",
    "fuel_consumption",
    "fuel_tank",
    "paint/replacement",
    "trade_in",
    "seller_type",
    "seller_name",
    "ad_price",
    "ad_url"
]

key_to_column = {
    "İlan No": "ad_Id",
    "İlan Tarihi": "ad_date",
    "Marka": "brand",
    "Seri": "series",
    "Model": "model",
    "Yıl": "year",
    "Kilometre": "mileage",
    "Vites Tipi": "transmission",
    "Yakıt Tipi": "fuel_type",
    "Kasa Tipi": "body_type",
    "Renk": "color",
    "Motor Hacmi": "engine_capacity",
    "Motor Gücü": "engine_power",
    "Çekiş": "drive_type",
    "Araç Durumu": "vehicle_condition",
    "Ortalama Yakıt Tüketimi": "fuel_consumption",
    "Yakıt Deposu": "fuel_tank",
    "Boya-değişen": "paint/replacement",
    "Takasa Uygun": "trade_in",
    "Kimden": "seller_type"
}

def wait_for_element_or_refresh(driver, timeout, locator):
    for attempt in range(3):
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            return True
        except TimeoutException:
            print(f"Attempt {attempt + 1}: Timeout waiting for element. Refreshing page.")
            driver.refresh()
    print("Element not found after 3 attempts. Skipping.")
    return False

def get_geckodriver_path():
    system = platform.system()
    if system == "Linux":
        # Try common Linux paths
        possible_paths = [
            "/usr/bin/geckodriver",
            "/usr/local/bin/geckodriver",
            os.path.expanduser("~/bin/geckodriver"),
            os.path.expanduser("~/.local/bin/geckodriver")
        ]
    elif system == "Darwin":  # macOS
        possible_paths = [
            "/Users/ysk/Downloads/chrome-mac-arm64 ",
            os.path.expanduser("~/bin/geckodriver"),
            os.path.expanduser("~/.local/bin/geckodriver")
        ]
    else:  # Windows
        possible_paths = [
            "C:\\geckodriver.exe",
            os.path.expanduser("~\\geckodriver.exe")
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise WebDriverException("Geckodriver not found. Please install geckodriver and make sure it's in your PATH or specify the path manually.")

def get_chromedriver_path():
    system = platform.system()
    if system == "Darwin":  # macOS
        possible_paths = [
            "/Users/ysk/Downloads/chrome-mac-arm64 ",
            "/opt/homebrew/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/usr/bin/chromedriver",
            os.path.expanduser("~/bin/chromedriver"),
            os.path.expanduser("~/.local/bin/chromedriver")
        ]
    elif system == "Linux":
        possible_paths = [
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
            "/usr/lib/chromium/chromedriver",
            os.path.expanduser("~/bin/chromedriver"),
            os.path.expanduser("~/.local/bin/chromedriver")
        ]
    else:  # Windows
        possible_paths = [
            "C:\\chromedriver.exe",
            "C:\\Program Files\\chromedriver.exe",
            "C:\\Program Files (x86)\\chromedriver.exe",
            os.path.expanduser("~\\chromedriver.exe")
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise WebDriverException("ChromeDriver not found. Please install ChromeDriver and make sure it's in your PATH.")
    
# Load model components
def load_model_components():
    """Load the trained model and its components."""
    try:
        logger.info("Loading model components...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        return model, scaler, feature_names
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        traceback.print_exc()
        return None, None, None

# Load the trained model and preprocessing components
model, scaler, feature_names = load_model_components()

def preprocess_data_for_prediction(data):
    """Preprocess the input data for prediction."""
    logger.info("Starting data preprocessing for prediction...")
    
    try:
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Replace 'Unspecified' with np.nan and convert dtypes safely
        # Fix for FutureWarning about downcasting in replace
        data = data.replace('Unspecified', np.nan).infer_objects(copy=False)
        
        # Convert expected numeric columns
        numeric_columns = ['mileage', 'year', 'fuel_tank', 'engine_capacity', 'engine_power']
        for col in numeric_columns:
            if col in data.columns:
                # Extract numeric values and convert to float
                try:
                    data[col] = pd.to_numeric(data[col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
                    logger.info(f"Converted {col} to numeric: {data[col].iloc[0]}")
                except Exception as e:
                    logger.warning(f"Error converting {col} to numeric: {str(e)}")
        
        # Additional feature creation
        current_year = datetime.now().year
        if 'year' in data.columns and pd.notna(data['year'].iloc[0]):
            data['car_age'] = current_year - data['year']
            logger.info(f"Added vehicle_age: {data['car_age'].iloc[0]}")
        else:
            data['car_age'] = 0
            
        # Add model_series safely
        if 'model' in data.columns and isinstance(data['model'].iloc[0], str):
            try:
                data['model_series'] = data['model'].str.extract(r'(\d+\.\d+|\d+)', expand=False)
                data['model_series'] = pd.to_numeric(data['model_series'], errors='coerce')
                logger.info(f"Extracted model_series: {data['model_series'].iloc[0]}")
            except Exception as e:
                logger.warning(f"Error extracting model_series: {str(e)}")
                data['model_series'] = 0
        else:
            data['model_series'] = 0
        
        # Specify whether it's original or not
        if 'paint/replacement' in data.columns:
            data['is_original'] = data['paint/replacement'].apply(
                lambda x: 1 if pd.notna(x) and 'orjinal' in str(x).lower() else 0
            )
            logger.info(f"Added is_original: {data['is_original'].iloc[0]}")
        else:
            data['is_original'] = 0
        
        # Drop unused columns
        cols_to_drop = ['ad_url', 'ad_Id', 'seller_name', 'ad_date', 
                        'paint/replacement', 'ad_loc1', 'ad_loc2', 'ad_price']
        data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore', inplace=True)
        
        # One-hot encode categorical variables
        categorical_cols = ['vehicle_condition', 'drive_type', 'transmission', 
                           'fuel_type', 'color', 'body_type', 'seller_type',
                           'brand', 'series', 'model']
        
        # Make sure all categorical columns exist
        for col in categorical_cols:
            if col not in data.columns:
                data[col] = 'unknown'
        
        # Create a copy of the data for encoding
        data_encoded = data.copy()
        
        # One-hot encode categorical variables
        for col in categorical_cols:
            if col in data_encoded.columns:
                # Create dummy variables
                dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=True)
                # Drop the original column
                data_encoded = data_encoded.drop(col, axis=1)
                # Add the dummy variables
                data_encoded = pd.concat([data_encoded, dummies], axis=1)
        
        # Check if feature_names is None and handle it
        if feature_names is None:
            logger.error("feature_names is None. Make sure model components are loaded correctly.")
            # Return a safe fallback - an empty DataFrame with the right structure
            return pd.DataFrame(columns=data_encoded.columns)
        
        # Ensure we add any missing features
        missing_features = [feature for feature in feature_names if feature not in data_encoded.columns]
        if missing_features:
            logger.info(f"Adding {len(missing_features)} missing features with zero values")
            zeros_df = pd.DataFrame(0, index=data_encoded.index, columns=missing_features)
            data_encoded = pd.concat([data_encoded, zeros_df], axis=1)
        
        # Ensure we return the correct columns in the same order as training
        logger.info(f"Processed data shape: {data_encoded.shape}")
        logger.info(f"Final processed data columns: {data_encoded.columns.tolist()}")
        
        # Make sure all feature_names columns exist in data_encoded
        result_df = pd.DataFrame(index=data_encoded.index)
        for feature in feature_names:
            if feature in data_encoded.columns:
                result_df[feature] = data_encoded[feature]
            else:
                result_df[feature] = 0
                
        return result_df.fillna(0)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        # Return None to indicate failure
        return None

def predict_car_price(car_data):
    """Predict car price using the trained model."""
    try:
        logger.info("Starting price prediction...")
        logger.info(f"Input car data: {car_data}")
        
        # Preprocess the input data
        processed_data = preprocess_data_for_prediction(car_data)
        
        # Check if preprocessing was successful
        if processed_data is None or processed_data.empty:
            logger.error("Data preprocessing failed or returned empty DataFrame")
            return None
            
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Processed data columns: {processed_data.columns.tolist()}")
        
        # Load model if not already loaded
        if model is None or scaler is None or feature_names is None:
            logger.info("Model components not loaded, loading now...")
            global model, scaler, feature_names
            model, scaler, feature_names = load_model_components()
            
            if model is None or scaler is None or feature_names is None:
                logger.error("Failed to load model components")
                return None
        
        # Prediction
        try:
            logger.info("Making prediction...")
            # Scale the features
            scaled_features = scaler.transform(processed_data)
            predicted_price = model.predict(scaled_features)[0]
            logger.info(f"Predicted Price: {predicted_price}")
            return predicted_price
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Error in predict_car_price: {str(e)}")
        logger.error(traceback.format_exc())
        return None


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get car data from request
        car_data = request.json

        # Validate incoming data
        if not car_data:
            return jsonify({"error": "Car data is required."}), 400

        # Predict price for the car
        predicted_price = predict_car_price(car_data)

        if predicted_price is None:
            return jsonify({"error": "Prediction failed."}), 500

        return jsonify({"predicted_price": predicted_price}), 200
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/view-data', methods=['GET'])
def view_data():
    try:
        if os.path.exists(csv_file_path):
            logger.info(f"Reading data from {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logger.info(f"Found {len(df)} records")

            # Clean and format the data
            df.replace('Unspecified', None, inplace=True)

            # Format prices
            df['ad_price'] = pd.to_numeric(df['ad_price'].str.replace(' TL', '').str.replace('.', ''), errors='coerce')
            df['ad_price'] = df['ad_price'].apply(lambda x: f"{x:,.0f} TL" if pd.notnull(x) else "N/A")

            # Format mileage
            df['mileage'] = pd.to_numeric(df['mileage'].str.replace(' km', '').str.replace('.', ''), errors='coerce')
            df['mileage'] = df['mileage'].apply(lambda x: f"{x:,.0f} km" if pd.notnull(x) else "N/A")

            cars = df.to_dict('records')
            return render_template('view_data.html', data=cars)
        else:
            logger.warning(f"Data file not found at {csv_file_path}")
            return render_template('view_data.html', data=None)
    except Exception as e:
        logger.error(f"Error in view_data: {e}")
        return render_template('view_data.html', data=None, error=str(e))

# Scrape car data from the provided URL
def scrape_single_car(url):
    try:
        print(f"Starting to scrape car data from: {url}")
        
        
        
        
        # Configure Chrome options
        options = Options()
        
        # Create a unique user data directory
        import tempfile
        import uuid
        import os
        import shutil
        
        # Use UUID to ensure uniqueness
        temp_dir = os.path.join(tempfile.gettempdir(), f"chrome_temp_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Using temp directory: {temp_dir}")
        
        # Set the user data directory
        options.add_argument(f'--user-data-dir={temp_dir}')
        
        # Headless mode settings
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Performance optimizations
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-notifications')
        
        # Anti-detection measures
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Timeout settings - important to handle renderer timeouts
        options.add_argument('--disable-hang-monitor')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-prompt-on-repost')
        
        # Memory and performance optimizations
        options.add_argument('--disable-features=TranslateUI,BlinkGenPropertyTrees')
        options.add_argument('--disable-ipc-flooding-protection')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--enable-features=NetworkServiceInProcess')
        
        # Window size and user agent
        options.add_argument('--window-size=1280,720')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36')
        
        # Set browser path based on platform
        if platform.system() == "Darwin":  # macOS
            options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        elif platform.system() == "Linux":
            options.binary_location = "/usr/bin/google-chrome-stable"
        else:  # Windows
            options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        
        # Get ChromeDriver path
        chromedriver_path = get_chromedriver_path()
        print(f"Using ChromeDriver at: {chromedriver_path}")
        
        # Initialize WebDriver with retry mechanism and better error handling
        max_retries = 3
        retry_count = 0
        last_error = None
        driver = None
        
        while retry_count < max_retries:
            try:
                service = Service(executable_path=chromedriver_path)
                driver = webdriver.Chrome(service=service, options=options)
                print("WebDriver initialized successfully")
                break
            except Exception as e:
                last_error = e
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")
                
                # Clean up after failure
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                    driver = None
                
                # Remove the temp directory and try again with a new one
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    temp_dir = os.path.join(tempfile.gettempdir(), f"chrome_temp_{uuid.uuid4().hex}")
                    os.makedirs(temp_dir, exist_ok=True)
                    options.add_argument(f'--user-data-dir={temp_dir}')
                    print(f"Created new temp directory: {temp_dir}")
                except:
                    pass
                
                # Additional cleanup between attempts
                

                
                if retry_count < max_retries:
                    time.sleep(3)  # Wait longer between attempts
                    continue
                else:
                    raise Exception(f"Failed to initialize WebDriver after {max_retries} attempts. Last error: {str(last_error)}")
        
        # Continue only if driver initialized successfully
        if not driver:
            raise Exception("Driver initialization failed")
            
        try:
            # Set page load timeout (increased for better reliability)
            driver.set_page_load_timeout(90)
            driver.set_script_timeout(90)
            
            # Navigate to URL with retry mechanism and timeout handling
            page_load_retries = 3
            for attempt in range(page_load_retries):
                try:
                    print(f"Loading URL (attempt {attempt+1}): {url}")
                    driver.get(url)
                    print("Page loaded successfully")
                    break
                except Exception as e:
                    if attempt == page_load_retries - 1:
                        raise Exception(f"Failed to load page after {page_load_retries} attempts: {str(e)}")
                    print(f"Page load attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(5)  # Increased wait time between page load attempts
            
            # Wait for page to fully load and stabilize
            time.sleep(5)
            
            # Initialize car data
            car_data = {column: "Unspecified" for column in columns}
            
            # Get basic information with more robust waiting
            try:
                wait = WebDriverWait(driver, 10)
                price_element = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='desktop-information-price']"))
                )
                
                # Extract price and convert to numeric
                price_text = price_element.text
                price = float(price_text.replace(' TL', '').replace('.', '').replace(',', '.'))
                
                # Get other car details with explicit waits for each element
                car_data["brand"] = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#wrapper > div:nth-child(5) > div.container > div > div.product-detail > div.product-detail-wrapper > div.product-properties-container > div.product-properties > div.product-properties-details.linear-gradient > div:nth-child(3) > div.property-value"))
                ).text
                
                car_data["model"] = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#wrapper > div:nth-child(5) > div.container > div > div.product-detail > div.product-detail-wrapper > div.product-properties-container > div.product-properties > div.product-properties-details.linear-gradient > div:nth-child(5) > div.property-value"))
                ).text
                
                car_data["year"] = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#wrapper > div:nth-child(5) > div.container > div > div.product-detail > div.product-detail-wrapper > div.product-properties-container > div.product-properties > div.product-properties-details.linear-gradient > div:nth-child(6) > div.property-value"))
                ).text
                
                car_data["mileage"] = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#wrapper > div:nth-child(5) > div.container > div > div.product-detail > div.product-detail-wrapper > div.product-properties-container > div.product-properties > div.product-properties-details.linear-gradient > div:nth-child(7) > div.property-value"))
                ).text
                
            except Exception as e:
                print(f"Error getting basic car information: {str(e)}")
                # Take a screenshot for debugging
                try:
                    screenshot_path = os.path.join(tempfile.gettempdir(), f"scraper_debug_{uuid.uuid4().hex}.png")
                    driver.save_screenshot(screenshot_path)
                    print(f"Debug screenshot saved to {screenshot_path}")
                except:
                    pass
                raise
            
            # Get additional details with error handling for each property
            try:
                details = driver.find_elements(By.CSS_SELECTOR, ".property-item")
                for detail in details:
                    try:
                        key = detail.find_element(By.CSS_SELECTOR, ".property-key").text
                        value = detail.find_element(By.CSS_SELECTOR, ".property-value").text
                        if key in key_to_column:
                            car_data[key_to_column[key]] = value
                    except Exception as detail_error:
                        print(f"Skipping property due to error: {str(detail_error)}")
                        continue
            except Exception as e:
                print(f"Warning: Error getting additional details: {str(e)}")
                # Continue with partial data
            
            return {
                "success": True,
                "price": price,
                "data": car_data
            }
            
        except Exception as e:
            print(f"Error scraping car data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Always ensure browser is closed and temp directories cleaned up
            if driver:
                try:
                    driver.quit()
                except Exception as quit_error:
                    print(f"Error during driver quit: {str(quit_error)}")
                driver = None
            
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Removed temp directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Error during temp directory cleanup: {str(cleanup_error)}")
            
            # Final process cleanup
  
            
    except Exception as e:
        print(f"Error in scrape_single_car: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-scraper', methods=['POST'])
def start_scraper():
    try:
        scrape_ads()  # Your existing scrape_ads implementation
        return jsonify({"message": "Scraping completed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/estimate-price', methods=['POST'])
def estimate_price():
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({"success": False, "error": "Car URL is required"}), 400

        logger.info(f"Received price estimation request for URL: {url}")
        scrape_result = scrape_single_car(url)
        
        if not scrape_result or not scrape_result.get("success"):
            error_msg = scrape_result.get('error', 'Failed to scrape car data') if scrape_result else 'Scraping failed'
            logger.error(f"Scraping error: {error_msg}")
            return jsonify({"success": False, "error": error_msg}), 500
            
        car_data = scrape_result["data"]
        actual_price = scrape_result.get("price")
        
        if not actual_price:
            logger.error("No price found in scraped data")
            return jsonify({"success": False, "error": "Could not find car price"}), 500

        logger.info(f"Successfully scraped car data. Actual price: {actual_price:,.0f} TL")

        # Predict price based on processed car_data
        predicted_price = predict_car_price(car_data)
        if predicted_price is None:
            logger.error("Price prediction failed")
            return jsonify({"success": False, "error": "Prediction failed."}), 500  

        # Generate comparison DataFrame
        comparison_df = pd.DataFrame({
            "Actual Price": [actual_price],
            "Predicted Price": [predicted_price],
            "Difference": [actual_price - predicted_price],
            "Error %": [abs((actual_price - predicted_price) / actual_price) * 100]
        })

        logger.info(f"Comparison:\n{comparison_df}")

        price_difference = actual_price - predicted_price
        price_difference_percentage = (price_difference / actual_price * 100) if actual_price else 0
        
        formatted_actual = f"{actual_price:,.0f} TL"
        formatted_predicted = f"{predicted_price:,.0f} TL"

        price_assessment = "OVERPRICED" if price_difference > 0 else "UNDERPRICED"
        if abs(price_difference_percentage) < 5:
            price_assessment = "FAIRLY PRICED"

        return jsonify({
            "success": True,
            "actual_price": formatted_actual,
            "predicted_price": formatted_predicted,
            "comparison": comparison_df.to_dict('records'),
            "price_difference": f"{price_difference:,.0f} TL",
            "price_difference_percentage": f"{price_difference_percentage:.1f}%",
            "price_assessment": price_assessment,
            "car_data": car_data
        })

    except Exception as e:
        logger.error(f"Error in estimate_price route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=False)
