import os
import base64
import json
import re
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import pytesseract
import cv2
import numpy as np

# Load environment variables from .env file
# Try multiple paths for .env file
import pathlib
current_dir = pathlib.Path(__file__).parent.absolute()
env_path = current_dir / '.env'
print(f"Looking for .env file at: {env_path}")
print(f"File exists: {env_path.exists()}")

if env_path.exists():
    load_dotenv(env_path)
    print("✓ Loaded .env file")
else:
    load_dotenv()  # Try default location
    print("Using default .env location")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
print(f"GEMINI_API_KEY found: {'Yes' if GEMINI_API_KEY else 'No'}")
gemini_available = False
model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models first
        print("=" * 50)
        print("Checking available Gemini models...")
        available_model_names = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_model_names.append(m.name)
                    print(f"  ✓ Available: {m.name}")
        except Exception as list_error:
            print(f"  Could not list models: {list_error}")
        
        # Use the first available model that supports vision/images
        # Priority: models with vision capability
        vision_models = [name for name in available_model_names if 'vision' in name.lower() or 'flash' in name.lower() or 'pro' in name.lower()]
        
        if vision_models:
            # Use the first vision-capable model
            model_name = vision_models[0]
            print(f"  Using model: {model_name}")
            model = genai.GenerativeModel(model_name)
            gemini_available = True
            print(f"✓ Gemini API configured successfully with model: {model_name}")
        elif available_model_names:
            # Use any available model
            model_name = available_model_names[0]
            print(f"  Using model: {model_name}")
            model = genai.GenerativeModel(model_name)
            gemini_available = True
            print(f"✓ Gemini API configured successfully with model: {model_name}")
        else:
            # Try common model names as fallback
            fallback_models = [
                'gemini-pro-vision',
                'gemini-pro',
                'gemini-1.0-pro-vision-latest',
                'gemini-1.0-pro',
            ]
            for model_name in fallback_models:
                try:
                    print(f"  Trying fallback model: {model_name}...")
                    model = genai.GenerativeModel(model_name)
                    gemini_available = True
                    print(f"✓ Gemini API configured with fallback model: {model_name}")
                    break
                except Exception as e:
                    print(f"    Failed: {str(e)[:80]}")
                    continue
        
        if not gemini_available:
            print("⚠ No Gemini models available. Using Tesseract OCR only.")
        print("=" * 50)
    except Exception as e:
        print(f"⚠ Gemini API configuration failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ GEMINI_API_KEY not found in .env file. Using Tesseract OCR only.")

# Health scoring criteria
HEALTH_CRITERIA = {
    'calories_per_serving': {
        'low': 150,
        'moderate': 250,
        'high': 400
    },
    'sugar_per_serving': {
        'low': 5,
        'moderate': 10,
        'high': 20
    },
    'saturated_fat_per_serving': {
        'low': 2,
        'moderate': 4,
        'high': 8
    },
    'sodium_per_serving': {
        'low': 140,
        'moderate': 400,
        'high': 800
    },
    'fiber_per_serving': {
        'low': 2,
        'moderate': 4,
        'high': 6
    },
    'protein_per_serving': {
        'low': 5,
        'moderate': 10,
        'high': 20
    }
}

def preprocess_image_for_ocr(image_path):
    """Preprocess image for better OCR results"""
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text detection
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Resize if image is too small
        height, width = denoised.shape
        if width < 800:
            scale = 800 / width
            denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return denoised
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_text_with_tesseract(image_path):
    """Extract text from image using Tesseract OCR"""
    try:
        # Try multiple OCR configurations for best results
        configs = [
            '--psm 6',  # Assume uniform block of text
            '--psm 4',  # Assume single column of text
            '--psm 3',  # Fully automatic page segmentation
            '--psm 11', # Sparse text
        ]
        
        best_text = ""
        
        # First try with preprocessed image
        processed_img = preprocess_image_for_ocr(image_path)
        
        for config in configs:
            try:
                if processed_img is not None:
                    text = pytesseract.image_to_string(processed_img, config=config)
                else:
                    img = Image.open(image_path)
                    text = pytesseract.image_to_string(img, config=config)
                
                # Keep the result with most content
                if len(text) > len(best_text):
                    best_text = text
            except Exception as config_error:
                print(f"OCR config {config} failed: {config_error}")
                continue
        
        # Also try with original image if preprocessed didn't work well
        if len(best_text) < 50:
            try:
                img = Image.open(image_path)
                original_text = pytesseract.image_to_string(img, config='--psm 6')
                if len(original_text) > len(best_text):
                    best_text = original_text
            except:
                pass
        
        print(f"=== OCR Extracted Text ({len(best_text)} chars) ===")
        print(best_text[:500] if len(best_text) > 500 else best_text)
        print("=== End OCR Text ===")
        
        return best_text.strip()
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def parse_nutrition_from_text(raw_text):
    """Parse nutrition data from OCR extracted text"""
    nutrition_data = {
        "nutrition_facts": {},
        "ingredients": "",
        "health_claims": [],
        "allergens": [],
        "analysis_notes": "Extracted using Tesseract OCR",
        "raw_text": raw_text
    }
    
    # Clean up OCR text - handle common OCR errors
    text_clean = raw_text.replace('|', 'l').replace('0g', '0 g').replace('0mg', '0 mg')
    text_lower = text_clean.lower()
    
    # Also try to find numbers that might be separated by spaces or special chars
    # Replace common OCR mistakes
    text_lower = re.sub(r'(\d)\s*[,\.]\s*(\d)', r'\1.\2', text_lower)  # Fix decimal points
    
    print(f"=== Parsing nutrition from text ===")
    
    # More flexible patterns for extracting nutrition values
    # These patterns handle various formats and OCR errors
    patterns = {
        'serving_size': [
            r'serving\s*size[:\s]*([^\n]+)',
            r'portion\s*size[:\s]*([^\n]+)',
            r'portion[:\s]*([^\n]+)',
            r'per\s*serving[:\s]*([^\n]+)',
        ],
        'servings_per_container': [
            r'servings?\s*per\s*container[:\s]*(\d+)',
            r'about\s*(\d+)\s*servings?',
            r'(\d+)\s*servings?\s*per',
        ],
        'calories': [
            r'calories[:\s]*(\d+)',
            r'calories\s+(\d+)',
            r'energy[:\s]*(\d+)\s*(?:kcal|cal)?',
            r'(\d+)\s*(?:kcal|calories)',
            r'cal(?:ories)?[:\s\.]*(\d+)',
        ],
        'total_fat': [
            r'total\s*fat[:\s]*(\d+\.?\d*)\s*g?',
            r'fat[:\s]*(\d+\.?\d*)\s*g',
            r'total\s*fat\s+(\d+\.?\d*)',
            r'fat\s+(\d+\.?\d*)\s*g',
        ],
        'saturated_fat': [
            r'saturated\s*fat[:\s]*(\d+\.?\d*)\s*g?',
            r'sat\.?\s*fat[:\s]*(\d+\.?\d*)\s*g?',
            r'saturated[:\s]*(\d+\.?\d*)\s*g',
        ],
        'trans_fat': [
            r'trans\s*fat[:\s]*(\d+\.?\d*)\s*g?',
            r'trans[:\s]*(\d+\.?\d*)\s*g',
        ],
        'cholesterol': [
            r'cholesterol[:\s]*(\d+)\s*m?g?',
            r'cholest\.?[:\s]*(\d+)',
        ],
        'sodium': [
            r'sodium[:\s]*(\d+)\s*m?g?',
            r'salt[:\s]*(\d+\.?\d*)\s*(?:mg|g)?',
            r'sodium\s+(\d+)',
        ],
        'total_carbohydrate': [
            r'total\s*carb(?:ohydrate)?s?[:\s]*(\d+\.?\d*)\s*g?',
            r'carb(?:ohydrate)?s?[:\s]*(\d+\.?\d*)\s*g?',
            r'carbs?[:\s]*(\d+\.?\d*)',
        ],
        'dietary_fiber': [
            r'dietary\s*fib(?:er|re)[:\s]*(\d+\.?\d*)\s*g?',
            r'fib(?:er|re)[:\s]*(\d+\.?\d*)\s*g?',
            r'fiber\s+(\d+\.?\d*)',
        ],
        'total_sugars': [
            r'total\s*sugars?[:\s]*(\d+\.?\d*)\s*g?',
            r'sugars?[:\s]*(\d+\.?\d*)\s*g?',
            r'sugar\s+(\d+\.?\d*)',
        ],
        'added_sugars': [
            r'added\s*sugars?[:\s]*(\d+\.?\d*)\s*g?',
            r'includes?\s*(\d+\.?\d*)\s*g?\s*added\s*sugars?',
        ],
        'protein': [
            r'protein[:\s]*(\d+\.?\d*)\s*g?',
            r'protein\s+(\d+\.?\d*)',
        ],
        'vitamin_d': [
            r'vitamin\s*d[:\s]*(\d+)\s*%?',
            r'vit\.?\s*d[:\s]*(\d+)',
        ],
        'calcium': [
            r'calcium[:\s]*(\d+)\s*%?',
        ],
        'iron': [
            r'iron[:\s]*(\d+)\s*%',
        ],
        'potassium': [
            r'potassium[:\s]*(\d+)\s*(?:mg|%)',
        ],
    }
    
    # Extract nutrition values
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = match.group(1).strip()
                    if key in ['serving_size']:
                        nutrition_data["nutrition_facts"][key] = value
                        print(f"  Found {key}: {value}")
                    else:
                        # Convert to number
                        numeric_match = re.search(r'[\d.]+', value)
                        if numeric_match:
                            numeric_value = float(numeric_match.group())
                            nutrition_data["nutrition_facts"][key] = numeric_value
                            print(f"  Found {key}: {numeric_value}")
                    break
                except (ValueError, AttributeError) as e:
                    print(f"  Error parsing {key}: {e}")
                    continue
    
    # Print summary of extracted values
    print(f"=== Extracted Nutrition Facts ===")
    for k, v in nutrition_data["nutrition_facts"].items():
        print(f"  {k}: {v}")
    print(f"=================================")
    
    # Extract ingredients
    ingredients_patterns = [
        r'ingredients[:\s]*([^\n]+(?:\n(?![A-Z][a-z]+:)[^\n]+)*)',
        r'contains[:\s]*([^\n]+)',
    ]
    
    for pattern in ingredients_patterns:
        match = re.search(pattern, text_lower)
        if match:
            nutrition_data["ingredients"] = match.group(1).strip()
            break
    
    # Detect allergens
    common_allergens = ['milk', 'eggs', 'fish', 'shellfish', 'tree nuts', 'peanuts', 
                        'wheat', 'soybeans', 'soy', 'gluten', 'sesame']
    
    for allergen in common_allergens:
        if allergen in text_lower:
            nutrition_data["allergens"].append(allergen.title())
    
    # Detect health claims
    health_claim_keywords = [
        'organic', 'low fat', 'low-fat', 'fat free', 'sugar free', 'no sugar added',
        'high fiber', 'whole grain', 'natural', 'non-gmo', 'gluten free', 'vegan',
        'vegetarian', 'keto', 'low carb', 'high protein', 'zero trans fat'
    ]
    
    for claim in health_claim_keywords:
        if claim in text_lower:
            nutrition_data["health_claims"].append(claim.title())
    
    return nutrition_data

def analyze_nutrition_with_gemini(image_path):
    """Use Gemini Vision API to analyze nutrition label"""
    global gemini_available, model
    
    if not gemini_available or model is None:
        print("Gemini not available, using Tesseract OCR")
        return analyze_nutrition_with_tesseract(image_path)
    
    try:
        print(f"=== Analyzing image with Gemini AI ===")
        print(f"Image path: {image_path}")
        
        # Open and prepare image
        img = Image.open(image_path)
        print(f"Image size: {img.size}, mode: {img.mode}")
        
        # Prepare prompt
        prompt = """Analyze this nutrition facts label and extract ALL information in JSON format.
        Look for these specific fields:
        1. serving_size (with units)
        2. servings_per_container
        3. calories
        4. total_fat (in grams)
        5. saturated_fat (in grams)
        6. trans_fat (in grams)
        7. cholesterol (in mg)
        8. sodium (in mg)
        9. total_carbohydrate (in grams)
        10. dietary_fiber (in grams)
        11. total_sugars (in grams)
        12. added_sugars (in grams)
        13. protein (in grams)
        14. vitamin_d (in %DV)
        15. calcium (in %DV)
        16. iron (in %DV)
        17. potassium (in %DV)
        
        Also analyze:
        - Ingredients list (if visible)
        - Any health claims (organic, low-fat, etc.)
        - Allergen information
        
        Return ONLY a valid JSON object with this structure:
        {
            "nutrition_facts": {
                "serving_size": "value",
                "servings_per_container": value,
                "calories": value,
                "total_fat": value,
                "saturated_fat": value,
                "trans_fat": value,
                "cholesterol": value,
                "sodium": value,
                "total_carbohydrate": value,
                "dietary_fiber": value,
                "total_sugars": value,
                "added_sugars": value,
                "protein": value,
                "vitamin_d": value,
                "calcium": value,
                "iron": value,
                "potassium": value
            },
            "ingredients": "list of ingredients",
            "health_claims": ["list", "of", "claims"],
            "allergens": ["list", "of", "allergens"],
            "analysis_notes": "brief analysis notes",
            "raw_text": "the full raw text extracted from the nutrition label"
        }
        
        If any field is not visible, use null or 0 as appropriate."""
        
        # Use Gemini to analyze the image
        print("Sending image to Gemini API...")
        response = model.generate_content([prompt, img])
        
        # Extract JSON from response
        response_text = response.text
        print(f"=== Gemini Response ({len(response_text)} chars) ===")
        print(response_text[:1000] if len(response_text) > 1000 else response_text)
        print("=== End Gemini Response ===")
        
        # Find JSON in the response (sometimes Gemini adds extra text)
        try:
            # Look for JSON pattern
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print("No JSON found in response, using Tesseract fallback")
                return analyze_nutrition_with_tesseract(image_path)
            
            json_str = response_text[json_start:json_end]
            
            # Parse JSON
            data = json.loads(json_str)
            data['analysis_notes'] = "Analyzed using Gemini Vision AI"
            
            # Print extracted values
            print("=== Gemini Extracted Nutrition Facts ===")
            for k, v in data.get('nutrition_facts', {}).items():
                print(f"  {k}: {v}")
            print("=========================================")
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"JSON string attempted: {json_str[:500] if json_str else 'None'}")
            # Fallback to Tesseract if JSON parsing fails
            return analyze_nutrition_with_tesseract(image_path)
            
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to Tesseract OCR
        return analyze_nutrition_with_tesseract(image_path)

def analyze_nutrition_with_tesseract(image_path):
    """Fallback: Use Tesseract OCR to analyze nutrition label"""
    print("Using Tesseract OCR for text extraction...")
    
    # Extract raw text
    raw_text = extract_text_with_tesseract(image_path)
    
    if not raw_text:
        return {
            "nutrition_facts": {},
            "ingredients": "",
            "health_claims": [],
            "allergens": [],
            "analysis_notes": "Failed to extract text from image. Please ensure the image is clear and contains a nutrition label.",
            "raw_text": ""
        }
    
    # Parse nutrition data from text
    nutrition_data = parse_nutrition_from_text(raw_text)
    
    return nutrition_data

def calculate_onqi_score(nutrition_facts):
    """
    Calculate Overall Nutritional Quality Index (ONQI) using extracted values.
    
    ONQI Formula (simplified version based on NuVal system):
    - Nutrients to ENCOURAGE: Fiber, Protein, Vitamins, Minerals
    - Nutrients to LIMIT: Saturated Fat, Trans Fat, Sodium, Sugar, Cholesterol
    
    Score Range: 1-100 (higher is better)
    """
    
    print("\n" + "="*50)
    print("ONQI CALCULATION")
    print("="*50)
    
    # Get values with defaults - handle None and empty values
    def safe_float(value, default=0):
        try:
            if value is None or value == '' or value == 'null':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    saturated_fat = safe_float(nutrition_facts.get('saturated_fat'))
    trans_fat = safe_float(nutrition_facts.get('trans_fat'))
    sodium = safe_float(nutrition_facts.get('sodium'))
    total_sugars = safe_float(nutrition_facts.get('total_sugars'))
    sugars = safe_float(nutrition_facts.get('sugars', total_sugars))
    dietary_fiber = safe_float(nutrition_facts.get('dietary_fiber'))
    protein = safe_float(nutrition_facts.get('protein'))
    calories = safe_float(nutrition_facts.get('calories'))
    total_fat = safe_float(nutrition_facts.get('total_fat'))
    cholesterol = safe_float(nutrition_facts.get('cholesterol'))
    total_carbohydrate = safe_float(nutrition_facts.get('total_carbohydrate'))
    
    # Use the higher of sugars or total_sugars
    sugar_value = max(sugars, total_sugars)
    
    print(f"Input Values:")
    print(f"  Calories: {calories} kcal")
    print(f"  Total Fat: {total_fat}g")
    print(f"  Saturated Fat: {saturated_fat}g")
    print(f"  Trans Fat: {trans_fat}g")
    print(f"  Cholesterol: {cholesterol}mg")
    print(f"  Sodium: {sodium}mg")
    print(f"  Total Carbs: {total_carbohydrate}g")
    print(f"  Dietary Fiber: {dietary_fiber}g")
    print(f"  Sugars: {sugar_value}g")
    print(f"  Protein: {protein}g")
    
    # Check if we have any meaningful data
    has_data = any([calories > 0, total_fat > 0, saturated_fat > 0, sodium > 0,
                    sugar_value > 0, protein > 0, dietary_fiber > 0])
    
    if not has_data:
        print("\n⚠ WARNING: No nutrition data extracted! Returning score of 50 (unknown)")
        return 50  # Return middle score if no data
    
    # ===== ONQI CALCULATION =====
    # Start with base score of 50 (neutral)
    score = 50
    
    print(f"\nStarting Score: {score}")
    
    # ===== NEGATIVE FACTORS (Penalties) - Max -50 points =====
    penalties = 0
    
    # 1. Saturated Fat Penalty (max -15 points)
    # Daily limit: 20g, per serving concern: >5g
    if saturated_fat > 0:
        sat_fat_penalty = min(15, (saturated_fat / 5) * 5)  # 5 points per 5g
        penalties += sat_fat_penalty
        print(f"  Saturated Fat ({saturated_fat}g): -{sat_fat_penalty:.1f} points")
    
    # 2. Trans Fat Penalty (max -15 points) - Very unhealthy
    # Any trans fat is bad
    if trans_fat > 0:
        trans_penalty = min(15, trans_fat * 10)  # 10 points per gram
        penalties += trans_penalty
        print(f"  Trans Fat ({trans_fat}g): -{trans_penalty:.1f} points")
    
    # 3. Sodium Penalty (max -10 points)
    # Daily limit: 2300mg, per serving concern: >600mg
    if sodium > 0:
        sodium_penalty = min(10, (sodium / 600) * 5)  # 5 points per 600mg
        penalties += sodium_penalty
        print(f"  Sodium ({sodium}mg): -{sodium_penalty:.1f} points")
    
    # 4. Sugar Penalty (max -10 points)
    # Daily limit: 50g, per serving concern: >12g
    if sugar_value > 0:
        sugar_penalty = min(10, (sugar_value / 12) * 5)  # 5 points per 12g
        penalties += sugar_penalty
        print(f"  Sugar ({sugar_value}g): -{sugar_penalty:.1f} points")
    
    # 5. Cholesterol Penalty (max -5 points)
    # Daily limit: 300mg, per serving concern: >100mg
    if cholesterol > 0:
        chol_penalty = min(5, (cholesterol / 100) * 2.5)  # 2.5 points per 100mg
        penalties += chol_penalty
        print(f"  Cholesterol ({cholesterol}mg): -{chol_penalty:.1f} points")
    
    # 6. High Calorie Penalty (max -5 points)
    # Per serving concern: >400 kcal
    if calories > 200:
        cal_penalty = min(5, ((calories - 200) / 200) * 2.5)
        penalties += cal_penalty
        print(f"  High Calories ({calories}kcal): -{cal_penalty:.1f} points")
    
    print(f"\nTotal Penalties: -{penalties:.1f} points")
    
    # ===== POSITIVE FACTORS (Bonuses) - Max +50 points =====
    bonuses = 0
    
    # 1. Dietary Fiber Bonus (max +15 points)
    # Daily goal: 25g, per serving good: >3g
    if dietary_fiber > 0:
        fiber_bonus = min(15, (dietary_fiber / 3) * 5)  # 5 points per 3g
        bonuses += fiber_bonus
        print(f"  Dietary Fiber ({dietary_fiber}g): +{fiber_bonus:.1f} points")
    
    # 2. Protein Bonus (max +15 points)
    # Daily goal: 50g, per serving good: >10g
    if protein > 0:
        protein_bonus = min(15, (protein / 10) * 5)  # 5 points per 10g
        bonuses += protein_bonus
        print(f"  Protein ({protein}g): +{protein_bonus:.1f} points")
    
    # 3. Low Fat Bonus (max +10 points)
    # If total fat is low (<5g) and not zero
    if 0 < total_fat <= 5:
        low_fat_bonus = 10 - (total_fat * 2)  # More bonus for lower fat
        bonuses += max(0, low_fat_bonus)
        print(f"  Low Fat ({total_fat}g): +{max(0, low_fat_bonus):.1f} points")
    
    # 4. Low Sugar Bonus (max +5 points)
    # If sugar is low (<5g) and not zero
    if 0 < sugar_value <= 5:
        low_sugar_bonus = 5 - sugar_value
        bonuses += max(0, low_sugar_bonus)
        print(f"  Low Sugar ({sugar_value}g): +{max(0, low_sugar_bonus):.1f} points")
    
    # 5. Low Sodium Bonus (max +5 points)
    # If sodium is low (<200mg) and not zero
    if 0 < sodium <= 200:
        low_sodium_bonus = 5 - (sodium / 40)
        bonuses += max(0, low_sodium_bonus)
        print(f"  Low Sodium ({sodium}mg): +{max(0, low_sodium_bonus):.1f} points")
    
    print(f"\nTotal Bonuses: +{bonuses:.1f} points")
    
    # Calculate final score
    score = 50 - penalties + bonuses
    
    # Ensure score is between 1-100
    final_score = max(1, min(100, round(score)))
    
    print(f"\n{'='*50}")
    print(f"FINAL ONQI SCORE: {final_score}/100")
    print(f"{'='*50}\n")
    
    return final_score

def get_health_analysis(nutrition_data, onqi_score):
    """Generate comprehensive health analysis based on extracted values"""
    analysis = {
        'score': onqi_score,
        'rating': '',
        'summary': '',
        'pros': [],
        'cons': [],
        'recommendations': [],
        'alternatives': []
    }
    
    nutrition_facts = nutrition_data.get('nutrition_facts', {})
    
    # Get values with safe defaults
    saturated_fat = float(nutrition_facts.get('saturated_fat', 0) or 0)
    trans_fat = float(nutrition_facts.get('trans_fat', 0) or 0)
    sodium = float(nutrition_facts.get('sodium', 0) or 0)
    total_sugars = float(nutrition_facts.get('total_sugars', 0) or 0)
    sugars = float(nutrition_facts.get('sugars', total_sugars) or 0)
    dietary_fiber = float(nutrition_facts.get('dietary_fiber', 0) or 0)
    protein = float(nutrition_facts.get('protein', 0) or 0)
    calories = float(nutrition_facts.get('calories', 0) or 0)
    cholesterol = float(nutrition_facts.get('cholesterol', 0) or 0)
    
    sugar_value = max(sugars, total_sugars)
    
    # Determine rating based on ONQI score
    if onqi_score >= 80:
        analysis['rating'] = 'Excellent 🏆'
        analysis['summary'] = 'This is a highly nutritious food choice! It has an excellent balance of nutrients.'
    elif onqi_score >= 60:
        analysis['rating'] = 'Good 👍'
        analysis['summary'] = 'This is a healthy food choice with good nutritional value.'
    elif onqi_score >= 40:
        analysis['rating'] = 'Fair ⚠️'
        analysis['summary'] = 'This food has moderate nutritional value. Consider consuming in moderation.'
    elif onqi_score >= 20:
        analysis['rating'] = 'Poor 👎'
        analysis['summary'] = 'This food has limited nutritional value and may not be the healthiest choice.'
    else:
        analysis['rating'] = 'Very Poor ❌'
        analysis['summary'] = 'This food is not nutritionally beneficial. Consider healthier alternatives.'
    
    # ===== ANALYZE POSITIVE ASPECTS (PROS) =====
    
    if dietary_fiber >= 5:
        analysis['pros'].append(f"High in fiber ({dietary_fiber}g) - excellent for digestive health")
    elif dietary_fiber >= 3:
        analysis['pros'].append(f"Good fiber content ({dietary_fiber}g) - supports healthy digestion")
    
    if protein >= 15:
        analysis['pros'].append(f"Excellent protein source ({protein}g) - great for muscle maintenance")
    elif protein >= 10:
        analysis['pros'].append(f"Good protein content ({protein}g) - helps with satiety")
    
    if saturated_fat <= 2 and saturated_fat >= 0:
        analysis['pros'].append(f"Low in saturated fat ({saturated_fat}g) - heart healthy choice")
    
    if sugar_value <= 5 and sugar_value >= 0:
        analysis['pros'].append(f"Low in sugar ({sugar_value}g) - better for blood sugar control")
    
    if sodium <= 200 and sodium > 0:
        analysis['pros'].append(f"Low sodium ({sodium}mg) - good for blood pressure")
    
    if trans_fat == 0:
        analysis['pros'].append("No trans fat - excellent for cardiovascular health")
    
    if calories <= 150 and calories > 0:
        analysis['pros'].append(f"Low calorie ({calories} kcal) - good for weight management")
    
    # ===== ANALYZE NEGATIVE ASPECTS (CONS) =====
    
    if saturated_fat > 5:
        analysis['cons'].append(f"High in saturated fat ({saturated_fat}g) - may increase heart disease risk")
    elif saturated_fat > 3:
        analysis['cons'].append(f"Moderate saturated fat ({saturated_fat}g) - consume in moderation")
    
    if trans_fat > 0:
        analysis['cons'].append(f"Contains trans fat ({trans_fat}g) - avoid for heart health")
    
    if sodium > 600:
        analysis['cons'].append(f"High in sodium ({sodium}mg) - may affect blood pressure")
    elif sodium > 400:
        analysis['cons'].append(f"Moderate sodium ({sodium}mg) - watch your daily intake")
    
    if sugar_value > 15:
        analysis['cons'].append(f"High in sugar ({sugar_value}g) - may contribute to weight gain and diabetes")
    elif sugar_value > 10:
        analysis['cons'].append(f"Moderate sugar content ({sugar_value}g) - be mindful of total daily sugar intake")
    
    if cholesterol > 60:
        analysis['cons'].append(f"Contains cholesterol ({cholesterol}mg) - monitor if you have heart concerns")
    
    if calories > 300:
        analysis['cons'].append(f"High calorie content ({calories} kcal) - consider portion size")
    
    # ===== RECOMMENDATIONS =====
    
    if onqi_score < 60:
        analysis['recommendations'].append("Consider consuming this product in moderation")
        analysis['recommendations'].append("Balance with healthier foods throughout the day")
    
    if sodium > 400:
        analysis['recommendations'].append("Drink plenty of water to help flush excess sodium")
    
    if sugar_value > 10:
        analysis['recommendations'].append("Pair with protein or fiber to slow sugar absorption")
    
    if saturated_fat > 3:
        analysis['recommendations'].append("Balance with foods rich in unsaturated fats like avocados or nuts")
    
    if dietary_fiber < 3:
        analysis['recommendations'].append("Consider adding fiber-rich foods to your meal")
    
    if protein < 5:
        analysis['recommendations'].append("Consider pairing with a protein source for better satiety")
    
    # ===== HEALTHIER ALTERNATIVES =====
    
    if onqi_score < 70:
        analysis['alternatives'].append("Fresh fruits and vegetables")
        analysis['alternatives'].append("Whole grain options")
        analysis['alternatives'].append("Lean proteins (chicken, fish, beans)")
        analysis['alternatives'].append("Nuts and seeds in moderation")
    
    if sugar_value > 15:
        analysis['alternatives'].append("Sugar-free or naturally sweetened alternatives")
    
    if sodium > 500:
        analysis['alternatives'].append("Low-sodium versions of similar products")
    
    return analysis

def save_uploaded_file(file):
    """Save uploaded file and return path"""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    filename = secure_filename(file.filename)
    # Add timestamp to avoid overwriting
    import time
    timestamp = int(time.time())
    name, ext = os.path.splitext(filename)
    filename = f"{name}_{timestamp}{ext}"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save file
        filepath = save_uploaded_file(file)
        print(f"File saved to: {filepath}")
        
        # Analyze with Gemini (falls back to Tesseract if needed)
        nutrition_data = analyze_nutrition_with_gemini(filepath)
        
        if not nutrition_data:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        # Calculate ONQI score using extracted values
        nutrition_facts = nutrition_data.get('nutrition_facts', {})
        onqi_score = calculate_onqi_score(nutrition_facts)
        
        # Get health analysis based on extracted values
        health_analysis = get_health_analysis(nutrition_data, onqi_score)
        
        # Store in session for result page
        session['analysis_result'] = {
            'image_path': filepath,
            'nutrition_data': nutrition_data,
            'health_analysis': health_analysis
        }

        return render_template('result1.html', 
                             image_path=filepath,
                             nutrition_data=nutrition_data,
                             health_analysis=health_analysis)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filepath = save_uploaded_file(file)
        nutrition_data = analyze_nutrition_with_gemini(filepath)
        
        if not nutrition_data:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        nutrition_facts = nutrition_data.get('nutrition_facts', {})
        onqi_score = calculate_onqi_score(nutrition_facts)
        health_analysis = get_health_analysis(nutrition_data, onqi_score)
        
        return jsonify({
            'success': True,
            'nutrition_data': nutrition_data,
            'onqi_score': onqi_score,
            'health_analysis': health_analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/ocr', methods=['POST'])
def debug_ocr():
    """Debug endpoint to test OCR extraction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filepath = save_uploaded_file(file)
        
        # Test Tesseract OCR
        raw_text = extract_text_with_tesseract(filepath)
        parsed_data = parse_nutrition_from_text(raw_text)
        
        return jsonify({
            'success': True,
            'raw_text': raw_text,
            'parsed_data': parsed_data,
            'gemini_available': gemini_available
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
