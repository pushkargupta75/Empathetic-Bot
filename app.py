from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import threading
import time
from collections import deque
import mediapipe as mp
import os
from PIL import Image
import io
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import requests
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini API configured successfully")
else:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
    gemini_model = None

# Global variables
emotion_detector = None
conversation_history = deque(maxlen=20)
emotion_history = deque(maxlen=10)

class GeminiEmotionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Initialize Gemini model
        self.gemini_model = gemini_model
        
        # Emotion cache to avoid too many API calls
        self.last_emotion_call = 0
        self.emotion_call_interval = 1.5  # 1.5 seconds between API calls
        self.cached_emotion = {'emotion': 'neutral', 'confidence': 0.0}
        
        # Face detection threshold
        self.face_confidence_threshold = 0.6
        
    def _create_emotion_prompt(self):
        """Create a detailed prompt for emotion detection"""
        return """Analyze this person's facial expression and determine their emotion. 

Look carefully at:
- Facial muscles and their tension
- Eye shape, eyebrow position, and gaze direction
- Mouth shape, lip position, and any smile/frown
- Overall facial expression and micro-expressions

Classify the emotion as ONE of these categories:
- happy: genuine joy, smiling, positive expressions, relaxed happiness
- sad: downturned mouth, droopy eyes, melancholy, disappointment
- angry: furrowed brow, tight lips, tense jaw, scowling
- fear: wide eyes, tense expression, worried look, anxiety
- surprise: raised eyebrows, wide eyes, open mouth from sudden shock/amazement
- disgust: wrinkled nose, slight frown, aversion, distaste
- neutral: relaxed, calm, no strong emotional expression, resting face

IMPORTANT: Be very careful to distinguish between:
- Surprise (sudden shock/amazement with raised eyebrows) vs just talking or mouth slightly open
- Happy (genuine smile with eye crinkles) vs neutral relaxed expression
- Neutral (calm resting face) vs other subtle emotions
- Fear (worried/anxious) vs surprise (sudden shock)

Pay special attention to:
- Eye crinkles for genuine happiness
- Eyebrow position for surprise vs other emotions
- Mouth tension for anger vs disgust
- Overall facial muscle tension

Respond in this EXACT JSON format:
{
    "emotion": "emotion_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation focusing on key facial features observed",
    "facial_features": "Specific features that led to this classification"
}

Confidence guidelines:
- 0.85-1.0: Very obvious, clear emotion with multiple strong indicators
- 0.65-0.85: Clear emotion with good supporting features
- 0.45-0.65: Moderate confidence, some clear indicators
- 0.25-0.45: Subtle emotion, few indicators
- 0.0-0.25: Very unclear or truly neutral expression

Be conservative - don't over-classify neutral or talking expressions as emotions."""

    def _extract_face_region(self, frame):
        """Extract face region from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            # Get the most confident detection
            best_detection = max(results.detections, 
                               key=lambda d: d.score[0] if d.score else 0)
            
            if best_detection.score[0] >= self.face_confidence_threshold:
                # Extract face bounding box with padding
                bboxC = best_detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # Add padding around face for better context
                padding = 0.4
                x = max(0, int((bboxC.xmin - padding * bboxC.width) * w))
                y = max(0, int((bboxC.ymin - padding * bboxC.height) * h))
                x2 = min(w, int((bboxC.xmin + bboxC.width * (1 + padding)) * w))
                y2 = min(h, int((bboxC.ymin + bboxC.height * (1 + padding)) * h))
                
                face_roi = frame[y:y2, x:x2]
                
                if face_roi.size > 0:
                    return face_roi, best_detection.score[0]
        
        return None, 0.0

    def _analyze_emotion_with_gemini(self, face_image):
        """Use Gemini to analyze emotion in face image"""
        try:
            # Convert face image to PIL format
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            
            # Enhance image quality for better analysis
            if pil_image.size[0] < 200 or pil_image.size[1] < 200:
                # Upscale small images
                new_size = (max(200, pil_image.size[0]), max(200, pil_image.size[1]))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            elif pil_image.size[0] > 800 or pil_image.size[1] > 800:
                # Downscale very large images
                pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            # Create prompt
            prompt = self._create_emotion_prompt()
            
            # Call Gemini Vision API
            response = self.gemini_model.generate_content([prompt, pil_image])
            response_text = response.text.strip()
            
            # Clean and parse JSON response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]
            
            emotion_data = json.loads(response_text)
            
            # Validate and clean response
            valid_emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
            if emotion_data.get('emotion') not in valid_emotions:
                emotion_data['emotion'] = 'neutral'
                emotion_data['confidence'] = 0.3
            
            # Ensure confidence is in valid range
            confidence = float(emotion_data.get('confidence', 0.3))
            emotion_data['confidence'] = max(0.0, min(1.0, confidence))
            
            return emotion_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw Gemini response: {response_text}")
            return {
                'emotion': 'neutral',
                'confidence': 0.2,
                'reasoning': 'Error parsing Gemini response',
                'facial_features': 'Unable to analyze due to parsing error'
            }
        except Exception as e:
            print(f"Gemini emotion analysis error: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.1,
                'reasoning': f'API Error: {str(e)}',
                'facial_features': 'Analysis failed'
            }

    def process_frame(self, frame):
        """Process frame for emotion detection using Gemini Vision"""
        current_time = time.time()
        
        # Extract face region first
        face_roi, detection_confidence = self._extract_face_region(frame)
        
        emotion_data = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'face_detected': False,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'No face detected',
            'facial_features': 'None',
            'detection_confidence': round(detection_confidence, 2)
        }
        
        if face_roi is not None:
            emotion_data['face_detected'] = True
            
            # Use Gemini for emotion analysis (with rate limiting to avoid API quota issues)
            if current_time - self.last_emotion_call >= self.emotion_call_interval:
                try:
                    print(f"🔍 Analyzing emotion with Gemini...")
                    gemini_result = self._analyze_emotion_with_gemini(face_roi)
                    emotion_data.update(gemini_result)
                    self.cached_emotion = gemini_result
                    self.last_emotion_call = current_time
                    print(f"✅ Gemini detected: {gemini_result['emotion']} ({gemini_result['confidence']:.2f})")
                    
                except Exception as e:
                    print(f"❌ Error in Gemini emotion analysis: {e}")
                    # Use cached emotion if available
                    if self.cached_emotion['emotion'] != 'neutral':
                        emotion_data.update(self.cached_emotion)
                        emotion_data['reasoning'] += " (cached - API error)"
            else:
                # Use cached emotion to avoid too many API calls
                if self.cached_emotion['emotion'] != 'neutral':
                    emotion_data.update(self.cached_emotion)
                    emotion_data['reasoning'] += " (cached)"
        
        return emotion_data

class AdvancedEmotionDetector:
    """Improved fallback detector when Gemini is not available"""
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def _calculate_advanced_features(self, landmarks, image_shape):
        """Calculate sophisticated facial features"""
        height, width = image_shape[:2]
        
        if not landmarks:
            return {}
            
        points = np.array([[lm.x * width, lm.y * height] for lm in landmarks.landmark])
        
        features = {}
        
        try:
            # Mouth analysis (more detailed)
            mouth_corners = [points[61], points[291]]  # Left and right corners
            mouth_top = points[13]  # Upper lip center
            mouth_bottom = points[14]  # Lower lip center
            
            mouth_width = np.linalg.norm(mouth_corners[1] - mouth_corners[0])
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
            
            # Mouth curvature (smile/frown detection)
            mouth_center_y = (mouth_corners[0][1] + mouth_corners[1][1]) / 2
            mouth_curvature = mouth_center_y - mouth_top[1]
            
            features['mouth_width'] = mouth_width
            features['mouth_height'] = mouth_height
            features['mouth_ratio'] = mouth_height / mouth_width if mouth_width > 0 else 0
            features['mouth_curvature'] = mouth_curvature / height  # Normalized
            
            # Eye analysis (both eyes)
            left_eye_points = points[[33, 7, 163, 144, 145, 153]]
            right_eye_points = points[[362, 382, 381, 380, 374, 373]]
            
            # Left eye aspect ratio
            left_eye_height = np.mean([
                np.linalg.norm(left_eye_points[1] - left_eye_points[5]),
                np.linalg.norm(left_eye_points[2] - left_eye_points[4])
            ])
            left_eye_width = np.linalg.norm(left_eye_points[0] - left_eye_points[3])
            
            # Right eye aspect ratio
            right_eye_height = np.mean([
                np.linalg.norm(right_eye_points[1] - right_eye_points[5]),
                np.linalg.norm(right_eye_points[2] - right_eye_points[4])
            ])
            right_eye_width = np.linalg.norm(right_eye_points[0] - right_eye_points[3])
            
            features['left_eye_ratio'] = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            features['right_eye_ratio'] = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            features['avg_eye_ratio'] = (features['left_eye_ratio'] + features['right_eye_ratio']) / 2
            
            # Eyebrow position
            left_eyebrow_y = np.mean([points[70][1], points[63][1], points[105][1]])
            right_eyebrow_y = np.mean([points[296][1], points[334][1], points[293][1]])
            left_eye_y = np.mean([points[33][1], points[133][1]])
            right_eye_y = np.mean([points[362][1], points[263][1]])
            
            features['eyebrow_distance'] = ((left_eye_y - left_eyebrow_y) + (right_eye_y - right_eyebrow_y)) / (2 * height)
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            
        return features
    
    def _classify_emotion_improved(self, features):
        """Improved emotion classification with better thresholds"""
        if not features:
            return 'neutral', 0.3
        
        mouth_ratio = features.get('mouth_ratio', 0)
        mouth_curvature = features.get('mouth_curvature', 0)
        eye_ratio = features.get('avg_eye_ratio', 0.2)
        eyebrow_distance = features.get('eyebrow_distance', 0.05)
        
        # Improved classification logic
        
        # Happy: Positive mouth curvature + moderate mouth opening
        if mouth_curvature > 0.008 and mouth_ratio > 0.015:
            confidence = min(0.85, 0.5 + mouth_curvature * 25 + mouth_ratio * 8)
            return 'happy', confidence
            
        # Sad: Negative mouth curvature + droopy eyes
        elif mouth_curvature < -0.003 and eye_ratio < 0.18:
            confidence = min(0.8, 0.5 + abs(mouth_curvature) * 20 + (0.18 - eye_ratio) * 4)
            return 'sad', confidence
            
        # Angry: Low eyebrows + tight mouth
        elif eyebrow_distance < 0.035 and mouth_ratio < 0.015:
            confidence = min(0.8, 0.5 + (0.035 - eyebrow_distance) * 15)
            return 'angry', confidence
            
        # Surprise: High eyebrows + wide eyes + open mouth (ALL THREE required)
        elif eyebrow_distance > 0.065 and eye_ratio > 0.25 and mouth_ratio > 0.035:
            confidence = min(0.85, 0.6 + (eyebrow_distance - 0.065) * 10 + eye_ratio * 2)
            return 'surprise', confidence
            
        # Fear: Wide eyes + high eyebrows (but not as extreme as surprise)
        elif eye_ratio > 0.24 and eyebrow_distance > 0.055 and mouth_ratio < 0.03:
            confidence = min(0.75, 0.5 + eye_ratio * 2)
            return 'fear', confidence
            
        # Disgust: Slight squint + downturned mouth
        elif eye_ratio < 0.16 and mouth_curvature < -0.001 and mouth_ratio < 0.02:
            confidence = min(0.7, 0.5 + (0.16 - eye_ratio) * 3)
            return 'disgust', confidence
            
        else:
            # Neutral - be more conservative
            return 'neutral', 0.4

    def process_frame(self, frame):
        """Process frame with improved rule-based detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        mesh_results = self.mp_face_mesh.process(rgb_frame)
        
        emotion_data = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'face_detected': False,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'No face detected',
            'facial_features': 'None'
        }
        
        if results.detections and mesh_results.multi_face_landmarks:
            detection = results.detections[0]
            if detection.score[0] >= 0.7:
                landmarks = mesh_results.multi_face_landmarks[0]
                features = self._calculate_advanced_features(landmarks, frame.shape)
                emotion, confidence = self._classify_emotion_improved(features)
                
                emotion_data = {
                    'emotion': emotion,
                    'confidence': round(confidence, 2),
                    'face_detected': True,
                    'timestamp': datetime.now().isoformat(),
                    'reasoning': f'Advanced rule-based: {emotion} detected',
                    'facial_features': f'Mouth: {features.get("mouth_ratio", 0):.3f}, Eyes: {features.get("avg_eye_ratio", 0):.3f}, Brows: {features.get("eyebrow_distance", 0):.3f}',
                    'detection_confidence': round(detection.score[0], 2)
                }
        
        return emotion_data

class MultimodalGeminiBot:
    def __init__(self):
        self.model = gemini_model
        self.conversation_context = []
        
    def analyze_image(self, image_data, prompt="Analyze this image and describe what you see"):
        """Analyze image using Gemini Vision"""
        if not self.model:
            return "Gemini API not configured. Please set GEMINI_API_KEY environment variable."
        
        try:
            # Convert base64 to PIL Image
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate response
            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def generate_empathetic_response(self, user_message, emotion_data=None, image_analysis=None):
        """Generate empathetic response using Gemini with enhanced emotion context"""
        if not self.model:
            return self._fallback_response(user_message, emotion_data)
        
        try:
            # Build enhanced context-aware prompt
            context = self._build_enhanced_context_prompt(user_message, emotion_data, image_analysis)
            
            # Generate response
            response = self.model.generate_content(context)
            response_text = response.text
            
            # Store in conversation history
            self.conversation_context.append({
                'user': user_message,
                'assistant': response_text,
                'emotion': emotion_data.get('emotion', 'neutral') if emotion_data else 'neutral',
                'emotion_confidence': emotion_data.get('confidence', 0) if emotion_data else 0,
                'timestamp': datetime.now().isoformat()
            })
            
            return response_text
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._fallback_response(user_message, emotion_data)
    
    def _build_enhanced_context_prompt(self, user_message, emotion_data, image_analysis):
        """Build enhanced context-aware prompt for Gemini"""
        prompt = """You are an empathetic AI assistant that understands human emotions deeply and provides supportive, caring responses.

Current conversation context:"""
        
        # Add recent conversation history
        if self.conversation_context:
            prompt += "\nRecent conversation history:\n"
            for msg in self.conversation_context[-3:]:
                emotion_info = f" (feeling {msg['emotion']})" if msg['emotion'] != 'neutral' else ""
                prompt += f"User{emotion_info}: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Add detailed emotion context
        if emotion_data and emotion_data.get('face_detected'):
            emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0)
            reasoning = emotion_data.get('reasoning', 'Unknown')
            features = emotion_data.get('facial_features', 'Not analyzed')
            
            prompt += f"""
EMOTIONAL ANALYSIS:
- Detected emotion: {emotion} 
- Confidence level: {confidence:.0%}
- Analysis reasoning: {reasoning}
- Facial features observed: {features}

RESPONSE GUIDELINES for {emotion.upper()} emotion:"""
            
            emotion_guidelines = {
                'happy': "The user appears joyful! Match their positive energy, celebrate with them, ask about what's making them happy, and encourage their good mood.",
                'sad': "The user seems sad or down. Be gentle, compassionate, and supportive. Acknowledge their feelings, offer comfort, and ask if they'd like to talk about what's bothering them.",
                'angry': "The user appears frustrated or angry. Stay calm and understanding. Don't dismiss their feelings. Ask what's wrong and how you can help them work through it.",
                'fear': "The user looks worried or anxious. Be reassuring and calming. Offer support and ask if there's something specific they're concerned about.",
                'surprise': "The user seems surprised or shocked. Be curious about what surprised them. Ask for details and show interest in their experience.",
                'disgust': "The user appears disgusted or put off by something. Be understanding and ask what's bothering them. Offer to help them process their feelings.",
                'neutral': "The user has a calm, neutral expression. Engage naturally and follow their conversational lead."
            }
            
            prompt += f"\n{emotion_guidelines.get(emotion, emotion_guidelines['neutral'])}\n"
        
        # Add image analysis context
        if image_analysis:
            prompt += f"\nIMAGE CONTEXT: {image_analysis}\nConsider this visual information in your response.\n"
        
        prompt += f"\nUSER'S CURRENT MESSAGE: {user_message}\n\n"
        prompt += """Please respond with:
1. Genuine empathy and emotional intelligence
2. Appropriate tone matching their emotional state
3. Supportive and caring language
4. Natural, conversational style
5. Specific acknowledgment of their feelings when appropriate

Keep responses concise but meaningful (2-4 sentences unless more detail is needed).

Your empathetic response:"""
        
        return prompt
    
    def _fallback_response(self, user_message, emotion_data):
        """Enhanced fallback responses when Gemini API is unavailable"""
        if not emotion_data or not emotion_data.get('face_detected'):
            return "I'm here to listen and help. Could you tell me more about what's on your mind?"
        
        emotion = emotion_data.get('emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0)
        
        # More nuanced fallback responses
        responses = {
            'happy': f"I can see you're feeling great (confidence: {confidence:.0%})! Your happiness is wonderful to see. What's bringing you such joy today?",
            'sad': f"I notice you seem to be feeling down (confidence: {confidence:.0%}). I'm here to listen. Sometimes sharing what's on our mind can help lighten the load.",
            'angry': f"I can sense some frustration in your expression (confidence: {confidence:.0%}). Take a moment to breathe. What's bothering you? I'm here to help you work through it.",
            'fear': f"You look a bit worried or anxious (confidence: {confidence:.0%}). That's completely understandable. What's on your mind? Sometimes talking about our concerns helps.",
            'surprise': f"You seem surprised about something (confidence: {confidence:.0%})! Did something unexpected happen? I'd love to hear about what caught you off guard.",
            'disgust': f"I can see something's bothering you (confidence: {confidence:.0%}). Would you like to share what's troubling you? Sometimes it helps to talk it out.",
            'neutral': f"You have a calm, peaceful expression (confidence: {confidence:.0%}). I'm here and ready to chat about whatever's on your mind. How can I help you today?"
        }
        
        return responses.get(emotion, responses['neutral'])

def initialize_services():
    """Initialize emotion detection services"""
    global emotion_detector, multimodal_bot
    
    print("🚀 Initializing Enhanced Empathetic AI...")
    
    # Always try Gemini-based emotion detection first if available
    if GEMINI_API_KEY and gemini_model:
        try:
            emotion_detector = GeminiEmotionDetector()
            print("✅ Gemini Vision emotion detection initialized!")
            print("🎯 This will provide much more accurate emotion recognition!")
        except Exception as e:
            print(f"❌ Gemini emotion detection failed: {e}")
            print("🔄 Falling back to improved rule-based detection...")
            emotion_detector = AdvancedEmotionDetector()
    else:
        print("⚠️ No Gemini API key found, using improved rule-based detection...")
        print("💡 Set GEMINI_API_KEY environment variable for AI-powered emotion detection!")
        emotion_detector = AdvancedEmotionDetector()
    
    multimodal_bot = MultimodalGeminiBot()
    print("✅ All services initialized successfully!")

def enhanced_smooth_emotion_history(history):
    """Enhanced temporal smoothing with better logic"""
    if not history:
        return {'emotion': 'neutral', 'confidence': 0.0, 'face_detected': False}
    
    recent = list(history)[-8:]  # Look at last 8 readings
    face_detected_readings = [r for r in recent if r.get('face_detected', False)]
    
    if not face_detected_readings:
        return recent[-1] if recent else {
            'emotion': 'neutral', 
            'confidence': 0.0, 
            'face_detected': False
        }
    
    # Weight recent readings more heavily
    weights = np.exp(np.linspace(-1.5, 0, len(face_detected_readings)))
    weights = weights / weights.sum()
    
    emotion_scores = {}
    total_reasoning = []
    
    for i, reading in enumerate(face_detected_readings):
        emotion = reading['emotion']
        confidence = reading['confidence']
        weight = weights[i]
        
        if emotion not in emotion_scores:
            emotion_scores[emotion] = 0
        emotion_scores[emotion] += confidence * weight
        
        if i >= len(face_detected_readings) - 2:  # Last 2 readings
            total_reasoning.append(reading.get('reasoning', ''))
    
    if emotion_scores:
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        final_confidence = min(0.95, best_emotion[1])
        
        # Get most recent reasoning
        latest_reading = face_detected_readings[-1]
        
        return {
            'emotion': best_emotion[0],
            'confidence': round(final_confidence, 2),
            'face_detected': True,
            'timestamp': datetime.now().isoformat(),
            'reasoning': latest_reading.get('reasoning', 'Smoothed analysis'),
            'facial_features': latest_reading.get('facial_features', 'Analyzed'),
            'stability': len([r for r in face_detected_readings if r['emotion'] == best_emotion[0]]) / len(face_detected_readings),
            'detection_confidence': latest_reading.get('detection_confidence', 0.8)
        }
    
    return recent[-1] if recent else {
        'emotion': 'neutral', 
        'confidence': 0.0, 
        'face_detected': False
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'emotion_detector': emotion_detector is not None,
        'gemini_available': gemini_model is not None,
        'detector_type': 'Gemini Vision' if isinstance(emotion_detector, GeminiEmotionDetector) else 'Rule-based',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    print('🔗 Client connected')
    detector_type = 'Gemini Vision AI' if isinstance(emotion_detector, GeminiEmotionDetector) else 'Advanced Rule-based'
    emit('status', {
        'message': f'Connected to Enhanced Empathetic AI ({detector_type})',
        'gemini_available': gemini_model is not None,
        'detector_type': detector_type
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frames for emotion detection with enhanced accuracy"""
    global emotion_detector, emotion_history
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None and emotion_detector:
            # Process frame
            emotion_data = emotion_detector.process_frame(frame)
            emotion_history.append(emotion_data)
            
            # Apply enhanced temporal smoothing
            smoothed_emotion = enhanced_smooth_emotion_history(emotion_history)
            
            # Add debug info
            if isinstance(emotion_detector, GeminiEmotionDetector):
                smoothed_emotion['analysis_method'] = 'Gemini Vision AI'
            else:
                smoothed_emotion['analysis_method'] = 'Advanced Rule-based'
                
            emit('emotion_update', smoothed_emotion)
                
    except Exception as e:
        print(f"❌ Error processing video frame: {e}")
        emit('error', {'message': 'Error processing video frame'})

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat messages with multimodal support"""
    global multimodal_bot, emotion_history
    
    try:
        user_message = data.get('message', '').strip()
        if not user_message:
            return
        
        # Get current emotion context
        current_emotion = emotion_history[-1] if emotion_history else None
        
        # Handle image analysis if image is provided
        image_analysis = None
        if 'image' in data and data['image']:
            image_analysis = multimodal_bot.analyze_image(
                data['image'], 
                data.get('image_prompt', 'Analyze this image and describe what you see')
            )
        
        # Generate empathetic response
        response = multimodal_bot.generate_empathetic_response(
            user_message, 
            current_emotion,
            image_analysis
        )
        
        # Emit response with enhanced context
        emit('chat_response', {
            'message': response,
            'emotion_context': current_emotion,
            'image_analysis': image_analysis,
            'timestamp': datetime.now().isoformat(),
            'analysis_method': current_emotion.get('analysis_method', 'Unknown') if current_emotion else 'Unknown'
        })
        
    except Exception as e:
        print(f"❌ Error handling chat message: {e}")
        emit('error', {'message': 'Error processing your message'})

@socketio.on('analyze_image')
def handle_image_analysis(data):
    """Handle standalone image analysis"""
    global multimodal_bot
    
    try:
        image_data = data.get('image')
        prompt = data.get('prompt', 'Analyze this image in detail')
        
        if not image_data:
            emit('error', {'message': 'No image provided'})
            return
        
        analysis = multimodal_bot.analyze_image(image_data, prompt)
        
        emit('image_analysis_result', {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        emit('error', {'message': 'Error analyzing image'})

if __name__ == '__main__':
    # Initialize services
    threading.Thread(target=initialize_services, daemon=True).start()
    time.sleep(3)  # Wait for initialization
    
    print("🌟 Starting Enhanced Empathetic AI Assistant...")
    print(f"🔑 Gemini API: {'✅ Available' if GEMINI_API_KEY else '❌ Not configured'}")
    print(f"🎯 Emotion Detection: {'Gemini Vision AI' if GEMINI_API_KEY else 'Advanced Rule-based'}")
    
    if not GEMINI_API_KEY:
        print("\n💡 To get much better emotion detection:")
        print("   1. Get API key from https://makersuite.google.com/app/apikey")
        print("   2. Set environment variable: export GEMINI_API_KEY=your_key")
        print("   3. Restart the application")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        allow_unsafe_werkzeug=True
    )