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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Set your API key in environment
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")
    gemini_model = None

# Global variables
emotion_detector = None
conversation_history = deque(maxlen=20)
emotion_history = deque(maxlen=10)

class AdvancedEmotionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Try to load pre-trained emotion model (if available)
        self.emotion_model = self._load_emotion_model()
        
        # Face landmark indices for emotion analysis
        self.mouth_landmarks = [61, 84, 17, 314, 405, 320, 308, 324, 318]
        self.eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.eyebrow_landmarks = [46, 53, 52, 65, 55, 70]
        
    def _load_emotion_model(self):
        """Try to load a pre-trained emotion detection model"""
        try:
            # You can replace this with your own trained model
            # For now, we'll use a simple rule-based approach
            return None
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            return None
    
    def _extract_facial_features(self, landmarks, image_shape):
        """Extract facial features for emotion detection"""
        height, width = image_shape[:2]
        
        features = {
            'mouth_aspect_ratio': 0,
            'eye_aspect_ratio': 0,
            'eyebrow_height': 0,
            'face_symmetry': 0
        }
        
        if not landmarks:
            return features
            
        # Convert normalized coordinates to pixel coordinates
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append([x, y])
        points = np.array(points)
        
        # Calculate mouth aspect ratio
        if len(points) > max(self.mouth_landmarks):
            mouth_points = points[self.mouth_landmarks]
            mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[4])
            mouth_height = np.linalg.norm(mouth_points[2] - mouth_points[6])
            features['mouth_aspect_ratio'] = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Calculate eye aspect ratio (average of both eyes)
        if len(points) > max(self.eye_landmarks):
            eye_points = points[self.eye_landmarks]
            left_eye_height = np.linalg.norm(eye_points[1] - eye_points[5])
            left_eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
            right_eye_height = np.linalg.norm(eye_points[7] - eye_points[11])
            right_eye_width = np.linalg.norm(eye_points[6] - eye_points[9])
            
            left_ear = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_ear = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            features['eye_aspect_ratio'] = (left_ear + right_ear) / 2
        
        return features
    
    def _classify_emotion(self, features, face_roi=None):
        """Classify emotion based on facial features"""
        if self.emotion_model and face_roi is not None:
            try:
                # Preprocess face for model prediction
                face_resized = cv2.resize(face_roi, (48, 48))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_array = img_to_array(face_gray)
                face_array = np.expand_dims(face_array, axis=0)
                face_array /= 255.0
                
                predictions = self.emotion_model.predict(face_array)
                emotion_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][emotion_idx])
                
                return self.emotion_labels[emotion_idx], confidence
            except Exception as e:
                print(f"Model prediction error: {e}")
        
        # Rule-based emotion detection
        mouth_ratio = features['mouth_aspect_ratio']
        eye_ratio = features['eye_aspect_ratio']
        
        # Emotion classification rules
        if mouth_ratio > 0.05:  # Mouth open/smiling
            if mouth_ratio > 0.08:
                return 'surprise', 0.75 + min(0.2, mouth_ratio * 2)
            else:
                return 'happy', 0.65 + min(0.3, mouth_ratio * 4)
        elif mouth_ratio < 0.02:  # Mouth closed/frowning
            if eye_ratio < 0.15:  # Eyes squinted
                return 'angry', 0.6 + min(0.3, (0.15 - eye_ratio) * 3)
            else:
                return 'sad', 0.6 + min(0.25, (0.02 - mouth_ratio) * 10)
        elif eye_ratio < 0.12:  # Eyes very closed
            return 'disgust', 0.65
        elif eye_ratio > 0.25:  # Eyes wide open
            return 'fear', 0.7
        else:
            return 'neutral', 0.5 + min(0.3, abs(mouth_ratio - 0.03) * 5)
    
    def process_frame(self, frame):
        """Process frame for emotion detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        emotion_data = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'face_detected': False,
            'timestamp': datetime.now().isoformat(),
            'features': {}
        }
        
        if results.detections and mesh_results.multi_face_landmarks:
            # Get face bounding box
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            
            x, y, box_w, box_h = bbox
            face_roi = frame[max(0, y):min(h, y+box_h), max(0, x):min(w, x+box_w)]
            
            # Extract facial features
            face_landmarks = mesh_results.multi_face_landmarks[0]
            features = self._extract_facial_features(face_landmarks, frame.shape)
            
            # Classify emotion
            if face_roi.size > 0:
                emotion, confidence = self._classify_emotion(features, face_roi)
                emotion_data = {
                    'emotion': emotion,
                    'confidence': round(confidence, 2),
                    'face_detected': True,
                    'timestamp': datetime.now().isoformat(),
                    'features': features
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
        """Generate empathetic response using Gemini"""
        if not self.model:
            return self._fallback_response(user_message, emotion_data)
        
        try:
            # Build context-aware prompt
            context = self._build_context_prompt(user_message, emotion_data, image_analysis)
            
            # Generate response
            response = self.model.generate_content(context)
            response_text = response.text
            
            # Store in conversation history
            self.conversation_context.append({
                'user': user_message,
                'assistant': response_text,
                'emotion': emotion_data.get('emotion', 'neutral') if emotion_data else 'neutral',
                'timestamp': datetime.now().isoformat()
            })
            
            return response_text
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._fallback_response(user_message, emotion_data)
    
    def _build_context_prompt(self, user_message, emotion_data, image_analysis):
        """Build context-aware prompt for Gemini"""
        prompt = """You are an empathetic AI assistant that understands human emotions and provides supportive responses. 

Current conversation context:"""
        
        # Add recent conversation history
        if self.conversation_context:
            prompt += "\nRecent conversation:\n"
            for msg in self.conversation_context[-3:]:  # Last 3 exchanges
                prompt += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Add emotion context
        if emotion_data and emotion_data.get('face_detected'):
            emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0)
            prompt += f"\nUser's current emotion: {emotion} (confidence: {confidence:.0%})\n"
            prompt += f"Please respond with empathy appropriate to their {emotion} emotional state.\n"
        
        # Add image analysis context
        if image_analysis:
            prompt += f"\nImage context: {image_analysis}\n"
            prompt += "Consider this visual information in your response.\n"
        
        prompt += f"\nUser's message: {user_message}\n\n"
        prompt += """Please provide a warm, empathetic response that:
1. Acknowledges their emotional state
2. Provides appropriate support or encouragement
3. Is conversational and natural
4. Shows genuine understanding and care

Response:"""
        
        return prompt
    
    def _fallback_response(self, user_message, emotion_data):
        """Fallback responses when Gemini API is unavailable"""
        if not emotion_data:
            return "I'm here to listen and help. Could you tell me more about what's on your mind?"
        
        emotion = emotion_data.get('emotion', 'neutral')
        responses = {
            'happy': "I can see you're in a great mood! That's wonderful. What's making you feel so positive today?",
            'sad': "I notice you seem a bit down. I'm here to listen. Sometimes talking about what's bothering us can help.",
            'angry': "I can sense some frustration. Take a deep breath. Would you like to talk about what's making you feel this way?",
            'fear': "You seem worried about something. That's completely understandable. What's on your mind?",
            'surprise': "You look surprised! Something unexpected happen? I'd love to hear about it.",
            'disgust': "I can see something's bothering you. Would you like to share what's troubling you?",
            'neutral': "I'm here and ready to chat. How can I help you today?"
        }
        
        return responses.get(emotion, responses['neutral'])

def initialize_services():
    """Initialize all services"""
    global emotion_detector, multimodal_bot
    
    print("🚀 Initializing Enhanced Empathetic AI...")
    emotion_detector = AdvancedEmotionDetector()
    multimodal_bot = MultimodalGeminiBot()
    print("✅ Services initialized successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'emotion_detector': emotion_detector is not None,
        'gemini_available': gemini_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    print('🔗 Client connected')
    emit('status', {
        'message': 'Connected to Enhanced Empathetic AI',
        'gemini_available': gemini_model is not None
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frames for emotion detection"""
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
            
            # Apply temporal smoothing
            smoothed_emotion = smooth_emotion_history(emotion_history)
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
        
        # Emit response
        emit('chat_response', {
            'message': response,
            'emotion_context': current_emotion,
            'image_analysis': image_analysis,
            'timestamp': datetime.now().isoformat()
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

def smooth_emotion_history(history):
    """Apply temporal smoothing to emotion detection"""
    if not history:
        return {'emotion': 'neutral', 'confidence': 0.0, 'face_detected': False}
    
    recent = list(history)[-5:]
    emotion_counts = {}
    total_confidence = 0
    face_detected_count = 0
    
    for reading in recent:
        emotion = reading['emotion']
        confidence = reading['confidence']
        
        if emotion not in emotion_counts:
            emotion_counts[emotion] = []
        emotion_counts[emotion].append(confidence)
        
        total_confidence += confidence
        if reading['face_detected']:
            face_detected_count += 1
    
    if emotion_counts:
        # Find most frequent emotion with highest confidence
        best_emotion = max(emotion_counts.items(), 
                          key=lambda x: (len(x[1]), np.mean(x[1])))
        
        return {
            'emotion': best_emotion[0],
            'confidence': round(np.mean(best_emotion[1]), 2),
            'face_detected': face_detected_count > len(recent) // 2,
            'timestamp': datetime.now().isoformat(),
            'stability': len(best_emotion[1]) / len(recent)
        }
    
    return recent[-1] if recent else {
        'emotion': 'neutral', 
        'confidence': 0.0, 
        'face_detected': False
    }

if __name__ == '__main__':
    # Initialize services
    threading.Thread(target=initialize_services, daemon=True).start()
    time.sleep(3)  # Wait for initialization
    
    print("🌟 Starting Enhanced Empathetic AI Assistant...")
    print(f"🔑 Gemini API: {'✅ Available' if GEMINI_API_KEY else '❌ Not configured'}")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        allow_unsafe_werkzeug=True
    )