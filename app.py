from flask import Flask, render_template, request
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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables for services
emotion_detector = None
text_generator = None
emotion_history = deque(maxlen=10)  # Store last 10 emotion readings

class EmotionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        
        # Emotion labels for basic emotion classification
        self.emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
    def detect_emotion_from_face(self, face_roi):
        """Simple rule-based emotion detection based on facial features"""
        # This is a simplified version - in production you'd use a trained model
        height, width = face_roi.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristic based on pixel intensity distribution
        # This is just for demo - replace with actual trained emotion model
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Simple emotion classification based on facial characteristics
        if mean_intensity > 130:
            emotion = 'happy'
            confidence = min(0.9, (mean_intensity - 100) / 100)
        elif mean_intensity < 90:
            emotion = 'sad'
            confidence = min(0.8, (100 - mean_intensity) / 50)
        elif std_intensity > 40:
            emotion = 'surprised'
            confidence = min(0.75, std_intensity / 60)
        else:
            emotion = 'neutral'
            confidence = 0.6 + (std_intensity / 100)
            
        return emotion, max(0.5, confidence)
    
    def process_frame(self, frame):
        """Process a single frame and return emotion data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        emotion_data = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'face_detected': False,
            'timestamp': datetime.now().isoformat()
        }
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                
                x, y, w, h = bbox
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    emotion, confidence = self.detect_emotion_from_face(face_roi)
                    emotion_data = {
                        'emotion': emotion,
                        'confidence': round(confidence, 2),
                        'face_detected': True,
                        'timestamp': datetime.now().isoformat()
                    }
                    break
        
        return emotion_data

class EmpathyBot:
    def __init__(self):
        print("Loading language model...")
        # Use a smaller model for better performance
        model_name = "microsoft/DialoGPT-medium"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Language model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None
    
    def generate_empathetic_response(self, user_message, emotion_data, conversation_history):
        """Generate an empathetic response based on user input and emotion"""
        if not self.model or not self.tokenizer:
            return self._fallback_response(user_message, emotion_data)
        
        emotion = emotion_data.get('emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.0)
        
        # Create empathetic system context
        emotion_context = self._get_emotion_context(emotion, confidence)
        
        # Build prompt with emotion awareness
        prompt = f"{emotion_context}\nUser: {user_message}\nAssistant:"
        
        try:
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response
            response = response.split("Assistant:")[-1].strip()
            
            return response if response else self._fallback_response(user_message, emotion_data)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._fallback_response(user_message, emotion_data)
    
    def _get_emotion_context(self, emotion, confidence):
        """Get contextual information based on detected emotion"""
        contexts = {
            'happy': "The user appears happy and upbeat. Respond with enthusiasm and positivity.",
            'sad': "The user seems sad or down. Be gentle, supportive, and offer comfort.",
            'angry': "The user appears frustrated or angry. Respond calmly and try to de-escalate.",
            'fearful': "The user seems anxious or worried. Provide reassurance and support.",
            'surprised': "The user appears surprised. Be engaging and match their energy.",
            'disgusted': "The user seems displeased. Be understanding and helpful.",
            'neutral': "The user appears calm. Respond naturally and helpfully."
        }
        
        base_context = contexts.get(emotion, contexts['neutral'])
        if confidence > 0.7:
            return f"{base_context} (High confidence: {confidence:.0%})"
        else:
            return f"{base_context} (Moderate confidence: {confidence:.0%})"
    
    def _fallback_response(self, user_message, emotion_data):
        """Fallback responses when model fails"""
        emotion = emotion_data.get('emotion', 'neutral')
        
        responses = {
            'happy': "I can see you're in a good mood! That's wonderful to hear. How can I help you today?",
            'sad': "I notice you might be feeling down. I'm here to listen and support you. What's on your mind?",
            'angry': "I sense some frustration. Take a deep breath - I'm here to help work through whatever is bothering you.",
            'fearful': "It seems like you might be worried about something. Don't worry, we can figure this out together.",
            'surprised': "You look surprised! What's caught your attention? I'm curious to know more.",
            'disgusted': "I can see something might be bothering you. Let's talk about it - I'm here to help.",
            'neutral': "I'm here and ready to chat! What would you like to talk about today?"
        }
        
        return responses.get(emotion, "I'm here to help! What can I do for you today?")

def initialize_services():
    """Initialize emotion detection and text generation services"""
    global emotion_detector, text_generator
    
    print("Initializing services...")
    emotion_detector = EmotionDetector()
    text_generator = EmpathyBot()
    print("Services initialized successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to Empathetic Chat Bot'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process incoming video frames for emotion detection"""
    global emotion_detector, emotion_history
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None and emotion_detector:
            # Process frame for emotion detection
            emotion_data = emotion_detector.process_frame(frame)
            
            # Add to history for temporal smoothing
            emotion_history.append(emotion_data)
            
            # Apply temporal smoothing
            if len(emotion_history) >= 3:
                smoothed_emotion = smooth_emotion_history(emotion_history)
                emit('emotion_update', smoothed_emotion)
            else:
                emit('emotion_update', emotion_data)
                
    except Exception as e:
        print(f"Error processing video frame: {e}")
        emit('error', {'message': 'Error processing video frame'})

def smooth_emotion_history(history):
    """Apply temporal smoothing to reduce emotion detection jitter"""
    if not history:
        return {'emotion': 'neutral', 'confidence': 0.0, 'face_detected': False}
    
    # Get recent emotions
    recent = list(history)[-5:]  # Last 5 readings
    
    # Count emotion frequencies
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
    
    # Find most frequent emotion with highest average confidence
    if emotion_counts:
        best_emotion = max(emotion_counts.items(), 
                          key=lambda x: (len(x[1]), np.mean(x[1])))
        
        return {
            'emotion': best_emotion[0],
            'confidence': round(np.mean(best_emotion[1]), 2),
            'face_detected': face_detected_count > len(recent) // 2,
            'timestamp': datetime.now().isoformat()
        }
    
    return recent[-1] if recent else {'emotion': 'neutral', 'confidence': 0.0, 'face_detected': False}

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle incoming chat messages and generate empathetic responses"""
    global text_generator, emotion_history
    
    try:
        user_message = data.get('message', '').strip()
        if not user_message:
            return
        
        # Get current emotion context
        current_emotion = emotion_history[-1] if emotion_history else {
            'emotion': 'neutral', 'confidence': 0.0, 'face_detected': False
        }
        
        # Generate empathetic response
        if text_generator:
            response = text_generator.generate_empathetic_response(
                user_message, 
                current_emotion, 
                []  # conversation history placeholder
            )
        else:
            response = "I'm here to listen and help. Could you tell me more about what's on your mind?"
        
        # Emit response back to client
        emit('chat_response', {
            'message': response,
            'emotion_context': current_emotion,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error handling chat message: {e}")
        emit('error', {'message': 'Error processing your message'})

if __name__ == '__main__':
    # Initialize services in a separate thread to avoid blocking
    threading.Thread(target=initialize_services, daemon=True).start()
    
    # Give services time to initialize
    time.sleep(2)
    
    print("Starting Empathetic Chat Bot...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)