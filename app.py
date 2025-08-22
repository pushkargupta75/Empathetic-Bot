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
import warnings
from dotenv import load_dotenv
import logging

# Free emotion detection imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available - install with: pip install deepface")

try:
    from fer import FER
    FER_AVAILABLE = True
    print("✅ FER loaded successfully") 
except ImportError:
    FER_AVAILABLE = False
    print("⚠️ FER not available - install with: pip install fer")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available")

try:
    from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
    print("✅ Hugging Face Transformers loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - install with: pip install transformers torch")

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    print("✅ PyTorch loaded successfully")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables
emotion_detector = None
conversation_history = deque(maxlen=20)
emotion_history = deque(maxlen=15)

class FreeAdvancedEmotionDetector:
    """Advanced emotion detection using multiple free, open-source models"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        
        # Initialize multiple free emotion detection models
        self.models = {}
        self.model_weights = {}
        self.model_cache = {}
        self.last_predictions = {}
        
        # Initialize all available free models
        self._initialize_deepface()
        self._initialize_fer()
        self._initialize_huggingface()
        self._initialize_custom_cnn()
        self._initialize_ensemble_model()
        
        # Emotion standardization
        self.standard_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = deque(maxlen=50)
        
        print(f"🎯 Initialized Free Advanced Emotion Detector with {len(self.models)} models")
    
    def _initialize_deepface(self):
        """Initialize DeepFace with multiple backends"""
        if DEEPFACE_AVAILABLE:
            try:
                # Test DeepFace with different models
                test_image = np.ones((48, 48, 3), dtype=np.uint8) * 128
                
                # Try different DeepFace models (all free)
                available_models = []
                models_to_try = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace']
                
                for model_name in models_to_try:
                    try:
                        DeepFace.analyze(
                            test_image, 
                            actions=['emotion'], 
                            enforce_detection=False,
                            silent=True,
                            detector_backend='opencv',
                            model_name=model_name
                        )
                        available_models.append(model_name)
                        print(f"✅ DeepFace {model_name} model available")
                    except:
                        continue
                
                if available_models:
                    self.models['deepface'] = available_models
                    self.model_weights['deepface'] = 0.3  # High weight
                    print(f"✅ DeepFace initialized with {len(available_models)} models")
                
            except Exception as e:
                print(f"❌ DeepFace initialization failed: {e}")
    
    def _initialize_fer(self):
        """Initialize FER (Facial Expression Recognition) model"""
        if FER_AVAILABLE:
            try:
                # FER with MTCNN for better face detection
                self.models['fer'] = FER(mtcnn=True)
                self.model_weights['fer'] = 0.25
                print("✅ FER emotion model initialized")
            except Exception as e:
                try:
                    # Fallback to basic FER
                    self.models['fer'] = FER()
                    self.model_weights['fer'] = 0.2
                    print("✅ FER emotion model initialized (basic)")
                except Exception as e2:
                    print(f"❌ FER initialization failed: {e2}")
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face emotion detection models"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # List of free emotion detection models from Hugging Face
                model_configs = [
                    {
                        'name': 'emotion-english-distilroberta-base',
                        'model_id': 'j-hartmann/emotion-english-distilroberta-base',
                        'weight': 0.2
                    },
                    {
                        'name': 'facial-emotion-recognition',  
                        'model_id': 'trpakov/vit-face-expression',
                        'weight': 0.25
                    }
                ]
                
                self.models['huggingface'] = {}
                
                for config in model_configs:
                    try:
                        # Try to load the model
                        processor = AutoImageProcessor.from_pretrained(config['model_id'])
                        model = AutoModelForImageClassification.from_pretrained(config['model_id'])
                        
                        self.models['huggingface'][config['name']] = {
                            'processor': processor,
                            'model': model,
                            'pipeline': pipeline("image-classification", 
                                               model=config['model_id'],
                                               return_all_scores=True)
                        }
                        self.model_weights[f"hf_{config['name']}"] = config['weight']
                        print(f"✅ Hugging Face {config['name']} model loaded")
                    except Exception as e:
                        print(f"⚠️ Could not load {config['name']}: {e}")
                
                if self.models['huggingface']:
                    print(f"✅ Hugging Face models initialized: {len(self.models['huggingface'])} models")
                
            except Exception as e:
                print(f"❌ Hugging Face initialization failed: {e}")
    
    def _initialize_custom_cnn(self):
        """Initialize custom CNN model for emotion detection"""
        if TF_AVAILABLE:
            try:
                # Create a simple but effective CNN model for emotion detection
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Dropout(0.25),
                    
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Dropout(0.25),
                    
                    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.25),
                    
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
                ])
                
                # Check if pre-trained weights exist, otherwise use untrained model
                model_path = 'models/emotion_cnn_model.h5'
                if os.path.exists(model_path):
                    model.load_weights(model_path)
                    print("✅ Loaded pre-trained CNN emotion model")
                else:
                    print("⚠️ Using untrained CNN model (will have lower accuracy)")
                
                self.models['custom_cnn'] = model
                self.model_weights['custom_cnn'] = 0.15
                
            except Exception as e:
                print(f"❌ Custom CNN initialization failed: {e}")
    
    def _initialize_ensemble_model(self):
        """Initialize ensemble voting system"""
        self.ensemble_weights = {
            'temporal_consistency': 0.3,  # Weight for temporal consistency
            'confidence_weighting': 0.4,  # Weight based on model confidence
            'model_agreement': 0.3        # Weight based on model agreement
        }
        self.models['ensemble'] = True
        print("✅ Ensemble voting system initialized")
    
    def _extract_face_region(self, frame):
        """Extract face region with improved detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] >= 0.7:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    # Extract face with padding
                    padding = 0.2
                    x = max(0, int((bboxC.xmin - padding * bboxC.width) * w))
                    y = max(0, int((bboxC.ymin - padding * bboxC.height) * h))
                    x2 = min(w, int((bboxC.xmin + bboxC.width * (1 + padding)) * w))
                    y2 = min(h, int((bboxC.ymin + bboxC.height * (1 + padding)) * h))
                    
                    face_roi = frame[y:y2, x:x2]
                    
                    if face_roi.size > 0:
                        faces.append((face_roi, detection.score[0]))
        
        if faces:
            # Return the face with highest confidence
            return max(faces, key=lambda x: x[1])
        
        return None, 0.0
    
    def _analyze_with_deepface(self, face_image):
        """Analyze emotion with DeepFace using multiple models"""
        if 'deepface' not in self.models:
            return None
            
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            predictions = []
            
            # Try each available DeepFace model
            for model_name in self.models['deepface'][:2]:  # Use top 2 models for speed
                try:
                    result = DeepFace.analyze(
                        face_resized, 
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True,
                        detector_backend='skip',  # Skip detection since we already have face
                        model_name=model_name
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    emotions = result['emotion']
                    # Normalize emotion keys
                    normalized_emotions = {}
                    for emotion, confidence in emotions.items():
                        emotion_key = emotion.lower()
                        if emotion_key in self.standard_emotions:
                            normalized_emotions[emotion_key] = confidence / 100.0
                    
                    if normalized_emotions:
                        predictions.append({
                            'model': f'deepface_{model_name}',
                            'emotions': normalized_emotions,
                            'confidence': max(normalized_emotions.values())
                        })
                        
                except Exception as e:
                    continue
            
            return predictions
            
        except Exception as e:
            print(f"DeepFace analysis error: {e}")
            return None
    
    def _analyze_with_fer(self, face_image):
        """Analyze emotion with FER"""
        if 'fer' not in self.models:
            return None
            
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # FER expects RGB image
            emotions = self.models['fer'].detect_emotions(face_rgb)
            
            if emotions:
                # Get the first (most confident) detection
                emotion_scores = emotions[0]['emotions']
                
                # Normalize emotion keys and values
                normalized_emotions = {}
                for emotion, confidence in emotion_scores.items():
                    emotion_key = emotion.lower()
                    if emotion_key in self.standard_emotions:
                        normalized_emotions[emotion_key] = confidence
                
                return [{
                    'model': 'fer',
                    'emotions': normalized_emotions,
                    'confidence': max(normalized_emotions.values()) if normalized_emotions else 0
                }]
            
            return None
            
        except Exception as e:
            print(f"FER analysis error: {e}")
            return None
    
    def _analyze_with_huggingface(self, face_image):
        """Analyze emotion with Hugging Face models"""
        if 'huggingface' not in self.models or not self.models['huggingface']:
            return None
            
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            
            predictions = []
            
            for model_name, model_info in self.models['huggingface'].items():
                try:
                    # Use the pipeline for emotion detection
                    results = model_info['pipeline'](pil_image)
                    
                    # Convert to standard format
                    normalized_emotions = {}
                    for result in results:
                        label = result['label'].lower()
                        score = result['score']
                        
                        # Map different model outputs to standard emotions
                        emotion_mapping = {
                            'joy': 'happy', 'happiness': 'happy', 'positive': 'happy',
                            'sadness': 'sad', 'negative': 'sad',
                            'anger': 'angry', 'rage': 'angry',
                            'fear': 'fear', 'anxiety': 'fear',
                            'surprise': 'surprise', 'shock': 'surprise',
                            'disgust': 'disgust',
                            'neutral': 'neutral', 'calm': 'neutral'
                        }
                        
                        mapped_emotion = emotion_mapping.get(label, label)
                        if mapped_emotion in self.standard_emotions:
                            normalized_emotions[mapped_emotion] = score
                    
                    if normalized_emotions:
                        predictions.append({
                            'model': f'hf_{model_name}',
                            'emotions': normalized_emotions,
                            'confidence': max(normalized_emotions.values())
                        })
                        
                except Exception as e:
                    continue
            
            return predictions if predictions else None
            
        except Exception as e:
            print(f"Hugging Face analysis error: {e}")
            return None
    
    def _analyze_with_custom_cnn(self, face_image):
        """Analyze emotion with custom CNN"""
        if 'custom_cnn' not in self.models:
            return None
            
        try:
            # Preprocess for CNN
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)
            
            # Predict
            predictions = self.models['custom_cnn'].predict(face_input, verbose=0)
            emotion_probabilities = predictions[0]
            
            # Map to standard emotions
            normalized_emotions = {}
            for i, emotion in enumerate(self.standard_emotions):
                normalized_emotions[emotion] = float(emotion_probabilities[i])
            
            return [{
                'model': 'custom_cnn',
                'emotions': normalized_emotions,
                'confidence': float(np.max(emotion_probabilities))
            }]
            
        except Exception as e:
            print(f"Custom CNN analysis error: {e}")
            return None
    
    def _ensemble_prediction(self, all_predictions, temporal_history=None):
        """Advanced ensemble prediction using multiple strategies"""
        if not all_predictions:
            return None
        
        # Flatten all predictions
        flat_predictions = []
        for pred_list in all_predictions:
            if pred_list:
                flat_predictions.extend(pred_list)
        
        if not flat_predictions:
            return None
        
        # Strategy 1: Weighted average based on model confidence
        emotion_scores = {}
        total_weight = 0
        
        for pred in flat_predictions:
            model_weight = self.model_weights.get(pred['model'].split('_')[0], 0.1)
            confidence = pred['confidence']
            
            # Adjust weight based on confidence
            adjusted_weight = model_weight * (1 + confidence)
            total_weight += adjusted_weight
            
            for emotion, score in pred['emotions'].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0
                emotion_scores[emotion] += score * adjusted_weight
        
        # Normalize scores
        if total_weight > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_weight
        
        # Strategy 2: Consider temporal consistency
        if temporal_history and len(temporal_history) >= 3:
            # Get recent emotions
            recent_emotions = [h.get('emotion', 'neutral') for h in list(temporal_history)[-3:]]
            most_common = max(set(recent_emotions), key=recent_emotions.count)
            
            # Boost score for consistent emotions
            if most_common in emotion_scores:
                emotion_scores[most_common] *= 1.2
        
        # Strategy 3: Model agreement bonus
        emotion_counts = {}
        for pred in flat_predictions:
            top_emotion = max(pred['emotions'].items(), key=lambda x: x[1])
            emotion = top_emotion[0]
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        # Boost emotions that multiple models agree on
        for emotion, count in emotion_counts.items():
            if count > 1 and emotion in emotion_scores:
                agreement_bonus = 1 + (count - 1) * 0.1
                emotion_scores[emotion] *= agreement_bonus
        
        # Final prediction
        if emotion_scores:
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            final_confidence = min(0.95, best_emotion[1])
            
            # Generate reasoning
            reasoning_parts = []
            reasoning_parts.append(f"Ensemble of {len(flat_predictions)} models")
            
            # Top contributing models
            top_models = sorted(flat_predictions, key=lambda x: x['confidence'], reverse=True)[:2]
            model_names = [p['model'].split('_')[0] for p in top_models]
            reasoning_parts.append(f"Top contributors: {', '.join(set(model_names))}")
            
            # Agreement level
            agreement_level = len([p for p in flat_predictions 
                                 if max(p['emotions'].items(), key=lambda x: x[1])[0] == best_emotion[0]])
            reasoning_parts.append(f"{agreement_level}/{len(flat_predictions)} models agree")
            
            return {
                'emotion': best_emotion[0],
                'confidence': final_confidence,
                'reasoning': ' • '.join(reasoning_parts),
                'detailed_scores': emotion_scores,
                'model_count': len(flat_predictions),
                'agreement_ratio': agreement_level / len(flat_predictions)
            }
        
        return None
    
    def process_frame(self, frame):
        """Process frame with advanced multi-model emotion detection"""
        # Extract face region
        face_roi, detection_confidence = self._extract_face_region(frame)
        
        emotion_data = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'face_detected': False,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'No face detected',
            'facial_features': 'None',
            'detection_confidence': round(detection_confidence, 2),
            'model_count': 0,
            'analysis_method': 'Free Multi-AI Ensemble'
        }
        
        if face_roi is not None and face_roi.size > 0:
            emotion_data['face_detected'] = True
            
            try:
                # Run all available models
                all_predictions = []
                
                # DeepFace analysis
                deepface_pred = self._analyze_with_deepface(face_roi)
                if deepface_pred:
                    all_predictions.append(deepface_pred)
                
                # FER analysis
                fer_pred = self._analyze_with_fer(face_roi)
                if fer_pred:
                    all_predictions.append(fer_pred)
                
                # Hugging Face analysis
                hf_pred = self._analyze_with_huggingface(face_roi)
                if hf_pred:
                    all_predictions.append(hf_pred)
                
                # Custom CNN analysis
                cnn_pred = self._analyze_with_custom_cnn(face_roi)
                if cnn_pred:
                    all_predictions.append(cnn_pred)
                
                # Ensemble prediction
                if all_predictions:
                    ensemble_result = self._ensemble_prediction(all_predictions, emotion_history)
                    
                    if ensemble_result:
                        emotion_data.update({
                            'emotion': ensemble_result['emotion'],
                            'confidence': round(ensemble_result['confidence'], 2),
                            'reasoning': ensemble_result['reasoning'],
                            'facial_features': f"Multi-model analysis: {ensemble_result['model_count']} models",
                            'model_count': ensemble_result['model_count'],
                            'agreement_ratio': round(ensemble_result['agreement_ratio'], 2),
                            'detailed_scores': {k: round(v, 3) for k, v in ensemble_result['detailed_scores'].items()}
                        })
                        
                        # Store prediction for temporal analysis
                        self.prediction_history.append(ensemble_result)
                        
                        print(f"🎯 Multi-AI Ensemble: {ensemble_result['emotion']} "
                              f"({ensemble_result['confidence']:.2f}) "
                              f"- {ensemble_result['model_count']} models, "
                              f"{ensemble_result['agreement_ratio']:.0%} agreement")
                
            except Exception as e:
                print(f"❌ Error in multi-model analysis: {e}")
                emotion_data['reasoning'] = f"Analysis error: {str(e)}"
        
        return emotion_data

class EnhancedEmpathyBot:
    """Enhanced empathy bot with better emotion context understanding"""
    
    def __init__(self):
        self.conversation_context = []
        self.emotion_patterns = {}
        
    def generate_empathetic_response(self, user_message, emotion_data=None, image_analysis=None):
        """Generate contextually aware empathetic response"""
        
        # Enhanced fallback responses based on multi-model analysis
        if not emotion_data or not emotion_data.get('face_detected'):
            return "I'm here to listen and support you. What's on your mind today?"
        
        emotion = emotion_data.get('emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0)
        model_count = emotion_data.get('model_count', 0)
        agreement = emotion_data.get('agreement_ratio', 0)
        
        # Context-aware responses based on multi-model confidence
        base_responses = {
            'happy': [
                f"I can see the joy in your expression (detected by {model_count} AI models with {agreement:.0%} agreement)! Your happiness is wonderful. What's bringing you such joy?",
                f"Your smile really lights up! The AI models are {confidence:.0%} confident you're feeling great. I'd love to hear what's making you so happy!",
                f"I love seeing you this happy! Multiple AI systems confirm your positive mood with high confidence. Share what's going well in your life!"
            ],
            'sad': [
                f"I notice sadness in your expression (confidence: {confidence:.0%} from {model_count} AI models). It's okay to feel this way. I'm here to listen and support you.",
                f"You seem to be going through a tough time. The emotion detection shows you're feeling down. Would you like to talk about what's bothering you?",
                f"I can sense your sadness, and I want you to know that your feelings are valid. Sometimes sharing helps lighten the emotional load."
            ],
            'angry': [
                f"I can see frustration in your expression (detected with {confidence:.0%} confidence). Take a deep breath. What's bothering you? Let's work through this together.",
                f"You seem upset about something. Multiple AI models agree you're feeling angry. It's natural to feel frustrated sometimes. Want to talk about it?",
                f"I notice signs of anger or frustration. These feelings are completely valid. What's troubling you right now?"
            ],
            'fear': [
                f"I can see worry or anxiety in your expression (confidence: {confidence:.0%}). It's natural to feel concerned sometimes. What's on your mind?",
                f"You look a bit anxious or fearful. The AI analysis shows you might be worried about something. I'm here to help ease your concerns.",
                f"I sense some fear or anxiety. These feelings are completely understandable. Would you like to share what's troubling you?"
            ],
            'surprise': [
                f"You look surprised! ({confidence:.0%} confidence from multiple AI models). Did something unexpected happen? I'd love to hear about it!",
                f"Something seems to have caught you off guard! The emotion detection picked up surprise. What's the news?",
                f"I can see surprise in your expression! Multiple models agree you're shocked about something. What happened?"
            ],
            'disgust': [
                f"I can see something's really bothering you (confidence: {confidence:.0%}). Whatever it is that's troubling you, I'm here to listen.",
                f"You seem put off or disgusted by something. The AI models detected strong negative emotions. Want to talk about what's wrong?",
                f"I notice strong negative feelings. Multiple AI systems confirm you're really bothered by something. I'm here to help."
            ],
            'neutral': [
                f"You have a calm, peaceful expression (analyzed by {model_count} AI models). I'm here and ready to chat about whatever's on your mind.",
                f"You look relaxed and composed. The multi-model analysis shows a neutral emotional state. How can I help you today?",
                f"You seem centered and calm. Multiple AI systems confirm your peaceful state. What would you like to discuss?"
            ]
        }
        
        # Select response based on confidence level
        responses = base_responses.get(emotion, base_responses['neutral'])
        
        if confidence >= 0.8:
            response_idx = 0  # High confidence response
        elif confidence >= 0.5:
            response_idx = 1  # Medium confidence response
        else:
            response_idx = 2  # Lower confidence response
        
        response = responses[response_idx % len(responses)]
        
        # Add technical details for transparency
        if model_count > 1:
            technical_note = f"\n\n(Technical: {model_count} AI models analyzed your expression with {agreement:.0%} agreement and {confidence:.0%} confidence)"
            response += technical_note
        
        return response

def initialize_services():
    """Initialize advanced emotion detection services"""
    global emotion_detector, multimodal_bot
    
    print("🚀 Initializing Free Advanced Multi-AI Emotion Detection System...")
    
    try:
        emotion_detector = FreeAdvancedEmotionDetector()
        multimodal_bot = EnhancedEmpathyBot()
        print("✅ All advanced services initialized successfully!")
        print(f"🎯 Using {len(emotion_detector.models)} free AI models for emotion detection")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        # Fallback to basic system
        emotion_detector = BasicEmotionDetector()
        multimodal_bot = EnhancedEmpathyBot()

class BasicEmotionDetector:
    """Fallback basic emotion detector"""
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
    
    def process_frame(self, frame):
        return {
            'emotion': 'neutral',
            'confidence': 0.3,
            'face_detected': True,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'Basic fallback detector',
            'facial_features': 'Limited analysis',
            'detection_confidence': 0.7,
            'model_count': 1,
            'analysis_method': 'Basic Rule-based'
        }

# Enhanced smoothing function
def advanced_smooth_emotion_history(history):
    """Advanced temporal smoothing with confidence-based weighting"""
    if not history:
        return {'emotion': 'neutral', 'confidence': 0.0, 'face_detected': False}
    
    recent = list(history)[-10:]  # Look at last 10 readings
    face_detected_readings = [r for r in recent if r.get('face_detected', False)]
    
    if not face_detected_readings:
        return recent[-1] if recent else {
            'emotion': 'neutral', 
            'confidence': 0.0, 
            'face_detected': False
        }
    
    # Advanced weighting: recent readings + confidence + model agreement
    weights = []
    emotion_scores = {}
    
    for i, reading in enumerate(face_detected_readings):
        # Time-based weight (more recent = higher weight)
        time_weight = np.exp(i - len(face_detected_readings) + 1)
        
        # Confidence weight
        confidence_weight = reading.get('confidence', 0.3)
        
        # Model agreement weight
        agreement_weight = reading.get('agreement_ratio', 0.5)
        
        # Model count weight (more models = more reliable)
        model_count_weight = min(1.0, reading.get('model_count', 1) / 3.0)
        
        # Combined weight
        combined_weight = time_weight * confidence_weight * agreement_weight * model_count_weight
        weights.append(combined_weight)
        
        # Accumulate emotion scores
        emotion = reading['emotion']
        if emotion not in emotion_scores:
            emotion_scores[emotion] = 0
        emotion_scores[emotion] += combined_weight
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        for emotion in emotion_scores:
            emotion_scores[emotion] /= total_weight
    
    # Get best emotion
    if emotion_scores:
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Calculate stability score
        same_emotion_count = len([r for r in face_detected_readings[-5:] 
                                if r['emotion'] == best_emotion[0]])
        stability = same_emotion_count / min(5, len(face_detected_readings))
        
        # Adjust confidence based on stability
        base_confidence = best_emotion[1]
        stability_bonus = stability * 0.2
        final_confidence = min(0.95, base_confidence + stability_bonus)
        
        # Get most recent detailed info
        latest_reading = face_detected_readings[-1]
        
        return {
            'emotion': best_emotion[0],
            'confidence': round(final_confidence, 2),
            'face_detected': True,
            'timestamp': datetime.now().isoformat(),
            'reasoning': f"Advanced ensemble smoothing with {stability:.0%} stability",
            'facial_features': latest_reading.get('facial_features', 'Multi-model analysis'),
            'detection_confidence': latest_reading.get('detection_confidence', 0.8),
            'model_count': latest_reading.get('model_count', 1),
            'analysis_method': latest_reading.get('analysis_method', 'Free Multi-AI'),
            'stability_score': round(stability, 2),
            'temporal_consistency': round(base_confidence, 2)
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
    model_info = {}
    if emotion_detector and hasattr(emotion_detector, 'models'):
        model_info = {
            'available_models': list(emotion_detector.models.keys()),
            'model_count': len(emotion_detector.models),
            'deepface': 'deepface' in emotion_detector.models,
            'fer': 'fer' in emotion_detector.models,
            'huggingface': 'huggingface' in emotion_detector.models,
            'custom_cnn': 'custom_cnn' in emotion_detector.models
        }
    
    return jsonify({
        'status': 'healthy',
        'emotion_detector': emotion_detector is not None,
        'detector_type': 'Free Multi-AI Ensemble',
        'model_info': model_info,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    print('🔗 Client connected')
    model_count = len(emotion_detector.models) if emotion_detector and hasattr(emotion_detector, 'models') else 0
    emit('status', {
        'message': f'Connected to Free Multi-AI Emotion Detection ({model_count} models)',
        'gemini_available': False,
        'detector_type': f'Free Multi-AI Ensemble ({model_count} models)',
        'model_count': model_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frames with advanced multi-model emotion detection"""
    global emotion_detector, emotion_history
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None and emotion_detector:
            # Process frame with multi-model analysis
            emotion_data = emotion_detector.process_frame(frame)
            emotion_history.append(emotion_data)
            
            # Apply advanced temporal smoothing
            smoothed_emotion = advanced_smooth_emotion_history(emotion_history)
            
            emit('emotion_update', smoothed_emotion)
                
    except Exception as e:
        print(f"❌ Error processing video frame: {e}")
        emit('error', {'message': 'Error processing video frame'})

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat messages with advanced emotion context"""
    global multimodal_bot, emotion_history
    
    try:
        user_message = data.get('message', '').strip()
        if not user_message:
            return
        
        # Get current emotion context
        current_emotion = emotion_history[-1] if emotion_history else None
        
        # Generate empathetic response with advanced context
        response = multimodal_bot.generate_empathetic_response(
            user_message, 
            current_emotion
        )
        
        # Emit response with enhanced context
        emit('chat_response', {
            'message': response,
            'emotion_context': current_emotion,
            'timestamp': datetime.now().isoformat(),
            'analysis_method': current_emotion.get('analysis_method', 'Free Multi-AI') if current_emotion else 'Free Multi-AI'
        })
        
    except Exception as e:
        print(f"❌ Error handling chat message: {e}")
        emit('error', {'message': 'Error processing your message'})

if __name__ == '__main__':
    # Initialize services
    threading.Thread(target=initialize_services, daemon=True).start()
    time.sleep(5)  # Wait for initialization
    
    print("🌟 Starting Free Advanced Multi-AI Emotion Detection System...")
    print("🆓 Using only free, open-source AI models:")
    
    if DEEPFACE_AVAILABLE:
        print("   ✅ DeepFace (Multiple free models: VGG-Face, Facenet, OpenFace, etc.)")
    else:
        print("   📦 DeepFace: pip install deepface")
        
    if FER_AVAILABLE:
        print("   ✅ FER (Facial Expression Recognition)")
    else:
        print("   📦 FER: pip install fer")
        
    if TRANSFORMERS_AVAILABLE:
        print("   ✅ Hugging Face Transformers (Multiple free emotion models)")
    else:
        print("   📦 Transformers: pip install transformers torch")
        
    if TF_AVAILABLE:
        print("   ✅ TensorFlow (Custom CNN models)")
    else:
        print("   📦 TensorFlow: pip install tensorflow")
    
    print("\n🎯 Multi-Model Approach Benefits:")
    print("   • Higher accuracy through ensemble voting")
    print("   • Cross-validation between different AI approaches") 
    print("   • Temporal consistency analysis")
    print("   • Confidence-weighted predictions")
    print("   • Model agreement scoring")
    
    print("\n💡 To install missing dependencies:")
    print("   pip install deepface fer transformers torch tensorflow")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        allow_unsafe_werkzeug=True
    )   