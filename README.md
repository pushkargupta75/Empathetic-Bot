# Empathetic-Bot

A real-time multimodal chatbot that reads facial emotions through webcam feed and provides empathetic responses using advanced language models. Built with Flask, OpenCV, and Hugging Face Transformers.

## 🚀 Features

### Core Functionality
- **Real-time Emotion Detection**: Uses webcam feed to detect facial emotions with confidence scores
- **Empathetic Conversations**: LLM responses are contextually aware of user's emotional state
- **Professional Dark Theme**: Beautiful UI with emerald accents and professional styling
- **WebSocket Integration**: Real-time communication for smooth video processing and chat
- **Temporal Smoothing**: Reduces emotion detection jitter for stable readings
- **Privacy-Focused**: Processes video frames in memory without storing personal data

### Supported Emotions
- Happy 😊
- Sad 😢  
- Angry 😠
- Fearful 😰
- Surprised 😲
- Disgusted 🤢
- Neutral 😐

### Technical Features
- Responsive design for desktop and mobile
- Face detection using MediaPipe
- Emotion context integration in conversations
- Connection status monitoring
- Camera controls and error handling

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- Webcam access
- Modern web browser with WebRTC support

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd empathetic-ai-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
Navigate to `http://localhost:5000`

### Docker Setup

1. **Build the image**
```bash
docker build -t empathetic-ai .
```

2. **Run the container**
```bash
docker run -p 5000:5000 empathetic-ai
```

## 📱 Usage

1. **Grant Camera Permission**: Allow camera access when prompted
2. **Start Chatting**: Type your message in the chat input
3. **Emotion Detection**: Your facial emotion will be detected and displayed
4. **Empathetic Responses**: AI responds based on your emotional state
5. **Real-time Updates**: Emotion detection runs continuously while chatting


## 🔒 Privacy & Ethics

- **No Data Storage**: Video frames are processed in memory only
- **Explicit Consent**: Clear camera permission requests
- **Local Processing**: Emotion detection happens client-side when possible
- **Transparent AI**: Shows confidence levels and reasoning
- **Bias Awareness**: Acknowledges limitations of emotion detection models

## 🚀 Deployment Options

### Production Deployment
- **Hugging Face Spaces**: Easy demo hosting with GPU support
- **Google Cloud Run**: Serverless container deployment
- **AWS ECS/Fargate**: Scalable container orchestration
- **Azure Container Instances**: Simple container deployment

### Environment Variables
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key
PORT=5000
```

## 🧪 Development


### Key Components
- **EmotionDetector**: Handles video processing and face detection
- **EmpathyBot**: Generates contextual responses using LLMs
- **WebSocket Handlers**: Real-time communication for video and chat
- **Frontend Controller**: Manages video stream and user interactions

## 📊 Performance Metrics

- **Emotion Detection**: ~500ms processing time per frame
- **Response Generation**: ~1-3 seconds depending on model size
- **Memory Usage**: ~500MB baseline + model overhead
- **Supported Concurrent Users**: 10-50 depending on deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🆘 Troubleshooting

### Common Issues

**Camera Access Denied**
- Check browser permissions in settings
- Ensure HTTPS for production deployments
- Try refreshing and re-granting permissions

**Model Loading Errors**
- Check internet connection for model downloads
- Verify sufficient disk space (2-3GB for models)
- Try clearing pip cache: `pip cache purge`

**Performance Issues**
- Reduce emotion detection frequency (increase interval)
- Use smaller language models
- Enable GPU acceleration if available

### Support
For issues and questions, please open an issue on the GitHub repository.
