class EmpathyChatBot {
    constructor() {
        this.socket = io();
        this.isVideoActive = false;
        this.videoElement = document.getElementById('videoElement');
        this.canvasElement = document.getElementById('canvasElement');
        this.context = this.canvasElement.getContext('2d');
        this.lastEmotionUpdate = Date.now();
        this.emotionUpdateInterval = 500; // Update emotion every 500ms
        
        this.init();
    }
    
    init() {
        this.setupSocketListeners();
        this.setupUIEventListeners();
        this.setupVideoStream();
        this.updateConnectionStatus();
    }
    
    setupSocketListeners() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
            this.showStatusMessage('Connected to Empathetic AI', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
            this.showStatusMessage('Connection lost. Reconnecting...', 'error');
        });
        
        // Emotion detection responses
        this.socket.on('emotion_update', (data) => {
            this.updateEmotionDisplay(data);
        });
        
        // Chat responses
        this.socket.on('chat_response', (data) => {
            this.addMessage(data.message, 'assistant', data.emotion_context);
            this.hideTypingIndicator();
        });
        
        // Error handling
        this.socket.on('error', (data) => {
            console.error('Socket error:', data);
            this.showStatusMessage(data.message || 'An error occurred', 'error');
            this.hideTypingIndicator();
        });
        
        // Status updates
        this.socket.on('status', (data) => {
            this.showStatusMessage(data.message, 'info');
        });
    }
    
    setupUIEventListeners() {
        // Chat input handling
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        sendButton.addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Camera toggle
        const toggleCamera = document.getElementById('toggleCamera');
        toggleCamera.addEventListener('click', () => {
            this.toggleVideoStream();
        });
        
        // Auto-resize message input
        messageInput.addEventListener('input', (e) => {
            this.adjustInputHeight(e.target);
        });
    }
    
    async setupVideoStream() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            this.videoElement.srcObject = stream;
            this.isVideoActive = true;
            
            // Start emotion detection after video loads
            this.videoElement.addEventListener('loadedmetadata', () => {
                this.startEmotionDetection();
            });
            
            this.showStatusMessage('Camera activated successfully', 'success');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showCameraError();
            this.showStatusMessage('Camera access denied or unavailable', 'error');
        }
    }
    
    startEmotionDetection() {
        if (!this.isVideoActive) return;
        
        const processFrame = () => {
            if (!this.isVideoActive || this.videoElement.paused) return;
            
            const now = Date.now();
            if (now - this.lastEmotionUpdate < this.emotionUpdateInterval) {
                requestAnimationFrame(processFrame);
                return;
            }
            
            try {
                // Set canvas size to match video
                this.canvasElement.width = this.videoElement.videoWidth;
                this.canvasElement.height = this.videoElement.videoHeight;
                
                // Draw current video frame to canvas
                this.context.drawImage(this.videoElement, 0, 0);
                
                // Convert canvas to base64 image data
                const imageData = this.canvasElement.toDataURL('image/jpeg', 0.7);
                
                // Send frame to server for emotion detection
                this.socket.emit('video_frame', { image: imageData });
                
                this.lastEmotionUpdate = now;
                
            } catch (error) {
                console.error('Error processing video frame:', error);
            }
            
            requestAnimationFrame(processFrame);
        };
        
        requestAnimationFrame(processFrame);
    }
    
    updateEmotionDisplay(emotionData) {
        const emotionEmoji = document.getElementById('emotionEmoji');
        const emotionLabel = document.getElementById('emotionLabel');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceText = document.getElementById('confidenceText');
        const faceStatus = document.getElementById('faceStatus');
        const emotionContext = document.getElementById('emotionContext');
        const contextEmotion = document.getElementById('contextEmotion');
        
        // Emotion emoji mapping
        const emojiMap = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fearful': '😰',
            'surprised': '😲',
            'disgusted': '🤢',
            'neutral': '😐'
        };
        
        // Update emotion display with animation
        const emoji = emojiMap[emotionData.emotion] || '😐';
        const confidence = Math.round((emotionData.confidence || 0) * 100);
        
        // Add transition animation
        emotionEmoji.classList.add('emotion-transition');
        setTimeout(() => emotionEmoji.classList.remove('emotion-transition'), 600);
        
        emotionEmoji.textContent = emoji;
        emotionLabel.textContent = emotionData.emotion || 'neutral';
        confidenceFill.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence}%`;
        
        // Update face detection status
        if (emotionData.face_detected) {
            faceStatus.classList.add('detected');
            faceStatus.innerHTML = '<span class="status-icon">✓</span><span>Face detected</span>';
        } else {
            faceStatus.classList.remove('detected');
            faceStatus.innerHTML = '<span class="status-icon">👤</span><span>Looking for face...</span>';
        }
        
        // Update chat context
        if (emotionData.face_detected && emotionData.emotion !== 'neutral') {
            emotionContext.style.display = 'block';
            contextEmotion.textContent = emotionData.emotion;
        } else {
            emotionContext.style.display = 'none';
        }
        
        // Color code confidence bar based on emotion
        const emotionColors = {
            'happy': '#22c55e',
            'sad': '#3b82f6',
            'angry': '#ef4444',
            'fearful': '#8b5cf6',
            'surprised': '#f59e0b',
            'disgusted': '#6b7280',
            'neutral': '#10b981'
        };
        
        const emotionColor = emotionColors[emotionData.emotion] || '#10b981';
        confidenceFill.style.background = `linear-gradient(90deg, ${emotionColor}, ${emotionColor}cc)`;
    }
    
    sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        messageInput.value = '';
        this.adjustInputHeight(messageInput);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Send message to server
        this.socket.emit('chat_message', { message: message });
    }
    
    addMessage(content, sender, emotionContext = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        const timestamp = this.formatTimestamp(new Date());
        
        let emotionInfo = '';
        if (sender === 'assistant' && emotionContext && emotionContext.emotion !== 'neutral') {
            const confidence = Math.round((emotionContext.confidence || 0) * 100);
            emotionInfo = `<div class="emotion-info">Responding to your ${emotionContext.emotion} mood (${confidence}% confidence)</div>`;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(content)}</div>
                ${emotionInfo}
                <div class="message-time">${timestamp}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    formatMessage(message) {
        // Basic message formatting (can be extended)
        return message.replace(/\n/g, '<br>');
    }
    
    formatTimestamp(date) {
        return date.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    showTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.style.display = 'flex';
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.style.display = 'none';
    }
    
    toggleVideoStream() {
        const toggleButton = document.getElementById('toggleCamera');
        
        if (this.isVideoActive) {
            // Stop video stream
            const stream = this.videoElement.srcObject;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            this.videoElement.srcObject = null;
            this.isVideoActive = false;
            toggleButton.innerHTML = '<span class="camera-icon">📷</span>';
            this.showStatusMessage('Camera deactivated', 'info');
        } else {
            // Restart video stream
            this.setupVideoStream();
            toggleButton.innerHTML = '<span class="camera-icon">⏸️</span>';
        }
    }
    
    updateConnectionStatus(connected = false) {
        const statusDot = document.getElementById('connectionStatus');
        const statusText = document.getElementById('statusText');
        
        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Connecting...';
        }
    }
    
    showStatusMessage(message, type = 'info') {
        // Create status message (could be implemented as toast notifications)
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Optionally show as temporary UI element
        if (type === 'error') {
            // Could implement error toast here
        }
    }
    
    showCameraError() {
        const videoContainer = this.videoElement.parentElement;
        const errorDiv = document.createElement('div');
        errorDiv.className = 'camera-error';
        errorDiv.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                <h4>Camera Access Required</h4>
                <p>Please allow camera access to enable emotion detection.</p>
                <button onclick="location.reload()" style="
                    background: var(--accent-primary);
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: var(--radius-md);
                    cursor: pointer;
                    margin-top: 1rem;
                ">Retry</button>
            </div>
        `;
        
        videoContainer.appendChild(errorDiv);
    }
    
    adjustInputHeight(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EmpathyChatBot();
});

// Handle page visibility changes to pause/resume video processing
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden - pausing video processing');
    } else {
        console.log('Page visible - resuming video processing');
    }
});