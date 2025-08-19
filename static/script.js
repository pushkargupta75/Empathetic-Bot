class EnhancedEmpathyBot {
    constructor() {
        this.socket = io();
        this.isVideoActive = false;
        this.videoElement = document.getElementById('videoElement');
        this.canvasElement = document.getElementById('canvasElement');
        this.context = this.canvasElement.getContext('2d');
        this.lastEmotionUpdate = Date.now();
        this.emotionUpdateInterval = 1000; // Reduced frequency for Gemini API
        this.currentImage = null;
        this.geminiAvailable = false;
        this.analysisMethod = 'Unknown';
        
        this.init();
    }
    
    init() {
        this.setupSocketListeners();
        this.setupUIEventListeners();
        this.setupVideoStream();
        this.setupMultimodal();
        this.updateConnectionStatus();
    }
    
    setupSocketListeners() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ Connected to Enhanced Empathetic AI');
            this.updateConnectionStatus(true);
            this.showStatusMessage('Connected to Enhanced AI', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.updateConnectionStatus(false);
            this.showStatusMessage('Connection lost. Reconnecting...', 'error');
        });
        
        // Status and capabilities
        this.socket.on('status', (data) => {
            this.geminiAvailable = data.gemini_available;
            this.analysisMethod = data.detector_type || 'Unknown';
            this.updateGeminiStatus();
            this.updateAnalysisMethod();
            this.showStatusMessage(data.message, 'success');
        });
        
        // Enhanced emotion detection
        this.socket.on('emotion_update', (data) => {
            this.updateEnhancedEmotionDisplay(data);
        });
        
        // Chat responses with emotion context
        this.socket.on('chat_response', (data) => {
            this.addEnhancedMessage(data.message, 'assistant', {
                emotion_context: data.emotion_context,
                image_analysis: data.image_analysis,
                analysis_method: data.analysis_method
            });
            this.hideTypingIndicator();
        });
        
        // Image analysis results
        this.socket.on('image_analysis_result', (data) => {
            this.showImageAnalysis(data.analysis);
        });
        
        // Error handling
        this.socket.on('error', (data) => {
            console.error('Socket error:', data);
            this.showStatusMessage(data.message || 'An error occurred', 'error');
            this.hideTypingIndicator();
        });
    }
    
    setupUIEventListeners() {
        // Chat input
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const imageBtn = document.getElementById('imageBtn');
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        messageInput.addEventListener('input', (e) => {
            this.adjustTextareaHeight(e.target);
        });
        
        sendBtn.addEventListener('click', () => this.sendMessage());
        imageBtn.addEventListener('click', () => this.triggerImageUpload());
        
        // Camera controls
        document.getElementById('toggleCamera').addEventListener('click', () => {
            this.toggleVideoStream();
        });
        
        // Quick capture
        document.getElementById('quickCapture').addEventListener('click', () => {
            this.quickCapture();
        });
        
        // Image analysis buttons
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeCurrentImage();
        });
        
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearImagePreview();
        });
    }
    
    setupMultimodal() {
        const imageInput = document.getElementById('imageInput');
        const uploadArea = document.getElementById('imageUploadArea');
        
        // File input change
        imageInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.handleImageUpload(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files[0] && files[0].type.startsWith('image/')) {
                this.handleImageUpload(files[0]);
            }
        });
        
        uploadArea.addEventListener('click', () => {
            imageInput.click();
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
            
            this.videoElement.addEventListener('loadedmetadata', () => {
                this.startEmotionDetection();
            });
            
            this.showStatusMessage('Camera activated successfully', 'success');
            
        } catch (error) {
            console.error('Camera error:', error);
            this.showCameraError();
            this.showStatusMessage('Camera access denied or unavailable', 'error');
        }
    }
    
    startEmotionDetection() {
        if (!this.isVideoActive) return;
        
        const processFrame = () => {
            if (!this.isVideoActive || this.videoElement.paused || document.hidden) {
                requestAnimationFrame(processFrame);
                return;
            }
            
            const now = Date.now();
            if (now - this.lastEmotionUpdate < this.emotionUpdateInterval) {
                requestAnimationFrame(processFrame);
                return;
            }
            
            try {
                this.canvasElement.width = this.videoElement.videoWidth;
                this.canvasElement.height = this.videoElement.videoHeight;
                
                this.context.drawImage(this.videoElement, 0, 0);
                const imageData = this.canvasElement.toDataURL('image/jpeg', 0.7);
                
                this.socket.emit('video_frame', { image: imageData });
                this.lastEmotionUpdate = now;
                
            } catch (error) {
                console.error('Frame processing error:', error);
            }
            
            requestAnimationFrame(processFrame);
        };
        
        requestAnimationFrame(processFrame);
    }
    
    updateEnhancedEmotionDisplay(emotionData) {
        const elements = {
            emoji: document.getElementById('emotionEmoji'),
            label: document.getElementById('emotionLabel'),
            confidenceFill: document.getElementById('confidenceFill'),
            confidenceText: document.getElementById('confidenceText'),
            faceStatus: document.getElementById('faceStatus'),
            emotionContext: document.getElementById('emotionContext'),
            contextEmotion: document.getElementById('contextEmotion')
        };
        
        // Emotion emoji mapping with more variety
        const emojiMap = {
            'happy': ['😊', '😄', '😁', '🙂', '☺️'][Math.floor(Math.random() * 5)],
            'sad': ['😢', '😞', '😔', '🙁', '😟'][Math.floor(Math.random() * 5)],
            'angry': ['😠', '😡', '😤', '🤬', '👿'][Math.floor(Math.random() * 5)],
            'fear': ['😰', '😨', '😟', '😧', '🫨'][Math.floor(Math.random() * 5)],
            'surprise': ['😲', '😮', '🤯', '😯', '😦'][Math.floor(Math.random() * 5)],
            'disgust': ['🤢', '🤮', '😖', '😣', '🫤'][Math.floor(Math.random() * 5)],
            'neutral': ['😐', '😑', '🙂', '😌', '😶'][Math.floor(Math.random() * 5)]
        };
        
        const emotion = emotionData.emotion || 'neutral';
        const confidence = Math.round((emotionData.confidence || 0) * 100);
        const emoji = emojiMap[emotion] || '😐';
        
        // Enhanced animation for emotion changes
        elements.emoji.classList.add('emotion-transition');
        setTimeout(() => elements.emoji.classList.remove('emotion-transition'), 800);
        
        elements.emoji.textContent = emoji;
        elements.label.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        // Enhanced confidence display with color coding
        elements.confidenceFill.style.width = `${confidence}%`;
        elements.confidenceText.textContent = `${confidence}% confidence`;
        
        // Color coding based on confidence level
        const confidenceColors = {
            high: '#22c55e',     // Green for high confidence (70%+)
            medium: '#eab308',   // Yellow for medium confidence (40-70%)
            low: '#ef4444'       // Red for low confidence (<40%)
        };
        
        let confidenceColor = confidenceColors.low;
        if (confidence >= 70) confidenceColor = confidenceColors.high;
        else if (confidence >= 40) confidenceColor = confidenceColors.medium;
        
        elements.confidenceFill.style.background = `linear-gradient(90deg, ${confidenceColor}, ${confidenceColor}cc)`;
        
        // Enhanced face detection status with analysis method
        if (emotionData.face_detected) {
            elements.faceStatus.classList.add('detected');
            const analysisMethod = emotionData.analysis_method || this.analysisMethod;
            elements.faceStatus.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <span>Face detected • ${analysisMethod}</span>
            `;
            
            // Show reasoning if available (for Gemini)
            if (emotionData.reasoning) {
                elements.faceStatus.title = `Reasoning: ${emotionData.reasoning}`;
            }
        } else {
            elements.faceStatus.classList.remove('detected');
            elements.faceStatus.innerHTML = '<i class="fas fa-search"></i><span>Looking for face...</span>';
            elements.faceStatus.title = '';
        }
        
        // Enhanced chat context display
        if (emotionData.face_detected && emotion !== 'neutral' && confidence >= 30) {
            elements.emotionContext.style.display = 'flex';
            elements.contextEmotion.textContent = emotion;
            
            // Add confidence level to context
            const contextLabel = elements.emotionContext.querySelector('.context-label');
            if (contextLabel) {
                contextLabel.textContent = `Current mood (${confidence}% confident):`;
            }
        } else {
            elements.emotionContext.style.display = 'none';
        }
        
        // Emotion-specific color themes
        const emotionColors = {
            'happy': '#22c55e',
            'sad': '#6366f1',
            'angry': '#ef4444',
            'fear': '#8b5cf6',
            'surprise': '#f59e0b',
            'disgust': '#84cc16',
            'neutral': '#10b981'
        };
        
        const color = emotionColors[emotion] || '#10b981';
        
        // Update UI theme based on emotion (subtle effect)
        document.documentElement.style.setProperty('--emotion-accent', color);
        
        // Log detailed emotion data for debugging
        if (confidence > 50) {
            console.log(`🎯 ${analysisMethod || this.analysisMethod} detected: ${emotion} (${confidence}% confident)`);
            if (emotionData.reasoning) {
                console.log(`💭 Reasoning: ${emotionData.reasoning}`);
            }
        }
    }
    
    sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message && !this.currentImage) return;
        
        // Add user message to chat
        if (message) {
            this.addEnhancedMessage(message, 'user');
        }
        
        // Show enhanced typing indicator
        this.showEnhancedTypingIndicator();
        
        // Prepare message data
        const messageData = { message: message || 'Analyze this image' };
        if (this.currentImage) {
            messageData.image = this.currentImage;
            messageData.image_prompt = message || 'Analyze this image and describe what you see in detail';
        }
        
        // Send message to server
        this.socket.emit('chat_message', messageData);
        
        // Clear input and image
        messageInput.value = '';
        this.adjustTextareaHeight(messageInput);
        if (this.currentImage) {
            this.clearImagePreview();
        }
    }
    
    addEnhancedMessage(content, sender, context = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message fade-in`;
        
        const avatar = sender === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';
        
        const timestamp = this.formatTimestamp(new Date());
        
        // Enhanced emotion context display
        let emotionInfo = '';
        if (sender === 'assistant' && context?.emotion_context && context.emotion_context.emotion !== 'neutral') {
            const emotion = context.emotion_context.emotion;
            const confidence = Math.round((context.emotion_context.confidence || 0) * 100);
            const method = context.analysis_method || 'Unknown';
            
            const emotionEmojis = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 
                'fear': '😰', 'surprise': '😲', 'disgust': '🤢'
            };
            
            emotionInfo = `
                <div class="emotion-context" style="margin-top: 0.5rem; display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2);">
                    ${emotionEmojis[emotion] || '💭'}
                    <span style="font-size: 0.875rem;">Responding to your <strong>${emotion}</strong> mood (${confidence}% confident, detected by ${method})</span>
                </div>
            `;
        }
        
        // Enhanced image analysis display
        let imageAnalysisInfo = '';
        if (context?.image_analysis) {
            imageAnalysisInfo = `
                <div class="image-analysis" style="margin-top: 0.75rem; padding: 1rem; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 10px;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; font-weight: 600; color: #8b5cf6;">
                        <i class="fas fa-eye"></i>
                        <span>Vision Analysis</span>
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.5;">${this.formatMessage(context.image_analysis)}</div>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(content)}</div>
                ${emotionInfo}
                ${imageAnalysisInfo}
                <div class="message-time">${timestamp}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    showEnhancedTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        const method = this.analysisMethod || 'AI';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="margin-left: 0.5rem;">${method} is analyzing and responding...</span>
        `;
        indicator.style.display = 'flex';
    }
    
    updateAnalysisMethod() {
        // Update analysis method indicator in the UI
        const methodIndicator = document.querySelector('.analysis-method-indicator');
        if (methodIndicator) {
            methodIndicator.textContent = this.analysisMethod;
        }
        
        // Update emotion detection interval based on method
        if (this.analysisMethod.includes('Gemini')) {
            this.emotionUpdateInterval = 2000; // 2 seconds for Gemini to avoid quota issues
        } else {
            this.emotionUpdateInterval = 500; // 0.5 seconds for rule-based
        }
    }
    
    handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            this.showStatusMessage('Please select a valid image file', 'error');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showStatusMessage('Image size must be less than 10MB', 'error');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = e.target.result;
            this.showImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }
    
    showImagePreview(imageData) {
        const previewContainer = document.getElementById('imagePreview');
        const previewImage = document.getElementById('previewImage');
        
        previewImage.src = imageData;
        previewContainer.style.display = 'block';
        previewContainer.classList.add('fade-in');
    }
    
    clearImagePreview() {
        const previewContainer = document.getElementById('imagePreview');
        previewContainer.style.display = 'none';
        this.currentImage = null;
    }
    
    quickCapture() {
        if (!this.isVideoActive) {
            this.showStatusMessage('Camera not available', 'error');
            return;
        }
        
        try {
            this.canvasElement.width = this.videoElement.videoWidth;
            this.canvasElement.height = this.videoElement.videoHeight;
            this.context.drawImage(this.videoElement, 0, 0);
            
            const capturedImage = this.canvasElement.toDataURL('image/jpeg', 0.8);
            this.currentImage = capturedImage;
            this.showImagePreview(capturedImage);
            
            this.showStatusMessage('Image captured successfully!', 'success');
        } catch (error) {
            console.error('Capture error:', error);
            this.showStatusMessage('Failed to capture image', 'error');
        }
    }
    
    analyzeCurrentImage() {
        if (!this.currentImage) {
            this.showStatusMessage('No image to analyze', 'error');
            return;
        }
        
        this.socket.emit('analyze_image', {
            image: this.currentImage,
            prompt: 'Analyze this image in detail and describe everything you see'
        });
        
        this.showStatusMessage('Analyzing image with AI...', 'info');
    }
    
    showImageAnalysis(analysis) {
        this.addEnhancedMessage('🔍 Image Analysis Result:', 'assistant', { 
            image_analysis: analysis 
        });
    }
    
    triggerImageUpload() {
        document.getElementById('imageInput').click();
    }
    
    formatMessage(message) {
        return message
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }
    
    formatTimestamp(date) {
        return date.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        indicator.style.display = 'none';
    }
    
    toggleVideoStream() {
        const toggleBtn = document.getElementById('toggleCamera');
        
        if (this.isVideoActive) {
            const stream = this.videoElement.srcObject;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            this.videoElement.srcObject = null;
            this.isVideoActive = false;
            toggleBtn.innerHTML = '<i class="fas fa-video-slash"></i>';
            this.showStatusMessage('Camera deactivated', 'info');
        } else {
            this.setupVideoStream();
            toggleBtn.innerHTML = '<i class="fas fa-video"></i>';
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
    
    updateGeminiStatus() {
        const geminiStatus = document.getElementById('geminiStatus');
        const geminiText = document.getElementById('geminiText');
        
        if (this.geminiAvailable) {
            geminiStatus.style.borderColor = 'var(--success)';
            geminiStatus.style.background = 'rgba(16, 185, 129, 0.1)';
            geminiText.innerHTML = '<i class="fas fa-brain"></i> Gemini AI Active';
        } else {
            geminiStatus.style.borderColor = 'var(--warning)';
            geminiStatus.style.background = 'rgba(234, 179, 8, 0.1)';
            geminiText.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Gemini Offline';
        }
    }
    
    showStatusMessage(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Create enhanced toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.style.cssText = `
            position: fixed;
            top: 100px;
            right: 2rem;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            color: white;
            font-size: 0.875rem;
            font-weight: 500;
            z-index: 2000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            max-width: 350px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        `;
        
        const colors = {
            success: '#22c55e',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#10b981'
        };
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        
        toast.style.background = `linear-gradient(135deg, ${colors[type] || colors.info}, ${colors[type] || colors.info}dd)`;
        toast.innerHTML = `${icons[type] || icons.info} ${message}`;
        
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        }, 100);
        
        // Remove after delay
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 4000);
    }
    
    showCameraError() {
        const videoContainer = this.videoElement.parentElement;
        const errorDiv = document.createElement('div');
        errorDiv.className = 'camera-error';
        errorDiv.innerHTML = `
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: var(--text-muted);
                z-index: 10;
                padding: 2rem;
            ">
                <i class="fas fa-video-slash" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.7;"></i>
                <h4 style="margin-bottom: 0.5rem;">Camera Access Required</h4>
                <p style="margin: 0.5rem 0; opacity: 0.8;">Please allow camera access to enable ${this.analysisMethod} emotion detection</p>
                <button onclick="location.reload()" style="
                    background: var(--primary);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 25px;
                    cursor: pointer;
                    margin-top: 1rem;
                    font-weight: 600;
                    transition: all 0.3s ease;
                ">
                    <i class="fas fa-redo"></i> Try Again
                </button>
            </div>
        `;
        
        videoContainer.appendChild(errorDiv);
    }
    
    adjustTextareaHeight(textarea) {
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120);
        textarea.style.height = newHeight + 'px';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Enhanced Empathetic AI with Gemini Vision...');
    new EnhancedEmpathyBot();
});

// Handle page visibility for performance optimization
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('📱 Page hidden - optimizing performance');
    } else {
        console.log('📱 Page visible - resuming full operation');
    }
});

// Enhanced keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const messageInput = document.getElementById('messageInput');
        if (document.activeElement === messageInput) {
            e.preventDefault();
            document.getElementById('sendBtn').click();
        }
    }
    
    // Escape to clear image preview
    if (e.key === 'Escape') {
        const previewContainer = document.getElementById('imagePreview');
        if (previewContainer.style.display !== 'none') {
            document.getElementById('clearBtn').click();
        }
    }
    
    // Ctrl/Cmd + I to trigger image upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'i') {
        e.preventDefault();
        document.getElementById('imageBtn').click();
    }
    
    // Ctrl/Cmd + Space to capture image
    if ((e.ctrlKey || e.metaKey) && e.key === ' ') {
        e.preventDefault();
        document.getElementById('quickCapture').click();
    }
});

// Add CSS for emotion-based theming
const style = document.createElement('style');
style.textContent = `
    :root {
        --emotion-accent: #10b981;
    }
    
    .emotion-transition {
        animation: emotionPulse 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes emotionPulse {
        0% {
            transform: scale(1);
        }
        30% {
            transform: scale(1.15);
            filter: drop-shadow(0 0 20px var(--emotion-accent));
        }
        100% {
            transform: scale(1);
        }
    }
    
    .confidence-fill {
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .face-status.detected {
        animation: detectPulse 0.6s ease-out;
    }
    
    @keyframes detectPulse {
        0% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
        }
        50% {
            transform: scale(1.02);
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
`;
document.head.appendChild(style);