class AdvancedMultiAIEmpathyBot {
    constructor() {
        this.socket = io();
        this.isVideoActive = false;
        this.videoElement = document.getElementById('videoElement');
        this.canvasElement = document.getElementById('canvasElement');
        this.context = this.canvasElement.getContext('2d');
        this.lastEmotionUpdate = Date.now();
        this.emotionUpdateInterval = 500; // Faster updates for multi-AI system
        this.currentImage = null;
        this.modelCount = 0;
        this.analysisMethod = 'Free Multi-AI Ensemble';
        this.emotionTrends = [];
        
        this.init();
    }
    
    init() {
        this.setupSocketListeners();
        this.setupUIEventListeners();
        this.setupVideoStream();
        this.setupAdvancedUI();
        this.updateConnectionStatus();
    }
    
    setupSocketListeners() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ Connected to Advanced Multi-AI Emotion Detection');
            this.updateConnectionStatus(true);
            this.showStatusMessage('Connected to Multi-AI System', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.updateConnectionStatus(false);
            this.showStatusMessage('Connection lost. Reconnecting...', 'error');
        });
        
        // Enhanced status with model information
        this.socket.on('status', (data) => {
            this.modelCount = data.model_count || 0;
            this.analysisMethod = data.detector_type || 'Multi-AI Ensemble';
            this.updateModelInfo();
            this.showStatusMessage(data.message, 'success');
        });
        
        // Advanced emotion detection with multi-model data
        this.socket.on('emotion_update', (data) => {
            this.updateAdvancedEmotionDisplay(data);
            this.trackEmotionTrends(data);
        });
        
        // Enhanced chat responses
        this.socket.on('chat_response', (data) => {
            this.addAdvancedMessage(data.message, 'assistant', {
                emotion_context: data.emotion_context,
                analysis_method: data.analysis_method
            });
            this.hideTypingIndicator();
        });
        
        // Error handling
        this.socket.on('error', (data) => {
            console.error('Socket error:', data);
            this.showStatusMessage(data.message || 'An error occurred', 'error');
            this.hideTypingIndicator();
        });
    }
    
    setupAdvancedUI() {
        // Add model information panel
        this.createModelInfoPanel();
        // Add emotion trends chart
        this.createEmotionTrendsChart();
        // Add confidence meter enhancements
        this.enhanceConfidenceMeter();
    }
    
    createModelInfoPanel() {
        const videoSection = document.querySelector('.video-section');
        if (!videoSection) return;
        
        const modelPanel = document.createElement('div');
        modelPanel.className = 'model-info-panel';
        modelPanel.innerHTML = `
            <div class="panel-header">
                <h4><i class="fas fa-brain"></i> AI Models</h4>
            </div>
            <div class="model-stats" id="modelStats">
                <div class="stat">
                    <span class="stat-label">Active Models:</span>
                    <span class="stat-value" id="modelCount">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Analysis Method:</span>
                    <span class="stat-value" id="analysisMethod">Loading...</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Agreement:</span>
                    <span class="stat-value" id="modelAgreement">--</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Stability:</span>
                    <span class="stat-value" id="emotionStability">--</span>
                </div>
            </div>
            <div class="model-indicators" id="modelIndicators">
                <!-- Dynamic model indicators will be added here -->
            </div>
        `;
        
        videoSection.appendChild(modelPanel);
    }
    
    createEmotionTrendsChart() {
        const rightPanel = document.querySelector('.right-panel');
        if (!rightPanel) return;
        
        const trendsPanel = document.createElement('div');
        trendsPanel.className = 'emotion-trends-panel';
        trendsPanel.innerHTML = `
            <div class="panel-header">
                <h4><i class="fas fa-chart-line"></i> Emotion Trends</h4>
                <button class="toggle-btn" id="toggleTrends">
                    <i class="fas fa-chevron-up"></i>
                </button>
            </div>
            <div class="trends-content" id="trendsContent">
                <canvas id="emotionChart" width="300" height="120"></canvas>
                <div class="trend-stats" id="trendStats">
                    <div class="trend-stat">
                        <span>Dominant:</span>
                        <span id="dominantEmotion">neutral</span>
                    </div>
                    <div class="trend-stat">
                        <span>Avg Confidence:</span>
                        <span id="avgConfidence">0%</span>
                    </div>
                </div>
            </div>
        `;
        
        rightPanel.insertBefore(trendsPanel, rightPanel.firstChild);
        this.initEmotionChart();
    }
    
    initEmotionChart() {
        const canvas = document.getElementById('emotionChart');
        if (!canvas) return;
        
        this.chartCtx = canvas.getContext('2d');
        this.emotionColors = {
            'happy': '#22c55e',
            'sad': '#3b82f6', 
            'angry': '#ef4444',
            'fear': '#8b5cf6',
            'surprise': '#f59e0b',
            'disgust': '#84cc16',
            'neutral': '#6b7280'
        };
        
        this.drawEmotionChart();
    }
    
    drawEmotionChart() {
        if (!this.chartCtx || this.emotionTrends.length === 0) return;
        
        const ctx = this.chartCtx;
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;
        
        // Horizontal lines
        for (let i = 0; i <= 4; i++) {
            const y = (height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw emotion trend lines
        const maxDataPoints = 50;
        const recentTrends = this.emotionTrends.slice(-maxDataPoints);
        
        if (recentTrends.length > 1) {
            const stepX = width / (maxDataPoints - 1);
            
            // Group by emotion
            const emotionData = {};
            recentTrends.forEach((trend, index) => {
                const emotion = trend.emotion;
                if (!emotionData[emotion]) {
                    emotionData[emotion] = [];
                }
                emotionData[emotion].push({
                    x: (index / (recentTrends.length - 1)) * width,
                    y: height - (trend.confidence * height),
                    confidence: trend.confidence
                });
            });
            
            // Draw lines for each emotion
            Object.entries(emotionData).forEach(([emotion, points]) => {
                if (points.length > 1) {
                    ctx.strokeStyle = this.emotionColors[emotion] || '#6b7280';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    points.forEach((point, index) => {
                        if (index === 0) {
                            ctx.moveTo(point.x, point.y);
                        } else {
                            ctx.lineTo(point.x, point.y);
                        }
                    });
                    
                    ctx.stroke();
                }
            });
        }
    }
    
    trackEmotionTrends(emotionData) {
        if (!emotionData.face_detected) return;
        
        this.emotionTrends.push({
            emotion: emotionData.emotion,
            confidence: emotionData.confidence,
            timestamp: Date.now(),
            modelCount: emotionData.model_count || 1,
            agreement: emotionData.agreement_ratio || 0
        });
        
        // Keep only last 100 data points
        if (this.emotionTrends.length > 100) {
            this.emotionTrends = this.emotionTrends.slice(-100);
        }
        
        // Update chart and stats
        this.drawEmotionChart();
        this.updateTrendStats();
    }
    
    updateTrendStats() {
        const recent = this.emotionTrends.slice(-20); // Last 20 readings
        
        if (recent.length === 0) return;
        
        // Find dominant emotion
        const emotionCounts = {};
        let totalConfidence = 0;
        
        recent.forEach(trend => {
            emotionCounts[trend.emotion] = (emotionCounts[trend.emotion] || 0) + 1;
            totalConfidence += trend.confidence;
        });
        
        const dominantEmotion = Object.entries(emotionCounts)
            .sort(([,a], [,b]) => b - a)[0]?.[0] || 'neutral';
        
        const avgConfidence = (totalConfidence / recent.length) * 100;
        
        // Update UI
        const dominantEl = document.getElementById('dominantEmotion');
        const avgConfEl = document.getElementById('avgConfidence');
        
        if (dominantEl) dominantEl.textContent = dominantEmotion;
        if (avgConfEl) avgConfEl.textContent = `${avgConfidence.toFixed(0)}%`;
    }
    
    updateAdvancedEmotionDisplay(emotionData) {
        // Update basic emotion display
        this.updateBasicEmotionDisplay(emotionData);
        
        // Update advanced multi-AI specific information
        this.updateModelAgreement(emotionData.agreement_ratio);
        this.updateStabilityScore(emotionData.stability_score);
        this.updateModelCount(emotionData.model_count);
        
        // Update detailed scores if available
        if (emotionData.detailed_scores) {
            this.updateDetailedScores(emotionData.detailed_scores);
        }
        
        // Log advanced info
        if (emotionData.confidence > 0.5) {
            console.log(`🎯 Multi-AI Detection: ${emotionData.emotion} ` +
                       `(${(emotionData.confidence * 100).toFixed(0)}% confidence, ` +
                       `${emotionData.model_count} models, ` +
                       `${(emotionData.agreement_ratio * 100).toFixed(0)}% agreement)`);
        }
    }
    
    updateBasicEmotionDisplay(emotionData) {
        const elements = {
            emoji: document.getElementById('emotionEmoji'),
            label: document.getElementById('emotionLabel'),
            confidenceFill: document.getElementById('confidenceFill'),
            confidenceText: document.getElementById('confidenceText'),
            faceStatus: document.getElementById('faceStatus'),
            emotionContext: document.getElementById('emotionContext'),
            contextEmotion: document.getElementById('contextEmotion')
        };
        
        // Enhanced emotion emojis with more variety
        const emojiSets = {
            'happy': ['😊', '😄', '😁', '🙂', '☺️', '😆', '🤗'],
            'sad': ['😢', '😞', '😔', '🙁', '😟', '😪', '💔'],
            'angry': ['😠', '😡', '😤', '🤬', '👿', '😾', '🗯️'],
            'fear': ['😰', '😨', '😟', '😧', '🫨', '😳', '🙀'],
            'surprise': ['😲', '😮', '🤯', '😯', '😦', '🫢', '😵'],
            'disgust': ['🤢', '🤮', '😖', '😣', '🫤', '😤', '🤧'],
            'neutral': ['😐', '😑', '🙂', '😌', '😶', '🤔', '😊']
        };
        
        const emotion = emotionData.emotion || 'neutral';
        const confidence = Math.round((emotionData.confidence || 0) * 100);
        const emojiSet = emojiSets[emotion] || emojiSets['neutral'];
        const emoji = emojiSet[Math.floor(Math.random() * emojiSet.length)];
        
        // Smooth emoji transition
        if (elements.emoji) {
            elements.emoji.classList.add('emotion-transition');
            setTimeout(() => {
                elements.emoji.textContent = emoji;
                elements.emoji.classList.remove('emotion-transition');
            }, 200);
        }
        
        if (elements.label) {
            elements.label.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        }
        
        // Enhanced confidence bar with gradient
        if (elements.confidenceFill && elements.confidenceText) {
            elements.confidenceFill.style.width = `${confidence}%`;
            elements.confidenceText.textContent = `${confidence}% confidence`;
            
            // Dynamic color based on confidence and emotion
            const emotionColors = {
                'happy': '#22c55e',
                'sad': '#3b82f6',
                'angry': '#ef4444',
                'fear': '#8b5cf6',
                'surprise': '#f59e0b',
                'disgust': '#84cc16',
                'neutral': '#6b7280'
            };
            
            const baseColor = emotionColors[emotion] || '#10b981';
            const gradient = `linear-gradient(90deg, ${baseColor}, ${baseColor}aa)`;
            elements.confidenceFill.style.background = gradient;
        }
        
        // Enhanced face detection status
        if (elements.faceStatus) {
            if (emotionData.face_detected) {
                elements.faceStatus.classList.add('detected');
                const modelCount = emotionData.model_count || 1;
                const method = emotionData.analysis_method || 'Multi-AI';
                elements.faceStatus.innerHTML = `
                    <i class="fas fa-check-circle"></i>
                    <span>Face detected • ${method} (${modelCount} models)</span>
                `;
                
                if (emotionData.reasoning) {
                    elements.faceStatus.title = `Analysis: ${emotionData.reasoning}`;
                }
            } else {
                elements.faceStatus.classList.remove('detected');
                elements.faceStatus.innerHTML = '<i class="fas fa-search"></i><span>Looking for face...</span>';
                elements.faceStatus.title = '';
            }
        }
        
        // Enhanced emotion context
        if (elements.emotionContext && elements.contextEmotion) {
            if (emotionData.face_detected && emotion !== 'neutral' && confidence >= 40) {
                elements.emotionContext.style.display = 'flex';
                elements.contextEmotion.textContent = emotion;
                
                // Add additional context info
                const contextInfo = elements.emotionContext.querySelector('.context-info');
                if (!contextInfo) {
                    const infoDiv = document.createElement('div');
                    infoDiv.className = 'context-info';
                    elements.emotionContext.appendChild(infoDiv);
                }
                
                const info = elements.emotionContext.querySelector('.context-info');
                if (info) {
                    info.textContent = `${confidence}% confidence • ${emotionData.model_count || 1} AI models`;
                }
            } else {
                elements.emotionContext.style.display = 'none';
            }
        }
    }
    
    updateModelAgreement(agreement) {
        const agreementEl = document.getElementById('modelAgreement');
        if (agreementEl && agreement !== undefined) {
            agreementEl.textContent = `${(agreement * 100).toFixed(0)}%`;
            
            // Color code based on agreement level
            if (agreement >= 0.8) {
                agreementEl.style.color = '#22c55e'; // High agreement - green
            } else if (agreement >= 0.6) {
                agreementEl.style.color = '#f59e0b'; // Medium agreement - yellow
            } else {
                agreementEl.style.color = '#ef4444'; // Low agreement - red
            }
        }
    }
    
    updateStabilityScore(stability) {
        const stabilityEl = document.getElementById('emotionStability');
        if (stabilityEl && stability !== undefined) {
            stabilityEl.textContent = `${(stability * 100).toFixed(0)}%`;
            
            // Color code based on stability
            if (stability >= 0.8) {
                stabilityEl.style.color = '#22c55e';
            } else if (stability >= 0.6) {
                stabilityEl.style.color = '#f59e0b';
            } else {
                stabilityEl.style.color = '#ef4444';
            }
        }
    }
    
    updateModelCount(count) {
        const countEl = document.getElementById('modelCount');
        if (countEl && count !== undefined) {
            countEl.textContent = count;
            this.modelCount = count;
        }
    }
    
    updateModelInfo() {
        const methodEl = document.getElementById('analysisMethod');
        if (methodEl) {
            methodEl.textContent = this.analysisMethod;
        }
        
        // Update model indicators
        this.updateModelIndicators();
    }
    
    updateModelIndicators() {
        const indicatorsEl = document.getElementById('modelIndicators');
        if (!indicatorsEl) return;
        
        const modelTypes = ['DeepFace', 'FER', 'HuggingFace', 'Custom CNN'];
        indicatorsEl.innerHTML = '';
        
        modelTypes.forEach((modelType, index) => {
            const indicator = document.createElement('div');
            indicator.className = 'model-indicator';
            
            // Simulate model status (in real app, this would come from backend)
            const isActive = index < this.modelCount;
            
            indicator.innerHTML = `
                <div class="indicator-dot ${isActive ? 'active' : 'inactive'}"></div>
                <span class="indicator-label">${modelType}</span>
            `;
            
            indicatorsEl.appendChild(indicator);
        });
    }
    
    updateDetailedScores(scores) {
        // Create or update detailed scores panel
        let scoresPanel = document.querySelector('.detailed-scores-panel');
        
        if (!scoresPanel) {
            scoresPanel = document.createElement('div');
            scoresPanel.className = 'detailed-scores-panel';
            scoresPanel.innerHTML = `
                <div class="panel-header">
                    <h5><i class="fas fa-chart-bar"></i> Detailed Scores</h5>
                    <button class="toggle-btn" id="toggleScores">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                </div>
                <div class="scores-content" id="scoresContent" style="display: none;">
                    <div class="emotion-bars" id="emotionBars"></div>
                </div>
            `;
            
            const emotionDisplay = document.querySelector('.emotion-display');
            if (emotionDisplay) {
                emotionDisplay.appendChild(scoresPanel);
            }
            
            // Add toggle functionality
            document.getElementById('toggleScores').addEventListener('click', () => {
                const content = document.getElementById('scoresContent');
                const icon = document.querySelector('#toggleScores i');
                
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    icon.className = 'fas fa-chevron-up';
                } else {
                    content.style.display = 'none';
                    icon.className = 'fas fa-chevron-down';
                }
            });
        }
        
        // Update emotion bars
        const barsEl = document.getElementById('emotionBars');
        if (barsEl && scores) {
            barsEl.innerHTML = '';
            
            // Sort emotions by score
            const sortedEmotions = Object.entries(scores)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 7); // Show top 7 emotions
            
            sortedEmotions.forEach(([emotion, score]) => {
                const barContainer = document.createElement('div');
                barContainer.className = 'emotion-bar-container';
                
                const percentage = (score * 100).toFixed(1);
                const color = this.emotionColors[emotion] || '#6b7280';
                
                barContainer.innerHTML = `
                    <div class="emotion-bar-label">
                        <span class="emotion-name">${emotion}</span>
                        <span class="emotion-score">${percentage}%</span>
                    </div>
                    <div class="emotion-bar-track">
                        <div class="emotion-bar-fill" style="width: ${percentage}%; background: ${color}"></div>
                    </div>
                `;
                
                barsEl.appendChild(barContainer);
            });
        }
    }
    
    setupUIEventListeners() {
        // Chat input
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            messageInput.addEventListener('input', (e) => {
                this.adjustTextareaHeight(e.target);
            });
        }
        
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        // Camera controls
        const toggleCamera = document.getElementById('toggleCamera');
        if (toggleCamera) {
            toggleCamera.addEventListener('click', () => {
                this.toggleVideoStream();
            });
        }
        
        // Trends panel toggle
        const toggleTrends = document.getElementById('toggleTrends');
        if (toggleTrends) {
            toggleTrends.addEventListener('click', () => {
                this.toggleTrendsPanel();
            });
        }
    }
    
    toggleTrendsPanel() {
        const content = document.getElementById('trendsContent');
        const icon = document.querySelector('#toggleTrends i');
        
        if (!content || !icon) return;
        
        if (content.style.display === 'none') {
            content.style.display = 'block';
            icon.className = 'fas fa-chevron-up';
        } else {
            content.style.display = 'none';
            icon.className = 'fas fa-chevron-down';
        }
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
                this.startAdvancedEmotionDetection();
            });
            
            this.showStatusMessage('Camera activated - Multi-AI analysis ready', 'success');
            
        } catch (error) {
            console.error('Camera error:', error);
            this.showCameraError();
            this.showStatusMessage('Camera access denied or unavailable', 'error');
        }
    }
    
    startAdvancedEmotionDetection() {
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
                const imageData = this.canvasElement.toDataURL('image/jpeg', 0.8);
                
                this.socket.emit('video_frame', { image: imageData });
                this.lastEmotionUpdate = now;
                
            } catch (error) {
                console.error('Frame processing error:', error);
            }
            
            requestAnimationFrame(processFrame);
        };
        
        requestAnimationFrame(processFrame);
    }
    
    sendMessage() {
        const messageInput = document.getElementById('messageInput');
        if (!messageInput) return;
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addAdvancedMessage(message, 'user');
        
        // Show enhanced typing indicator
        this.showAdvancedTypingIndicator();
        
        // Send message to server
        this.socket.emit('chat_message', { message });
        
        // Clear input
        messageInput.value = '';
        this.adjustTextareaHeight(messageInput);
    }
    
    addAdvancedMessage(content, sender, context = null) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message fade-in`;
        
        const avatar = sender === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';
        
        const timestamp = this.formatTimestamp(new Date());
        
        // Enhanced emotion context display for multi-AI system
        let emotionInfo = '';
        if (sender === 'assistant' && context?.emotion_context && context.emotion_context.emotion !== 'neutral') {
            const emotion = context.emotion_context.emotion;
            const confidence = Math.round((context.emotion_context.confidence || 0) * 100);
            const modelCount = context.emotion_context.model_count || 1;
            const agreement = context.emotion_context.agreement_ratio || 0;
            const method = context.analysis_method || 'Multi-AI';
            
            const emotionEmojis = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 
                'fear': '😰', 'surprise': '😲', 'disgust': '🤢'
            };
            
            emotionInfo = `
                <div class="advanced-emotion-context">
                    <div class="context-header">
                        ${emotionEmojis[emotion] || '💭'}
                        <span>Responding to your <strong>${emotion}</strong> mood</span>
                    </div>
                    <div class="context-details">
                        <div class="detail-item">
                            <i class="fas fa-percentage"></i>
                            <span>${confidence}% confidence</span>
                        </div>
                        <div class="detail-item">
                            <i class="fas fa-brain"></i>
                            <span>${modelCount} AI models</span>
                        </div>
                        <div class="detail-item">
                            <i class="fas fa-handshake"></i>
                            <span>${(agreement * 100).toFixed(0)}% agreement</span>
                        </div>
                        <div class="detail-item">
                            <i class="fas fa-cogs"></i>
                            <span>${method}</span>
                        </div>
                    </div>
                </div>
            `;
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
    
    showAdvancedTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (!indicator) return;
        
        const modelText = this.modelCount > 0 ? `${this.modelCount} AI models` : 'Multi-AI system';
        indicator.innerHTML = `
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span class="typing-text">${modelText} analyzing and responding...</span>
        `;
        indicator.style.display = 'flex';
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
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
    
    toggleVideoStream() {
        const toggleBtn = document.getElementById('toggleCamera');
        if (!toggleBtn) return;
        
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
        
        if (statusDot && statusText) {
            if (connected) {
                statusDot.classList.add('connected');
                statusText.textContent = `Connected (${this.modelCount} AI models)`;
            } else {
                statusDot.classList.remove('connected');
                statusText.textContent = 'Connecting...';
            }
        }
    }
    
    showStatusMessage(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Enhanced toast with model count info
        const toast = document.createElement('div');
        toast.className = `advanced-toast toast-${type}`;
        
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
            max-width: 400px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            background: linear-gradient(135deg, ${colors[type] || colors.info}, ${colors[type] || colors.info}dd);
        `;
        
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${icons[type] || icons.info}</span>
                <div class="toast-message">
                    <div class="toast-title">${message}</div>
                    ${this.modelCount > 0 ? `<div class="toast-subtitle">Multi-AI System (${this.modelCount} models active)</div>` : ''}
                </div>
            </div>
        `;
        
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
        }, 5000);
    }
    
    showCameraError() {
        const videoContainer = this.videoElement.parentElement;
        const errorDiv = document.createElement('div');
        errorDiv.className = 'camera-error';
        errorDiv.innerHTML = `
            <div class="camera-error-content">
                <i class="fas fa-video-slash camera-error-icon"></i>
                <h4 class="camera-error-title">Multi-AI Camera Access Required</h4>
                <p class="camera-error-message">
                    Please allow camera access to enable advanced emotion detection
                    using multiple AI models for higher accuracy.
                </p>
                <button onclick="location.reload()" class="camera-retry-btn">
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
    console.log('🚀 Initializing Advanced Multi-AI Emotion Detection System...');
    new AdvancedMultiAIEmpathyBot();
});

// Enhanced keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const messageInput = document.getElementById('messageInput');
        if (document.activeElement === messageInput) {
            e.preventDefault();
            document.getElementById('sendBtn')?.click();
        }
    }
    
    // F1 to toggle trends panel
    if (e.key === 'F1') {
        e.preventDefault();
        document.getElementById('toggleTrends')?.click();
    }
    
    // F2 to toggle detailed scores
    if (e.key === 'F2') {
        e.preventDefault();
        document.getElementById('toggleScores')?.click();
    }
});

// Performance monitoring
let performanceMetrics = {
    frameProcessingTimes: [],
    emotionDetectionAccuracy: [],
    modelAgreementScores: []
};

// Log performance data
setInterval(() => {
    if (performanceMetrics.frameProcessingTimes.length > 0) {
        const avgProcessingTime = performanceMetrics.frameProcessingTimes.reduce((a, b) => a + b, 0) / performanceMetrics.frameProcessingTimes.length;
        console.log(`📊 Performance: Avg frame processing ${avgProcessingTime.toFixed(2)}ms`);
        
        // Reset metrics
        performanceMetrics.frameProcessingTimes = [];
    }
}, 30000); // Log every 30 seconds