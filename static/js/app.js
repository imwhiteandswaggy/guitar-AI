// ============================================================================
// Guitar Teacher - Professional Web App JavaScript
// ============================================================================

// State Management
const state = {
    currentMode: 'free_play',
    currentChord: 'E Minor',
    availableChords: [],
    availableCameras: [],
    currentCameraId: 0,
    overlayEnabled: true,
    frameCount: 0,
    lastTime: Date.now(),
    fps: 0,
    calibrationMode: false,
    calibrationClicks: []
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await initializeApp();
});

async function initializeApp() {
    try {
        // Load cameras
        await loadCameras();
        
        // Load chords
        await loadChords();
        
        // Setup event listeners
        setupEventListeners();
        
        // Start detection loop
        startDetectionLoop();
        
        // Hide loading screen
        setTimeout(() => {
            document.getElementById('loadingScreen').classList.add('hidden');
        }, 1000);
        
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to initialize app. Please refresh the page.');
    }
}

// ============================================================================
// Camera Management
// ============================================================================

async function loadCameras() {
    try {
        const response = await fetch('/api/cameras');
        const data = await response.json();
        
        state.availableCameras = data.cameras || [];
        state.currentCameraId = data.current || 0;
        
        updateCameraSelector();
    } catch (error) {
        console.error('Error loading cameras:', error);
    }
}

function updateCameraSelector() {
    const select = document.getElementById('cameraSelect');
    
    if (state.availableCameras.length === 0) {
        select.innerHTML = '<option value="">No cameras found</option>';
        return;
    }
    
    select.innerHTML = state.availableCameras.map(cam => 
        `<option value="${cam.id}" ${cam.id === state.currentCameraId ? 'selected' : ''}>
            ${cam.name}
        </option>`
    ).join('');
}

async function switchCamera(cameraId) {
    try {
        const response = await fetch(`/api/set_camera/${cameraId}`);
        const data = await response.json();
        
        if (data.success) {
            state.currentCameraId = cameraId;
            updateCameraSelector();
            
            // Reload video feed
            const videoFeed = document.getElementById('videoFeed');
            const src = videoFeed.src;
            videoFeed.src = '';
            setTimeout(() => {
                videoFeed.src = src + '?t=' + Date.now();
            }, 100);
            
            showNotification('Camera switched successfully');
        } else {
            showError(data.error || 'Failed to switch camera');
        }
    } catch (error) {
        console.error('Error switching camera:', error);
        showError('Failed to switch camera');
    }
}

// ============================================================================
// Chord Management
// ============================================================================

async function loadChords() {
    try {
        const response = await fetch('/get_chords');
        const data = await response.json();
        
        state.availableChords = data.beginner_chords || data.chords || [];
        state.currentChord = state.availableChords[0] || 'E Minor';
        
        updateChordSelector();
    } catch (error) {
        console.error('Error loading chords:', error);
    }
}

function updateChordSelector() {
    const selector = document.getElementById('chordSelector');
    
    if (state.availableChords.length === 0) {
        selector.innerHTML = '<p style="color: rgba(255,255,255,0.7); text-align: center;">No chords available</p>';
        return;
    }
    
    selector.innerHTML = state.availableChords.map((chord, index) => 
        `<button class="chord-btn ${chord === state.currentChord ? 'active' : ''}" 
                 data-chord="${chord}"
                 onclick="selectChord('${chord}')">
            ${chord}
        </button>`
    ).join('');
}

async function selectChord(chordName) {
    try {
        const response = await fetch(`/set_chord/${encodeURIComponent(chordName)}`);
        const data = await response.json();
        
        if (data.success) {
            state.currentChord = chordName;
            
            // Update UI
            document.querySelectorAll('.chord-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.chord === chordName);
            });
            
            document.getElementById('currentChordName').textContent = chordName;
            document.getElementById('perfectBadge').style.display = 'none';
            
            showNotification(`Switched to ${chordName}`);
        }
    } catch (error) {
        console.error('Error selecting chord:', error);
    }
}

// ============================================================================
// Overlay Management
// ============================================================================

async function toggleOverlay() {
    try {
        const response = await fetch('/api/toggle_overlay');
        const data = await response.json();
        
        if (data.success) {
            state.overlayEnabled = data.enabled;
            updateOverlayToggle();
            showNotification(`Overlay ${data.enabled ? 'enabled' : 'disabled'}`);
        }
    } catch (error) {
        console.error('Error toggling overlay:', error);
    }
}

function updateOverlayToggle() {
    const toggle = document.getElementById('overlayToggle');
    toggle.classList.toggle('active', state.overlayEnabled);
}

// ============================================================================
// Mode Management
// ============================================================================

async function setMode(mode) {
    try {
        const response = await fetch(`/set_mode/${mode}`);
        const data = await response.json();
        
        if (data.success) {
            state.currentMode = mode;
            
            // Update buttons
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            
            // Update content
            document.getElementById('freePlayMode').classList.toggle('active', mode === 'free_play');
            document.getElementById('chordTrainerMode').classList.toggle('active', mode === 'chord_trainer');
            
            // Load chords if entering chord trainer
            if (mode === 'chord_trainer' && state.availableChords.length === 0) {
                await loadChords();
            }
        }
    } catch (error) {
        console.error('Error setting mode:', error);
    }
}

// ============================================================================
// Detection Loop
// ============================================================================

async function startDetectionLoop() {
    setInterval(updateDetectionData, 100); // Update every 100ms
}

async function updateDetectionData() {
    try {
        const response = await fetch('/detection_data');
        const data = await response.json();
        
        // Calculate FPS
        state.frameCount++;
        const now = Date.now();
        if (now - state.lastTime >= 1000) {
            state.fps = state.frameCount;
            state.frameCount = 0;
            state.lastTime = now;
            document.getElementById('fpsDisplay').textContent = `${state.fps} FPS`;
        }
        
        // Update detection status
        updateDetectionStatus(data);
        
        // Update mode-specific content
        if (state.currentMode === 'free_play') {
            updateFreePlayMode(data);
        } else if (state.currentMode === 'chord_trainer') {
            updateChordTrainerMode(data);
        }
        
        // Hide video placeholder if video is working
        const placeholder = document.getElementById('videoPlaceholder');
        if (data.neck_detected !== undefined) {
            placeholder.classList.add('hidden');
        }
        
    } catch (error) {
        console.error('Error fetching detection data:', error);
        document.getElementById('videoPlaceholder').classList.remove('hidden');
    }
}

function updateDetectionStatus(data) {
    const statusElement = document.getElementById('detectionStatus');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    if (data.neck_detected) {
        statusElement.className = 'overlay-card detection-status success';
        statusElement.innerHTML = '<span class="status-icon">✓</span><span>Guitar Detected</span>';
        statusDot.className = 'status-dot';
        statusText.textContent = 'Live';
    } else {
        statusElement.className = 'overlay-card detection-status warning';
        statusElement.innerHTML = '<span class="status-icon">⚠</span><span>No Guitar</span>';
        statusDot.className = 'status-dot warning';
        statusText.textContent = 'Waiting...';
    }
}

function updateFreePlayMode(data) {
    // Update audio detection
    const audioNote = document.getElementById('audioNote');
    const audioFreq = document.getElementById('audioFreq');
    
    if (data.audio_note) {
        audioNote.textContent = data.audio_note;
        audioFreq.textContent = `${data.audio_freq.toFixed(1)} Hz`;
    } else {
        audioNote.textContent = '--';
        audioFreq.textContent = 'Listening...';
    }
    
    // Update stats
    document.getElementById('neckStatus').textContent = data.neck_detected ? '✓' : '✗';
    document.getElementById('fretsCount').textContent = data.frets_detected || 0;
    
    // Update detected notes
    updateNotesList(data.notes || []);
}

function updateNotesList(notes) {
    const notesList = document.getElementById('notesList');
    const notesCount = document.getElementById('notesCount');
    
    notesCount.textContent = notes.length;
    
    if (notes.length === 0) {
        notesList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🎸</div>
                <div class="empty-text">
                    Start playing your guitar<br>
                    Notes will appear here
                </div>
            </div>
        `;
        return;
    }
    
    notesList.innerHTML = notes.map(note => {
        const matchClass = note.matches_audio === true ? 'correct' :
                          note.matches_audio === false ? 'incorrect' : '';
        const badge = note.matches_audio === true ? '✓' :
                     note.matches_audio === false ? '?' : '•';
        
        const position = note.fret === 0 ? 'Open' : `Fret ${note.fret}`;
        
        return `
            <div class="note-item ${matchClass}">
                <div class="note-info">
                    <div class="note-name">${note.note}</div>
                    <div class="note-details">
                        ${note.finger} • String ${note.string} • ${position}
                    </div>
                </div>
                <div class="note-badge">${badge}</div>
            </div>
        `;
    }).join('');
}

function updateChordTrainerMode(data) {
    // Update stats
    document.getElementById('neckStatus2').textContent = data.neck_detected ? '✓' : '✗';
    document.getElementById('fretsCount2').textContent = data.frets_detected || 0;
    
    // Update chord evaluation
    if (data.evaluation && data.target_chord) {
        const eval = data.evaluation;
        const chord = data.target_chord;
        
        // Update progress ring
        const progressPercent = Math.round(eval.accuracy);
        document.getElementById('progressPercent').textContent = `${progressPercent}%`;
        
        const progressBar = document.getElementById('progressBar');
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (progressPercent / 100) * circumference;
        progressBar.style.strokeDashoffset = offset;
        
        // Update finger checklist
        const checklist = document.getElementById('fingerChecklist');
        checklist.innerHTML = chord.fingering.map(([string, fret, finger]) => {
            const isCorrect = eval.correct_fingers.some(cf => 
                cf.target[0] === string && cf.target[1] === fret && cf.target[2] === finger
            );
            
            return `
                <div class="finger-item ${isCorrect ? 'correct' : ''}">
                    <div class="finger-checkbox ${isCorrect ? 'checked' : ''}">
                        ${isCorrect ? '✓' : ''}
                    </div>
                    <div>
                        <strong>${finger}</strong>: String ${string}, Fret ${fret}
                    </div>
                </div>
            `;
        }).join('');
        
        // Show perfect badge
        document.getElementById('perfectBadge').style.display = 
            eval.is_perfect ? 'flex' : 'none';
    } else {
        // No fingers detected
        document.getElementById('progressPercent').textContent = '0%';
        const progressBar = document.getElementById('progressBar');
        progressBar.style.strokeDashoffset = 2 * Math.PI * 45;
        
        if (data.target_chord) {
            const checklist = document.getElementById('fingerChecklist');
            checklist.innerHTML = data.target_chord.fingering.map(([string, fret, finger]) => `
                <div class="finger-item">
                    <div class="finger-checkbox"></div>
                    <div>
                        <strong>${finger}</strong>: String ${string}, Fret ${fret}
                    </div>
                </div>
            `).join('');
        }
        
        document.getElementById('perfectBadge').style.display = 'none';
    }
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Camera selector
    document.getElementById('cameraSelect').addEventListener('change', (e) => {
        const cameraId = parseInt(e.target.value);
        if (!isNaN(cameraId)) {
            switchCamera(cameraId);
        }
    });
    
    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            setMode(mode);
        });
    });
    
    // Overlay toggle
    document.getElementById('overlayToggle').addEventListener('click', toggleOverlay);
    
    // Video feed click handler for calibration
    const videoFeed = document.getElementById('videoFeed');
    videoFeed.addEventListener('click', handleVideoClick);
    
    // Initial overlay state
    updateOverlayToggle();
    
    // Check overlay status on load
    fetch('/api/overlay_status')
        .then(res => res.json())
        .then(data => {
            state.overlayEnabled = data.enabled;
            if (data.chord) {
                state.currentChord = data.chord;
            }
            updateOverlayToggle();
        })
        .catch(err => console.error('Error fetching overlay status:', err));
}

function handleVideoClick(event) {
    if (!state.calibrationMode) return;
    
    const videoFeed = document.getElementById('videoFeed');
    const rect = videoFeed.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Scale coordinates to video dimensions
    const videoWidth = videoFeed.videoWidth || videoFeed.naturalWidth || rect.width;
    const videoHeight = videoFeed.videoHeight || videoFeed.naturalHeight || rect.height;
    const scaleX = videoWidth / rect.width;
    const scaleY = videoHeight / rect.height;
    
    const scaledY = y * scaleY;
    
    state.calibrationClicks.push(scaledY);
    
    // Visual feedback
    showCalibrationFeedback(x, y, state.calibrationClicks.length);
    
    if (state.calibrationClicks.length >= 6) {
        // Send calibration data
        calibrateStrings(state.calibrationClicks);
        state.calibrationMode = false;
        state.calibrationClicks = [];
    }
}

function showCalibrationFeedback(x, y, clickNumber) {
    // Create temporary marker
    const marker = document.createElement('div');
    marker.style.position = 'absolute';
    marker.style.left = x + 'px';
    marker.style.top = y + 'px';
    marker.style.width = '20px';
    marker.style.height = '20px';
    marker.style.borderRadius = '50%';
    marker.style.background = '#34c759';
    marker.style.border = '2px solid white';
    marker.style.display = 'flex';
    marker.style.alignItems = 'center';
    marker.style.justifyContent = 'center';
    marker.style.color = 'white';
    marker.style.fontSize = '12px';
    marker.style.fontWeight = 'bold';
    marker.style.pointerEvents = 'none';
    marker.style.zIndex = '1000';
    marker.textContent = clickNumber;
    
    const videoWrapper = document.querySelector('.video-wrapper');
    videoWrapper.appendChild(marker);
    
    setTimeout(() => marker.remove(), 2000);
}

async function calibrateStrings(positions) {
    try {
        const response = await fetch('/api/calibrate_strings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ positions: positions })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('String calibration saved! Overlay should now be accurate.');
            // Reload page to apply calibration
            setTimeout(() => location.reload(), 1000);
        } else {
            showError(data.error || 'Calibration failed');
        }
    } catch (error) {
        console.error('Error calibrating strings:', error);
        showError('Failed to save calibration');
    }
}

function startCalibration() {
    state.calibrationMode = true;
    state.calibrationClicks = [];
    showNotification('Click on each string from top to bottom (thinnest to thickest). Click 6 times.');
}

async function resetCalibration() {
    try {
        const response = await fetch('/api/reset_calibration', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showNotification('Calibration reset. Using auto-detection.');
            location.reload();
        }
    } catch (error) {
        console.error('Error resetting calibration:', error);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function showNotification(message) {
    // Simple notification - could be enhanced with a toast library
    console.log('Notification:', message);
}

function showError(message) {
    // Simple error display - could be enhanced
    console.error('Error:', message);
    alert(message);
}

// Make functions globally available
window.selectChord = selectChord;
window.switchCamera = switchCamera;
window.toggleOverlay = toggleOverlay;
window.setMode = setMode;
window.startCalibration = startCalibration;
window.resetCalibration = resetCalibration;
