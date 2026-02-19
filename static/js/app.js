// ============================================================================
// Guitar Teacher - Web App JavaScript
// ============================================================================

// State Management
const state = {
    currentMode: 'free_play',
    currentChord: 'E Minor',
    availableChords: [],
    availableCameras: [],
    currentCameraId: 0,
    overlayEnabled: true,
    debugMode: false,
    rawFeed: false,
    frameCount: 0,
    lastTime: Date.now(),
    fps: 0,
    calibrationMode: false,
    calibrationClicks: [],
    fetchInFlight: false  // Guard against overlapping polls
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await initializeApp();
});

async function initializeApp() {
    try {
        await loadCameras();
        await loadChords();
        setupEventListeners();
        startDetectionLoop();

        // Fade out loading screen
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            loadingScreen.classList.add('fade-out');
            loadingScreen.addEventListener('transitionend', () => {
                loadingScreen.style.display = 'none';
            }, { once: true });
        }
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
        if (!response.ok) return;
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
    if (!select) return;

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
        const response = await fetch(`/api/set_camera/${cameraId}`, { method: 'POST' });
        if (!response.ok) return;
        const data = await response.json();

        if (data.success) {
            state.currentCameraId = cameraId;
            updateCameraSelector();

            // Reload video feed cleanly
            const videoFeed = document.getElementById('videoFeed');
            if (videoFeed) {
                videoFeed.src = '/video_feed?t=' + Date.now();
            }
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
        if (!response.ok) return;
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
    if (!selector) return;

    if (state.availableChords.length === 0) {
        selector.innerHTML = '<p style="color: rgba(255,255,255,0.7); text-align: center;">No chords available</p>';
        return;
    }

    selector.innerHTML = state.availableChords.map(chord =>
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
        if (!response.ok) return;
        const data = await response.json();

        if (data.success) {
            state.currentChord = chordName;
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
// Overlay / Debug / Raw Feed Toggles
// ============================================================================

async function toggleOverlay() {
    try {
        const response = await fetch('/api/toggle_overlay');
        if (!response.ok) return;
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

async function toggleDebug() {
    try {
        const response = await fetch('/api/toggle_debug', { method: 'POST' });
        if (!response.ok) return;
        const data = await response.json();
        if (data.success) {
            state.debugMode = data.debug_mode;
            updateDebugButton();
            showNotification(`Debug mode ${state.debugMode ? 'enabled' : 'disabled'}`);
        }
    } catch (error) {
        console.error('Error toggling debug:', error);
    }
}

async function toggleRawFeed() {
    try {
        const response = await fetch('/api/toggle_raw_feed', { method: 'POST' });
        if (!response.ok) return;
        const data = await response.json();
        if (data.success) {
            state.rawFeed = data.raw_feed;
            updateRawFeedButton();
            showNotification(`Raw feed ${state.rawFeed ? 'enabled' : 'disabled'}`);
        }
    } catch (error) {
        console.error('Error toggling raw feed:', error);
    }
}

function updateOverlayToggle() {
    const toggle = document.getElementById('overlayToggle');
    if (toggle) toggle.classList.toggle('active', state.overlayEnabled);
}

function updateDebugButton() {
    const btn = document.getElementById('debugToggle');
    if (btn) btn.classList.toggle('active', state.debugMode);
}

function updateRawFeedButton() {
    const btn = document.getElementById('rawFeedToggle');
    if (btn) btn.classList.toggle('active', state.rawFeed);
}

// ============================================================================
// Mode Management
// ============================================================================

async function setMode(mode) {
    try {
        const response = await fetch(`/set_mode/${mode}`);
        if (!response.ok) return;
        const data = await response.json();

        if (data.success) {
            state.currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            document.getElementById('freePlayMode').classList.toggle('active', mode === 'free_play');
            document.getElementById('chordTrainerMode').classList.toggle('active', mode === 'chord_trainer');

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

function startDetectionLoop() {
    setInterval(updateDetectionData, 500);  // Poll every 500ms (was 100ms)
}

async function updateDetectionData() {
    // Skip if a request is already in flight
    if (state.fetchInFlight) return;
    state.fetchInFlight = true;

    try {
        const response = await fetch('/detection_data');
        if (!response.ok) {
            state.fetchInFlight = false;
            return;
        }
        const data = await response.json();

        // Calculate FPS
        state.frameCount++;
        const now = Date.now();
        if (now - state.lastTime >= 1000) {
            state.fps = state.frameCount;
            state.frameCount = 0;
            state.lastTime = now;
            const fpsEl = document.getElementById('fpsDisplay');
            if (fpsEl) fpsEl.textContent = `${state.fps} FPS`;
        }

        updateDetectionStatus(data);

        if (state.currentMode === 'free_play') {
            updateFreePlayMode(data);
        } else if (state.currentMode === 'chord_trainer') {
            updateChordTrainerMode(data);
        }

        const placeholder = document.getElementById('videoPlaceholder');
        if (placeholder && data.neck_detected !== undefined) {
            placeholder.classList.add('hidden');
        }
    } catch (error) {
        console.error('Error fetching detection data:', error);
    } finally {
        state.fetchInFlight = false;
    }
}

function updateDetectionStatus(data) {
    const statusElement = document.getElementById('detectionStatus');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    if (!statusElement || !statusDot || !statusText) return;

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
    const audioNote = document.getElementById('audioNote');
    const audioFreq = document.getElementById('audioFreq');

    if (audioNote && audioFreq) {
        if (data.audio_note) {
            audioNote.textContent = data.audio_note;
            audioFreq.textContent = `${data.audio_freq.toFixed(1)} Hz`;
        } else {
            audioNote.textContent = '--';
            audioFreq.textContent = 'Listening...';
        }
    }

    const neckEl = document.getElementById('neckStatus');
    const fretsEl = document.getElementById('fretsCount');
    if (neckEl) neckEl.textContent = data.neck_detected ? '✓' : '✗';
    if (fretsEl) fretsEl.textContent = data.frets_detected || 0;

    updateNotesList(data.notes || []);
}

function updateNotesList(notes) {
    const notesList = document.getElementById('notesList');
    const notesCount = document.getElementById('notesCount');
    if (!notesList || !notesCount) return;

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
    const neckEl = document.getElementById('neckStatus2');
    const fretsEl = document.getElementById('fretsCount2');
    if (neckEl) neckEl.textContent = data.neck_detected ? '✓' : '✗';
    if (fretsEl) fretsEl.textContent = data.frets_detected || 0;

    if (data.evaluation && data.target_chord) {
        const chordEval = data.evaluation;
        const chord = data.target_chord;

        const progressPercent = Math.round(chordEval.accuracy);
        const percentEl = document.getElementById('progressPercent');
        if (percentEl) percentEl.textContent = `${progressPercent}%`;

        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            const circumference = 2 * Math.PI * 45;
            const offset = circumference - (progressPercent / 100) * circumference;
            progressBar.style.strokeDashoffset = offset;
        }

        const checklist = document.getElementById('fingerChecklist');
        if (checklist) {
            checklist.innerHTML = chord.fingering.map(([string, fret, finger]) => {
                const isCorrect = chordEval.correct_fingers.some(cf =>
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
        }

        const perfectBadge = document.getElementById('perfectBadge');
        if (perfectBadge) {
            perfectBadge.style.display = chordEval.is_perfect ? 'flex' : 'none';
        }
    } else {
        const percentEl = document.getElementById('progressPercent');
        if (percentEl) percentEl.textContent = '0%';

        const progressBar = document.getElementById('progressBar');
        if (progressBar) progressBar.style.strokeDashoffset = 2 * Math.PI * 45;

        if (data.target_chord) {
            const checklist = document.getElementById('fingerChecklist');
            if (checklist) {
                checklist.innerHTML = data.target_chord.fingering.map(([string, fret, finger]) => `
                    <div class="finger-item">
                        <div class="finger-checkbox"></div>
                        <div>
                            <strong>${finger}</strong>: String ${string}, Fret ${fret}
                        </div>
                    </div>
                `).join('');
            }
        }

        const perfectBadge = document.getElementById('perfectBadge');
        if (perfectBadge) perfectBadge.style.display = 'none';
    }
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Camera selector
    const cameraSelect = document.getElementById('cameraSelect');
    if (cameraSelect) {
        cameraSelect.addEventListener('change', (e) => {
            const cameraId = parseInt(e.target.value);
            if (!isNaN(cameraId)) switchCamera(cameraId);
        });
    }

    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    // Toggle buttons (single registration each)
    const overlayToggle = document.getElementById('overlayToggle');
    if (overlayToggle) overlayToggle.addEventListener('click', toggleOverlay);

    const debugToggle = document.getElementById('debugToggle');
    if (debugToggle) debugToggle.addEventListener('click', toggleDebug);

    const rawFeedToggle = document.getElementById('rawFeedToggle');
    if (rawFeedToggle) rawFeedToggle.addEventListener('click', toggleRawFeed);

    // Video click handler for calibration
    const videoFeed = document.getElementById('videoFeed');
    if (videoFeed) videoFeed.addEventListener('click', handleVideoClick);

    // Initial overlay state
    updateOverlayToggle();

    // Sync overlay status from server
    fetch('/api/overlay_status')
        .then(res => {
            if (!res.ok) throw new Error('Not OK');
            return res.json();
        })
        .then(data => {
            state.overlayEnabled = data.enabled;
            if (data.chord) state.currentChord = data.chord;
            updateOverlayToggle();
        })
        .catch(err => console.error('Error fetching overlay status:', err));
}

// ============================================================================
// Calibration
// ============================================================================

function handleVideoClick(event) {
    if (!state.calibrationMode) return;

    const videoFeed = document.getElementById('videoFeed');
    const rect = videoFeed.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const videoWidth = videoFeed.videoWidth || videoFeed.naturalWidth || rect.width;
    const videoHeight = videoFeed.videoHeight || videoFeed.naturalHeight || rect.height;
    const scaleY = videoHeight / rect.height;
    const scaledY = y * scaleY;

    state.calibrationClicks.push(scaledY);
    showCalibrationFeedback(x, y, state.calibrationClicks.length);

    if (state.calibrationClicks.length >= 6) {
        calibrateStrings(state.calibrationClicks);
        state.calibrationMode = false;
        state.calibrationClicks = [];
    }
}

function showCalibrationFeedback(x, y, clickNumber) {
    const marker = document.createElement('div');
    marker.style.cssText = `
        position: absolute; left: ${x}px; top: ${y}px;
        width: 20px; height: 20px; border-radius: 50%;
        background: #34c759; border: 2px solid white;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 12px; font-weight: bold;
        pointer-events: none; z-index: 1000;
    `;
    marker.textContent = clickNumber;

    const videoWrapper = document.querySelector('.video-wrapper');
    if (videoWrapper) {
        videoWrapper.appendChild(marker);
        setTimeout(() => marker.remove(), 2000);
    }
}

async function calibrateStrings(positions) {
    try {
        const response = await fetch('/api/calibrate_strings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ positions })
        });
        if (!response.ok) return;
        const data = await response.json();

        if (data.success) {
            showNotification('String calibration saved!');
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
        if (!response.ok) return;
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
    console.log('Notification:', message);
}

function showError(message) {
    console.error('Error:', message);
    alert(message);
}

// Make functions globally available
window.selectChord = selectChord;
window.switchCamera = switchCamera;
window.toggleOverlay = toggleOverlay;
window.toggleDebug = toggleDebug;
window.toggleRawFeed = toggleRawFeed;
window.setMode = setMode;
window.startCalibration = startCalibration;
window.resetCalibration = resetCalibration;
