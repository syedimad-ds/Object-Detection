// ============================================================
// CORE ELEMENTS
// ============================================================
const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');
const switchCamBtn = document.getElementById('switchCamBtn');
const fpsDisplay = document.getElementById('fpsDisplay');
const detectionCount = document.getElementById('detectionCount');

let model;
let currentFacingMode = 'environment';
let animationFrameId = null;
let isDetecting = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// ============================================================
// DEVICE DETECTION
// ============================================================
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

if (!isMobile && switchCamBtn) {
    switchCamBtn.style.display = 'none';
}

const modelPath = isMobile
    ? './yolov8n_web_model/model.json'
    : './yolov8s_web_model/model.json';

const CONF_THRESHOLD = isMobile ? 0.28 : 0.38;
const IOU_THRESHOLD  = 0.40;
const INPUT_SIZE     = 640;

// ============================================================
// YOLO CLASS NAMES & COLORS
// ============================================================
const yoloClasses = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

const classColors = [
    '#FF3838','#FF9D97','#FF701F','#FFB21D','#CFD231','#48F90A','#92CC17','#3DDB86',
    '#1A9334','#00D4BB','#2C99A8','#00C2FF','#344593','#6473FF','#0018EC','#8438FF',
    '#520085','#CB38FF','#FF95C8','#FF37C7'
];

function getColor(classId) {
    return classColors[classId % classColors.length];
}

// ============================================================
// BUG FIX #1: CORRECT IOU CALCULATION (b2 vars were missing!)
// ============================================================
function calculateIOU(box1, box2) {
    const b1Left   = box1.x - box1.w / 2;
    const b1Right  = box1.x + box1.w / 2;
    const b1Top    = box1.y - box1.h / 2;
    const b1Bottom = box1.y + box1.h / 2;

    const b2Left   = box2.x - box2.w / 2;
    const b2Right  = box2.x + box2.w / 2;
    const b2Top    = box2.y - box2.h / 2;
    const b2Bottom = box2.y + box2.h / 2;

    const xA = Math.max(b1Left, b2Left);
    const yA = Math.max(b1Top, b2Top);
    const xB = Math.min(b1Right, b2Right);
    const yB = Math.min(b1Bottom, b2Bottom);

    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const unionArea = (box1.w * box1.h) + (box2.w * box2.h) - intersectionArea;

    if (unionArea === 0) return 0;
    return intersectionArea / unionArea;
}

// ============================================================
// MODEL LOADING
// ============================================================
async function loadModel() {
    try {
        setStatus('loading', '⏳ Initializing WebGL backend...');

        // Force WebGL for GPU acceleration
        await tf.setBackend('webgl');
        await tf.ready();

        // Enable memory optimizations
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FLUSH_THRESHOLD', -1);

        setStatus('loading', '⏳ Downloading model weights...');
        model = await tf.loadGraphModel(modelPath);

        // BUG FIX #2: Warm-up with actual inference size so the GPU compiles
        // shaders properly — this is what causes the 1-2 min freeze on first frame
        setStatus('loading', '🔥 Warming up GPU shaders (one-time, ~10s)...');
        const warmupTensor = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        const warmupResult = await model.executeAsync(warmupTensor);
        // Properly dispose array of tensors or single tensor
        if (Array.isArray(warmupResult)) {
            warmupResult.forEach(t => t.dispose());
        } else {
            warmupResult.dispose();
        }
        warmupTensor.dispose();

        const modeLabel = isMobile ? 'Nano · Fast Mode' : 'Small · High Accuracy';
        setStatus('active', `✅ System Active — ${modeLabel}`);

        startWebcam();
    } catch (err) {
        setStatus('error', `❌ Model load failed: ${err.message}`);
        console.error(err);
    }
}

// ============================================================
// WEBCAM
// ============================================================
async function startWebcam() {
    // Stop any existing stream
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    // Cancel existing detection loop while switching
    isDetecting = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    try {
        const constraints = {
            video: {
                facingMode: isMobile ? currentFacingMode : 'user',
                width:  { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        await new Promise(resolve => {
            video.onloadedmetadata = resolve;
        });
        await video.play();

        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;

        isDetecting = true;
        detectFrame();

    } catch (err) {
        setStatus('error', '❌ Camera access denied. Please allow permissions.');
        console.error(err);
    }
}

// Camera switch button
if (switchCamBtn) {
    switchCamBtn.addEventListener('click', async () => {
        if (!isMobile) return;
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        switchCamBtn.disabled = true;
        switchCamBtn.innerText = '🔄 Switching...';
        await startWebcam();
        switchCamBtn.disabled = false;
        switchCamBtn.innerText = '🔄 Switch Camera';
    });
}

// ============================================================
// FPS TRACKER
// ============================================================
function updateFPS() {
    frameCount++;
    const now = performance.now();
    const delta = now - lastFrameTime;
    if (delta >= 1000) {
        fps = Math.round((frameCount * 1000) / delta);
        frameCount = 0;
        lastFrameTime = now;
        if (fpsDisplay) fpsDisplay.textContent = `${fps} FPS`;
    }
}

// ============================================================
// MAIN DETECTION LOOP
// BUG FIX #3: Use typed array (data()) not .array() — 10-50x faster
// BUG FIX #4: Proper tensor disposal for arrays of tensors
// BUG FIX #5: Bounding box mirror logic unified and correct
// ============================================================
async function detectFrame() {
    if (!isDetecting) return;

    // Don't run inference if video isn't ready
    if (video.readyState < 2) {
        animationFrameId = requestAnimationFrame(detectFrame);
        return;
    }

    const isFrontCam = currentFacingMode === 'user' || !isMobile;

    // ---- Preprocessing: build input tensor inside tf.tidy ----
    const inputTensor = tf.tidy(() => {
        return tf.browser.fromPixels(video)
            .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
            .expandDims(0)
            .toFloat()
            .div(255.0);
    });

    let rawOutput = null;

    try {
        rawOutput = await model.executeAsync(inputTensor);

        // Normalize output to always be a single tensor
        const outputTensor = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
        const shape = outputTensor.shape;

        // BUG FIX #3: Use .data() (Float32Array) instead of .array() — massively faster
        const flatData = await outputTensor.data();

        // ---- Parse detections based on output shape ----
        let candidates = [];

        if (shape.length === 3 && (shape[1] === 84 || shape[1] === 80)) {
            // Shape: [1, 84, 8400] — columns are boxes
            const numClasses = shape[1] - 4;
            const numBoxes   = shape[2];

            for (let col = 0; col < numBoxes; col++) {
                let maxConf = 0;
                let classId = -1;

                for (let cls = 0; cls < numClasses; cls++) {
                    // Index: (cls+4) * numBoxes + col
                    const conf = flatData[(cls + 4) * numBoxes + col];
                    if (conf > maxConf) {
                        maxConf = conf;
                        classId = cls;
                    }
                }

                if (maxConf > CONF_THRESHOLD) {
                    candidates.push({
                        x:       flatData[0 * numBoxes + col],
                        y:       flatData[1 * numBoxes + col],
                        w:       flatData[2 * numBoxes + col],
                        h:       flatData[3 * numBoxes + col],
                        conf:    maxConf,
                        classId: classId
                    });
                }
            }

        } else if (shape.length === 3 && (shape[2] === 84 || shape[2] === 80)) {
            // Shape: [1, 8400, 84] — rows are boxes
            const numBoxes   = shape[1];
            const numCols    = shape[2];
            const numClasses = numCols - 4;

            for (let row = 0; row < numBoxes; row++) {
                const base = row * numCols;
                let maxConf = 0;
                let classId = -1;

                for (let cls = 0; cls < numClasses; cls++) {
                    const conf = flatData[base + 4 + cls];
                    if (conf > maxConf) {
                        maxConf = conf;
                        classId = cls;
                    }
                }

                if (maxConf > CONF_THRESHOLD) {
                    candidates.push({
                        x:       flatData[base + 0],
                        y:       flatData[base + 1],
                        w:       flatData[base + 2],
                        h:       flatData[base + 3],
                        conf:    maxConf,
                        classId: classId
                    });
                }
            }
        }

        // ---- Non-Maximum Suppression ----
        candidates.sort((a, b) => b.conf - a.conf);
        const finalBoxes = [];

        while (candidates.length > 0) {
            const best = candidates.shift();
            finalBoxes.push(best);
            candidates = candidates.filter(box =>
                box.classId !== best.classId || calculateIOU(best, box) < IOU_THRESHOLD
            );
        }

        // ---- Draw: video frame first ----
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // BUG FIX #5: Mirror the canvas for front cam so it looks natural
        ctx.save();
        if (isFrontCam) {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        // ---- Draw detections ----
        const scaleX = canvas.width  / INPUT_SIZE;
        const scaleY = canvas.height / INPUT_SIZE;

        finalBoxes.forEach(box => {
            const color = getColor(box.classId);
            const label = `${yoloClasses[box.classId] || 'Object'} ${(box.conf * 100).toFixed(0)}%`;

            let { x, y, w, h } = box;

            // Handle un-normalized coordinates (rare but possible)
            if (w <= 2.0 && h <= 2.0) {
                x *= INPUT_SIZE; y *= INPUT_SIZE;
                w *= INPUT_SIZE; h *= INPUT_SIZE;
            }

            let left = (x - w / 2) * scaleX;
            let top  = (y - h / 2) * scaleY;
            const boxW = w * scaleX;
            const boxH = h * scaleY;

            // BUG FIX #5: Mirror box X for front camera
            if (isFrontCam) {
                left = canvas.width - left - boxW;
            }

            // Bounding box
            ctx.shadowColor = color;
            ctx.shadowBlur  = 10;
            ctx.strokeStyle = color;
            ctx.lineWidth   = 2.5;
            ctx.strokeRect(left, top, boxW, boxH);

            // Corner accent markers
            const corner = 14;
            ctx.lineWidth = 4;
            // Top-left
            ctx.beginPath(); ctx.moveTo(left, top + corner); ctx.lineTo(left, top); ctx.lineTo(left + corner, top); ctx.stroke();
            // Top-right
            ctx.beginPath(); ctx.moveTo(left + boxW - corner, top); ctx.lineTo(left + boxW, top); ctx.lineTo(left + boxW, top + corner); ctx.stroke();
            // Bottom-left
            ctx.beginPath(); ctx.moveTo(left, top + boxH - corner); ctx.lineTo(left, top + boxH); ctx.lineTo(left + corner, top + boxH); ctx.stroke();
            // Bottom-right
            ctx.beginPath(); ctx.moveTo(left + boxW - corner, top + boxH); ctx.lineTo(left + boxW, top + boxH); ctx.lineTo(left + boxW, top + boxH - corner); ctx.stroke();

            ctx.shadowBlur = 0;

            // Label background pill
            ctx.font = 'bold 13px "Segoe UI", sans-serif';
            const textW = ctx.measureText(label).width;
            const padX = 8, padY = 4, labelH = 22;
            const lx = Math.max(0, left);
            const ly = top > labelH + 4 ? top - labelH - 4 : top + 4;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(lx, ly, textW + padX * 2, labelH, 6);
            ctx.fill();

            ctx.fillStyle = '#fff';
            ctx.fillText(label, lx + padX, ly + labelH - padY);
        });

        // ---- Update UI ----
        if (finalBoxes.length > 0) {
            setStatus('active', `🟢 ${finalBoxes.length} object${finalBoxes.length > 1 ? 's' : ''} detected`);
            if (detectionCount) detectionCount.textContent = finalBoxes.length;
        } else {
            setStatus('scanning', '📡 Scanning...');
            if (detectionCount) detectionCount.textContent = '0';
        }

    } catch (err) {
        // Silently continue the loop — don't crash on a single bad frame
        console.warn('Frame error:', err.message);
    } finally {
        // BUG FIX #4: Always dispose properly
        inputTensor.dispose();
        if (rawOutput) {
            if (Array.isArray(rawOutput)) {
                rawOutput.forEach(t => t.dispose());
            } else {
                rawOutput.dispose();
            }
        }
    }

    updateFPS();

    // Schedule next frame — no double-wait, just rAF
    animationFrameId = requestAnimationFrame(detectFrame);
}

// ============================================================
// STATUS HELPER
// ============================================================
function setStatus(type, message) {
    statusDiv.className = 'status-pill';
    if (type === 'active')   statusDiv.classList.add('status-active');
    if (type === 'loading')  statusDiv.classList.add('status-loading');
    if (type === 'scanning') statusDiv.classList.add('status-scanning');
    if (type === 'error')    statusDiv.classList.add('status-error');
    statusDiv.innerHTML = message;
}

// ============================================================
// START
// ============================================================
loadModel();
