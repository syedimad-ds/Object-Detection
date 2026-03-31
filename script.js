// ============================================================
//  UNIVERSAL EDGE AI — Object Detection (Max FPS & Fallback)
// ============================================================

const video          = document.getElementById('webcam');
const canvas         = document.getElementById('output_canvas');
const ctx            = canvas.getContext('2d', { alpha: false }); 
const statusDiv      = document.getElementById('status');
const switchCamBtn   = document.getElementById('switchCamBtn');
const hdModeBtn      = document.getElementById('hdModeBtn');
const fpsDisplay     = document.getElementById('fpsDisplay');
const detectionCount = document.getElementById('detectionCount');
const modelIndicator = document.getElementById('modelIndicator');

// ── Session State ──────────────────────────────────────────
let model             = null;
let currentFacingMode = 'environment';
let streamActive      = false;
let inferenceRunning  = false;
let lastBoxes         = [];
let renderLoopId      = null;
let sessionIsFront    = false;
let isHDMode          = false; // Default to Nano for universal support
let backendName       = 'unknown';

let fpsCount    = 0;
let fpsLastTime = performance.now();

const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
if (!isMobile && switchCamBtn) switchCamBtn.style.display = 'none';

const modelPathNano  = './yolov8n_web_model/model.json';
const modelPathSmall = './yolov8s_web_model/model.json';

const CONF_THRESHOLD = isMobile ? 0.35 : 0.40;
const IOU_THRESHOLD  = 0.45;
const MAX_DETECTIONS = 15;
const INPUT_W        = 640;
const INPUT_H        = 640;

const COCO_CLASSES = [
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

const CLASS_COLORS = [
    '#FF3838','#FF9D97','#FF701F','#FFB21D','#CFD231','#48F90A','#92CC17','#3DDB86',
    '#1A9334','#00D4BB','#2C99A8','#00C2FF','#344593','#6473FF','#0018EC','#8438FF',
    '#520085','#CB38FF','#FF95C8','#FF37C7'
];
const getColor = id => CLASS_COLORS[id % CLASS_COLORS.length];

// ── Math Helpers ───────────────────────────────────────────
function iouScore(a, b) {
    const aL = a.x - a.w / 2, aR = a.x + a.w / 2, aT = a.y - a.h / 2, aB = a.y + a.h / 2;
    const bL = b.x - b.w / 2, bR = b.x + b.w / 2, bT = b.y - b.h / 2, bB = b.y + b.h / 2;
    const iW = Math.max(0, Math.min(aR, bR) - Math.max(aL, bL));
    const iH = Math.max(0, Math.min(aB, bB) - Math.max(aT, bT));
    const inter = iW * iH;
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
}

function nms(candidates) {
    candidates.sort((a, b) => b.conf - a.conf);
    const suppressed = new Uint8Array(candidates.length);
    const keep = [];
    for (let i = 0; i < candidates.length; i++) {
        if (suppressed[i]) continue;
        keep.push(candidates[i]);
        if (keep.length >= MAX_DETECTIONS) break;
        for (let j = i + 1; j < candidates.length; j++) {
            if (suppressed[j]) continue;
            if (candidates[i].classId === candidates[j].classId || iouScore(candidates[i], candidates[j]) > IOU_THRESHOLD) {
                suppressed[j] = 1;
            }
        }
    }
    return keep;
}

// ── Fallback Backend System ────────────────────────────────
async function initBackend() {
    try {
        await tf.setBackend('webgl');
        if (tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
            tf.env().set('WEBGL_FORCE_F16_PIPELINES', true);
        }
        tf.env().set('WEBGL_PACK', true);
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        backendName = 'WebGL (GPU)';
    } catch (err) {
        console.warn('WebGL failed. Trying WASM...');
        try {
            await tf.setBackend('wasm');
            backendName = 'WASM (CPU)';
        } catch (err2) {
            await tf.setBackend('cpu');
            backendName = 'Standard CPU';
        }
    }
    await tf.ready();
    console.log(`✅ Backend Active: ${backendName}`);
}

// ── Model Management ───────────────────────────────────────
async function loadModel() {
    try {
        setStatus('loading', `⏳ Initializing ${backendName}...`);
        
        // Clear memory if a model is already loaded (for HD switching)
        if (model) {
            model.dispose();
            model = null;
        }

        const path = isHDMode ? modelPathSmall : modelPathNano;
        setStatus('loading', `⏳ Downloading ${isHDMode ? 'Small' : 'Nano'} model...`);
        
        model = await tf.loadGraphModel(path);

        setStatus('loading', '🔥 Warming up Engine...');
        const dummy = tf.zeros([1, INPUT_H, INPUT_W, 3]);
        const warmup = await model.executeAsync(dummy);
        if (Array.isArray(warmup)) warmup.forEach(t => t.dispose());
        else warmup.dispose();
        dummy.dispose();

        modelIndicator.textContent = `Real-Time Object Detection (${isHDMode ? 'Small' : 'Nano'}) [${backendName}]`;
        setStatus('active', `✅ System Active`);
        
        if (!streamActive) await startWebcam();
    } catch (err) {
        setStatus('error', `❌ Load failed: ${err.message}`);
        console.error(err);
    }
}

// ── UI Listeners ───────────────────────────────────────────
if (hdModeBtn) {
    hdModeBtn.addEventListener('click', async () => {
        isHDMode = !isHDMode;
        hdModeBtn.classList.toggle('active', isHDMode);
        streamActive = false; // pause briefly
        await loadModel();
        streamActive = true;
        runInferenceLoop(); // restart inference
    });
}

// ── Webcam ─────────────────────────────────────────────────
async function startWebcam() {
    streamActive = false;
    inferenceRunning = false;
    lastBoxes = [];

    if (renderLoopId !== null) cancelAnimationFrame(renderLoopId);
    if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: isMobile ? currentFacingMode : 'user',
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            }
        });

        video.srcObject = stream;
        await new Promise(resolve => { video.onloadedmetadata = resolve; });

        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;

        await video.play();
        sessionIsFront = isMobile ? (currentFacingMode === 'user') : true;
        streamActive = true;
        
        startRenderLoop();
        runInferenceLoop();
    } catch (err) {
        setStatus('error', '❌ Camera denied — check permissions.');
    }
}

if (switchCamBtn) {
    switchCamBtn.addEventListener('click', async () => {
        if (!isMobile) return;
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        switchCamBtn.disabled = true;
        await startWebcam();
        switchCamBtn.disabled = false;
    });
}

// ── Render Loop ────────────────────────────────────────────
function startRenderLoop() {
    function frame() {
        if (!streamActive) return;

        if (video.readyState >= 2) {
            if (sessionIsFront) {
                ctx.save();
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
            } else {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            }
        }

        drawBoxes(lastBoxes);

        fpsCount++;
        const now = performance.now();
        if (now - fpsLastTime >= 1000) {
            if (fpsDisplay) fpsDisplay.textContent = Math.round((fpsCount * 1000) / (now - fpsLastTime));
            fpsCount = 0;
            fpsLastTime = now;
        }

        renderLoopId = requestAnimationFrame(frame);
    }
    renderLoopId = requestAnimationFrame(frame);
}

// ── Fast Inference Loop ────────────────────────────────────
async function runInferenceLoop() {
    while (streamActive) {
        if (!model || video.readyState < 2 || inferenceRunning) {
            await waitFrame();
            continue;
        }

        inferenceRunning = true;

        try {
            // Tidy wraps all tensor math to prevent memory leaks
            const predictions = await tf.tidy(() => {
                const inputTensor = tf.browser.fromPixels(video)
                    .resizeBilinear([INPUT_H, INPUT_W])
                    .expandDims(0)
                    .toFloat()
                    .div(255.0);
                return model.execute(inputTensor); 
            });

            const tensorOutput = Array.isArray(predictions) ? predictions[0] : predictions;
            const shape = tensorOutput.shape;

            // Ultimate speed: flat buffer read
            const flatData = await tensorOutput.data();
            
            if (Array.isArray(predictions)) predictions.forEach(t => t.dispose());
            else predictions.dispose();

            const candidates = parseFlatDetections(flatData, shape);
            lastBoxes = nms(candidates);

            if (lastBoxes.length > 0) {
                setStatus('active', `🟢 ${lastBoxes.length} object${lastBoxes.length !== 1 ? 's' : ''} detected`);
                if (detectionCount) detectionCount.textContent = lastBoxes.length;
            } else {
                setStatus('scanning', '📡 Scanning...');
                if (detectionCount) detectionCount.textContent = '0';
            }

        } catch (err) {
            console.warn('Inference error:', err.message);
        }

        inferenceRunning = false;
        
        // ❄️ THERMAL COOLING DELAY ❄️
        // Prevents phone/laptop from freezing by yielding to main thread
        await new Promise(resolve => setTimeout(resolve, 20)); 
    }
}

// ── Flat Array Parser ──────────────────────────────────────
function parseFlatDetections(data, shape) {
    const candidates = [];
    const dim1 = shape[1];
    const dim2 = shape[2];

    if (dim1 === 84 || dim1 === 80) {
        const numClasses = dim1 - 4;
        const numBoxes = dim2;

        for (let col = 0; col < numBoxes; col++) {
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = data[(4 + cls) * numBoxes + col];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf > CONF_THRESHOLD && classId >= 0) {
                let x = data[0 * numBoxes + col];
                let y = data[1 * numBoxes + col];
                let w = data[2 * numBoxes + col];
                let h = data[3 * numBoxes + col];
                
                if (w <= 2 && h <= 2) { x *= INPUT_W; y *= INPUT_H; w *= INPUT_W; h *= INPUT_H; }
                candidates.push({ x, y, w, h, conf: maxConf, classId });
            }
        }
    } else if (dim2 === 84 || dim2 === 80) {
        const numBoxes = dim1;
        const numClasses = dim2 - 4;

        for (let row = 0; row < numBoxes; row++) {
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = data[row * dim2 + (4 + cls)];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf > CONF_THRESHOLD && classId >= 0) {
                let x = data[row * dim2 + 0];
                let y = data[row * dim2 + 1];
                let w = data[row * dim2 + 2];
                let h = data[row * dim2 + 3];
                
                if (w <= 2 && h <= 2) { x *= INPUT_W; y *= INPUT_H; w *= INPUT_W; h *= INPUT_H; }
                candidates.push({ x, y, w, h, conf: maxConf, classId });
            }
        }
    }
    return candidates;
}

// ── UI Drawing ─────────────────────────────────────────────
function drawBoxes(boxes) {
    if (!boxes.length) return;

    const scaleX = canvas.width  / INPUT_W;
    const scaleY = canvas.height / INPUT_H;

    boxes.forEach(box => {
        const color = getColor(box.classId);
        const name  = COCO_CLASSES[box.classId] || 'Object';
        const label = `${name} ${Math.round(box.conf * 100)}%`;

        let boxW = box.w * scaleX;
        let boxH = box.h * scaleY;
        let left = box.x * scaleX - boxW / 2;
        let top  = box.y * scaleY - boxH / 2;

        if (sessionIsFront) { left = canvas.width - left - boxW; }

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2.5;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 8;
        ctx.strokeRect(left, top, boxW, boxH);

        const c = Math.max(8, Math.min(18, boxW * 0.15, boxH * 0.15));
        ctx.lineWidth  = 3.5;
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.moveTo(left, top + c); ctx.lineTo(left, top); ctx.lineTo(left + c, top);
        ctx.moveTo(left + boxW - c, top); ctx.lineTo(left + boxW, top); ctx.lineTo(left + boxW, top + c);
        ctx.moveTo(left, top + boxH - c); ctx.lineTo(left, top + boxH); ctx.lineTo(left + c, top + boxH);
        ctx.moveTo(left + boxW - c, top + boxH); ctx.lineTo(left + boxW, top + boxH); ctx.lineTo(left + boxW, top + boxH - c);
        ctx.stroke();

        ctx.font = 'bold 13px "Outfit", sans-serif';
        const tw  = ctx.measureText(label).width;
        const ph  = 22, px = 8;
        const lx  = Math.max(0, Math.min(left, canvas.width - tw - px * 2 - 1));
        const ly  = top > ph + 4 ? top - ph - 4 : top + 4;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + px * 2, ph, 6);
        ctx.fill();

        ctx.fillStyle = '#000000';
        ctx.fillText(label, lx + px, ly + ph - 6);
        ctx.restore();
    });
}

const waitFrame = () => new Promise(r => requestAnimationFrame(r));
function setStatus(type, msg) {
    statusDiv.className = `status-pill status-${type}`;
    statusDiv.innerHTML = msg;
}

// ── Boot System ────────────────────────────────────────────
initBackend().then(() => loadModel());