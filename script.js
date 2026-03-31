// ============================================================
//  EDGE AI — Object Detection Engine  (fully corrected)
//
//  Root causes fixed in this version:
//
//  FIX A — Shape parsing was INVERTED:
//    model.execute() on this TF.js graph model returns shape
//    [1, 84, 8400].  shape[1]=84, shape[2]=8400.
//    The old condition (shape[1] > shape[2]) → 84 > 8400 → FALSE
//    so it fell into the WRONG branch, treating data as [1,8400,84].
//    That read coordinates as class scores and vice-versa,
//    producing "Object 615 1%" and boxes in the wrong corner.
//    Fix: detect layout by checking which dim equals 8400 explicitly,
//    with a safe fallback.
//
//  FIX B — isFront was false on laptop:
//    Laptop webcams are FRONT-facing. Setting isFront=false meant
//    no mirror on the video AND no coordinate mirror on boxes,
//    so everything was backwards. Fix: isFront = !isMobile on
//    laptop (always mirror), mobile only mirrors on 'user' facingMode.
//
//  FIX C — Canvas race condition (previous fix retained):
//    renderLoopId tracked and cancelled before new session starts.
//    Canvas sized before video.play() so it's never 0.
// ============================================================

// ── DOM refs ───────────────────────────────────────────────
const video          = document.getElementById('webcam');
const canvas         = document.getElementById('output_canvas');
const ctx            = canvas.getContext('2d');
const statusDiv      = document.getElementById('status');
const switchCamBtn   = document.getElementById('switchCamBtn');
const fpsDisplay     = document.getElementById('fpsDisplay');
const detectionCount = document.getElementById('detectionCount');

// ── State ──────────────────────────────────────────────────
let model             = null;
let currentFacingMode = 'environment';
let streamActive      = false;
let inferenceRunning  = false;
let lastBoxes         = [];
let renderLoopId      = null;

// ── FPS ────────────────────────────────────────────────────
let fpsFrameCount = 0;
let fpsLastTime   = performance.now();

// ── Device detection ───────────────────────────────────────
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

if (!isMobile && switchCamBtn) switchCamBtn.style.display = 'none';

const modelPath = isMobile
    ? './yolov8n_web_model/model.json'
    : './yolov8s_web_model/model.json';

// ── Tuning ─────────────────────────────────────────────────
const CONF_THRESHOLD = isMobile ? 0.40 : 0.45;
const IOU_THRESHOLD  = 0.35;
const MAX_DETECTIONS = 12;
const INPUT_SIZE     = 640;
const NUM_CLASSES    = 80;   // COCO — hard-coded, no guessing from shape

// ── Class names (80 COCO) ──────────────────────────────────
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
];   // exactly 80 entries

const CLASS_COLORS = [
    '#FF3838','#FF9D97','#FF701F','#FFB21D','#CFD231','#48F90A','#92CC17','#3DDB86',
    '#1A9334','#00D4BB','#2C99A8','#00C2FF','#344593','#6473FF','#0018EC','#8438FF',
    '#520085','#CB38FF','#FF95C8','#FF37C7'
];

const getColor = id => CLASS_COLORS[Math.abs(id) % CLASS_COLORS.length];

// ── IOU ────────────────────────────────────────────────────
function iouScore(a, b) {
    const aL = a.cx - a.w / 2,  aR = a.cx + a.w / 2;
    const aT = a.cy - a.h / 2,  aB = a.cy + a.h / 2;
    const bL = b.cx - b.w / 2,  bR = b.cx + b.w / 2;
    const bT = b.cy - b.h / 2,  bB = b.cy + b.h / 2;
    const iW = Math.max(0, Math.min(aR, bR) - Math.max(aL, bL));
    const iH = Math.max(0, Math.min(aB, bB) - Math.max(aT, bT));
    const inter = iW * iH;
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
}

// ── Cross-class NMS ────────────────────────────────────────
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
            if (candidates[i].classId === candidates[j].classId ||
                iouScore(candidates[i], candidates[j]) > IOU_THRESHOLD) {
                suppressed[j] = 1;
            }
        }
    }
    return keep;
}

// ── Model loading ──────────────────────────────────────────
async function loadModel() {
    try {
        setStatus('loading', '⏳ Initializing WebGL...');
        await tf.setBackend('webgl');
        await tf.ready();
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_PACK', true);
        tf.env().set('WEBGL_CONV_IM2COL', true);

        setStatus('loading', '⏳ Downloading model...');
        model = await tf.loadGraphModel(modelPath);

        // Warm-up: compiles GPU shaders so first real frame is instant
        setStatus('loading', '🔥 Warming up GPU (~5s)...');
        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        let warmOut;
        try   { warmOut = model.execute(dummy); }
        catch { warmOut = await model.executeAsync(dummy); }
        if (Array.isArray(warmOut)) warmOut.forEach(t => t.dispose());
        else warmOut.dispose();
        dummy.dispose();

        const label = isMobile ? 'Nano · Fast Mode' : 'Small · High Accuracy';
        setStatus('active', `✅ System Active — ${label}`);
        await startWebcam();

    } catch (err) {
        setStatus('error', `❌ Load failed: ${err.message}`);
        console.error(err);
    }
}

// ── Webcam ─────────────────────────────────────────────────
async function startWebcam() {
    // Stop old loops before touching the stream
    streamActive     = false;
    inferenceRunning = false;
    lastBoxes        = [];

    if (renderLoopId !== null) {
        cancelAnimationFrame(renderLoopId);
        renderLoopId = null;
    }

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    // Let any queued rAF callbacks drain before starting fresh
    await waitFrame();
    await waitFrame();

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: isMobile ? currentFacingMode : 'user',
                width:      { ideal: 1280 },
                height:     { ideal: 720  },
                frameRate:  { ideal: 30   }
            }
        });

        video.srcObject = stream;
        await new Promise(resolve => { video.onloadedmetadata = resolve; });

        // Set canvas size while dimensions are known, before play()
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;

        await video.play();

        // ── FIX B: isFront logic ──────────────────────────
        // Laptop: always front-facing (mirror video + mirror boxes)
        // Mobile: mirror only when using selfie cam
        const isFront = isMobile
            ? (currentFacingMode === 'user')
            : true;   // laptop webcam is always front-facing

        streamActive = true;
        startRenderLoop(isFront);
        runInferenceLoop(isFront);

    } catch (err) {
        setStatus('error', '❌ Camera denied — check browser permissions.');
        console.error(err);
    }
}

// Camera switch (mobile only)
if (switchCamBtn) {
    switchCamBtn.addEventListener('click', async () => {
        if (!isMobile) return;
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        switchCamBtn.disabled = true;
        await startWebcam();
        switchCamBtn.disabled = false;
    });
}

// ============================================================
//  RENDER LOOP — 60 fps, never waits for inference
// ============================================================
function startRenderLoop(isFront) {
    function frame() {
        if (!streamActive) return;

        // Always clear to avoid ghost frames if video stalls
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (video.readyState >= 2) {
            if (isFront) {
                ctx.save();
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
            } else {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            }
        }

        drawBoxes(lastBoxes, isFront);

        // FPS
        fpsFrameCount++;
        const now = performance.now();
        if (now - fpsLastTime >= 1000) {
            if (fpsDisplay) fpsDisplay.textContent =
                Math.round((fpsFrameCount * 1000) / (now - fpsLastTime));
            fpsFrameCount = 0;
            fpsLastTime   = now;
        }

        renderLoopId = requestAnimationFrame(frame);
    }
    renderLoopId = requestAnimationFrame(frame);
}

// ============================================================
//  INFERENCE LOOP — runs as fast as GPU allows
// ============================================================
async function runInferenceLoop(isFront) {
    while (streamActive) {
        if (!model || video.readyState < 2 || inferenceRunning) {
            await waitFrame();
            continue;
        }

        inferenceRunning = true;

        try {
            const inputTensor = tf.tidy(() =>
                tf.browser.fromPixels(video)
                    .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
                    .expandDims(0)
                    .toFloat()
                    .div(255.0)
            );

            let rawOut;
            try   { rawOut = model.execute(inputTensor); }
            catch { rawOut = await model.executeAsync(inputTensor); }
            inputTensor.dispose();

            const outTensor = Array.isArray(rawOut) ? rawOut[0] : rawOut;
            const shape     = outTensor.shape;        // e.g. [1, 84, 8400]
            const flatData  = await outTensor.data(); // Float32Array

            if (Array.isArray(rawOut)) rawOut.forEach(t => t.dispose());
            else rawOut.dispose();

            // ── FIX A: parse with explicit layout detection ──
            const candidates = parseDetections(flatData, shape);
            lastBoxes = nms(candidates);

            if (lastBoxes.length > 0) {
                setStatus('active',
                    `🟢 ${lastBoxes.length} object${lastBoxes.length !== 1 ? 's' : ''} detected`);
                if (detectionCount) detectionCount.textContent = lastBoxes.length;
            } else {
                setStatus('scanning', '📡 Scanning...');
                if (detectionCount) detectionCount.textContent = '0';
            }

        } catch (err) {
            console.warn('Inference error:', err.message);
        }

        inferenceRunning = false;
        await waitFrame();
    }
}

// ============================================================
//  PARSE DETECTIONS
//
//  YOLOv8 TF.js output is ALWAYS [1, 84, 8400]:
//    - dim 1 = 84  = 4 box coords + 80 class scores (rows)
//    - dim 2 = 8400 = number of candidate boxes (columns)
//
//  Memory layout (row-major, batch=1 squeezed):
//    index = row * 8400 + col
//    row 0 → cx for all 8400 boxes
//    row 1 → cy for all 8400 boxes
//    row 2 → w  for all 8400 boxes
//    row 3 → h  for all 8400 boxes
//    row 4..83 → class scores for all 8400 boxes
//
//  The old code used (shape[1] > shape[2]) which is 84>8400=false
//  so it fell into the WRONG branch every time.
//  We now detect layout by checking which dimension equals
//  NUM_BOXES (8400) and which equals NUM_CLASSES+4 (84).
// ============================================================
function parseDetections(flatData, shape) {
    const candidates = [];

    if (!shape || shape.length < 2) return candidates;

    // Strip leading batch dimension if present
    const dims = shape.length === 3 ? [shape[1], shape[2]] : [shape[0], shape[1]];

    // Figure out which dimension is "channels" (84) and which is "boxes" (8400)
    // channels = NUM_CLASSES + 4 = 84
    // We identify by size: the smaller dim is channels, larger is boxes
    // Both must be sane values — guard against weird shapes
    const CHANNELS = NUM_CLASSES + 4;  // 84

    let numChannels, numBoxes;

    if (dims[0] === CHANNELS) {
        // Layout [84, 8400] — standard YOLOv8 TF.js output
        numChannels = dims[0];
        numBoxes    = dims[1];
    } else if (dims[1] === CHANNELS) {
        // Layout [8400, 84] — transposed
        numBoxes    = dims[0];
        numChannels = dims[1];
    } else {
        // Unexpected shape — try to recover by treating smaller as channels
        if (dims[0] < dims[1]) {
            numChannels = dims[0];
            numBoxes    = dims[1];
        } else {
            numBoxes    = dims[0];
            numChannels = dims[1];
        }
        console.warn('Unexpected output shape:', shape, '— guessing layout');
    }

    const numClasses = numChannels - 4;  // should be 80

    // Safety check: classId must stay within COCO_CLASSES array
    if (numClasses <= 0 || numClasses > 200) {
        console.error('Parsed numClasses is invalid:', numClasses, 'from shape:', shape);
        return candidates;
    }

    if (dims[0] === CHANNELS || (dims[0] < dims[1] && dims[0] !== numBoxes)) {
        // ── Layout [channels, boxes] = [84, 8400] ──────────
        // index formula: flatData[row * numBoxes + col]
        for (let col = 0; col < numBoxes; col++) {
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = flatData[(4 + cls) * numBoxes + col];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf >= CONF_THRESHOLD && classId >= 0 && classId < COCO_CLASSES.length) {
                candidates.push({
                    cx:      flatData[0 * numBoxes + col],
                    cy:      flatData[1 * numBoxes + col],
                    w:       flatData[2 * numBoxes + col],
                    h:       flatData[3 * numBoxes + col],
                    conf:    maxConf,
                    classId: classId
                });
            }
        }
    } else {
        // ── Layout [boxes, channels] = [8400, 84] ──────────
        // index formula: flatData[row * numChannels + col]
        for (let row = 0; row < numBoxes; row++) {
            const base = row * numChannels;
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = flatData[base + 4 + cls];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf >= CONF_THRESHOLD && classId >= 0 && classId < COCO_CLASSES.length) {
                candidates.push({
                    cx:      flatData[base + 0],
                    cy:      flatData[base + 1],
                    w:       flatData[base + 2],
                    h:       flatData[base + 3],
                    conf:    maxConf,
                    classId: classId
                });
            }
        }
    }

    return candidates;
}

// ── Draw boxes ─────────────────────────────────────────────
function drawBoxes(boxes, isFront) {
    if (!boxes.length) return;

    const sx = canvas.width  / INPUT_SIZE;
    const sy = canvas.height / INPUT_SIZE;

    boxes.forEach(box => {
        const color = getColor(box.classId);
        const name  = (box.classId >= 0 && box.classId < COCO_CLASSES.length)
            ? COCO_CLASSES[box.classId]
            : 'Object';
        const label = `${name} ${Math.round(box.conf * 100)}%`;

        // box coords are in [0, INPUT_SIZE] space (cx, cy, w, h)
        let { cx, cy, w, h } = box;

        // Scale to canvas pixels
        let left = (cx - w / 2) * sx;
        let top  = (cy - h / 2) * sy;
        const bw = w * sx;
        const bh = h * sy;

        // Mirror X coordinate for front camera
        if (isFront) left = canvas.width - left - bw;

        // Clamp to canvas bounds
        left = Math.max(0, Math.min(left, canvas.width  - 1));
        top  = Math.max(0, Math.min(top,  canvas.height - 1));

        ctx.save();

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 6;
        ctx.strokeRect(left, top, bw, bh);

        // Corner brackets (tactical style)
        const c = Math.max(6, Math.min(16, bw * 0.15, bh * 0.15));
        ctx.lineWidth  = 3;
        ctx.shadowBlur = 0;
        ctx.beginPath();
        // top-left
        ctx.moveTo(left,          top + c);      ctx.lineTo(left,          top);
        ctx.lineTo(left + c,      top);
        // top-right
        ctx.moveTo(left + bw - c, top);          ctx.lineTo(left + bw,     top);
        ctx.lineTo(left + bw,     top + c);
        // bottom-left
        ctx.moveTo(left,          top + bh - c); ctx.lineTo(left,          top + bh);
        ctx.lineTo(left + c,      top + bh);
        // bottom-right
        ctx.moveTo(left + bw - c, top + bh);     ctx.lineTo(left + bw,     top + bh);
        ctx.lineTo(left + bw,     top + bh - c);
        ctx.stroke();

        // Label pill
        ctx.font = 'bold 12px "Outfit", sans-serif';
        const tw = ctx.measureText(label).width;
        const ph = 20, px = 7;
        // Keep label inside canvas horizontally
        const lx = Math.max(0, Math.min(left, canvas.width  - tw - px * 2 - 1));
        const ly = top > ph + 4 ? top - ph - 3 : top + 3;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + px * 2, ph, 5);
        ctx.fill();

        ctx.fillStyle = '#000';
        ctx.fillText(label, lx + px, ly + ph - 5);

        ctx.restore();
    });
}

// ── Helpers ────────────────────────────────────────────────
const waitFrame = () => new Promise(r => requestAnimationFrame(r));

function setStatus(type, msg) {
    statusDiv.className = `status-pill status-${type}`;
    statusDiv.innerHTML = msg;
}

// ── Boot ───────────────────────────────────────────────────
loadModel();
