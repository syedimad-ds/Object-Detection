// ============================================================
//  EDGE AI — Object Detection  |  Final corrected version
//
//  Complete bug list found by reading original working code:
//
//  BUG 1 — Coordinate scaling guard was removed
//    Original code: if (w <= 2 && h <= 2) { x*=640; y*=640; w*=640; h*=640 }
//    This guard handles BOTH output formats:
//      • [0,1]   normalized  → multiply up to pixel space
//      • [0,640] pixel space → leave as-is
//    Without this guard, if model outputs [0,1] coords, every box
//    gets placed at ~0-2 pixels → top-left corner. EXACTLY what was seen.
//
//  BUG 2 — Used .data() (flat Float32Array) instead of .array() (nested)
//    The original indexes as detections[row][col] on a 2D array.
//    Using flat .data() requires different indexing. Both are valid BUT
//    switching between them while keeping old index expressions breaks everything.
//    Solution: use .array() exactly like the original, OR use .data() with
//    CORRECT flat indexing. This version uses .array() to match original exactly.
//
//  BUG 3 — Race condition: two render loops running simultaneously
//    Fixed: renderLoopId tracked, old loop cancelled before new one starts.
//
//  BUG 4 — canvas.width reset wiped context mid-draw
//    Fixed: canvas sized after onloadedmetadata, before play().
//
//  BUG 5 — isFront logic wrong on laptop
//    Laptop webcam IS front-facing — needs mirror on both video AND boxes.
//    Fixed: isFront = true on laptop always.
//
//  BUG 6 — model.execute() try/catch is unreliable
//    Some TF.js graph models only work with executeAsync.
//    Fixed: always use executeAsync (reliable, still fast enough).
// ============================================================

// ── DOM ────────────────────────────────────────────────────
const video          = document.getElementById('webcam');
const canvas         = document.getElementById('output_canvas');
const ctx            = canvas.getContext('2d');
const statusDiv      = document.getElementById('status');
const switchCamBtn   = document.getElementById('switchCamBtn');
const fpsDisplay     = document.getElementById('fpsDisplay');
const detectionCount = document.getElementById('detectionCount');

// ── Session state ──────────────────────────────────────────
let model             = null;
let currentFacingMode = 'environment';
let streamActive      = false;
let inferenceRunning  = false;
let lastBoxes         = [];
let renderLoopId      = null;
let sessionIsFront    = false;   // set once per startWebcam(), used by both loops

// ── FPS ────────────────────────────────────────────────────
let fpsCount    = 0;
let fpsLastTime = performance.now();

// ── Device ─────────────────────────────────────────────────
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
);

if (!isMobile && switchCamBtn) switchCamBtn.style.display = 'none';

const modelPath = isMobile
    ? './yolov8n_web_model/model.json'
    : './yolov8s_web_model/model.json';

// ── Detection constants ────────────────────────────────────
const CONF_THRESHOLD = isMobile ? 0.35 : 0.40;
const IOU_THRESHOLD  = 0.40;
const MAX_DETECTIONS = 15;
const INPUT_W        = 640;
const INPUT_H        = 640;

// ── COCO class names (80) ──────────────────────────────────
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

// ── IOU ────────────────────────────────────────────────────
function iouScore(a, b) {
    const aL = a.x - a.w / 2,  aR = a.x + a.w / 2;
    const aT = a.y - a.h / 2,  aB = a.y + a.h / 2;
    const bL = b.x - b.w / 2,  bR = b.x + b.w / 2;
    const bT = b.y - b.h / 2,  bB = b.y + b.h / 2;
    const iW  = Math.max(0, Math.min(aR, bR) - Math.max(aL, bL));
    const iH  = Math.max(0, Math.min(aB, bB) - Math.max(aT, bT));
    const inter = iW * iH;
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
}

// ── NMS (cross-class) ──────────────────────────────────────
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

// ── Model load ─────────────────────────────────────────────
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

        setStatus('loading', '🔥 Warming up GPU (~5s)...');
        const dummy  = tf.zeros([1, INPUT_H, INPUT_W, 3]);
        const warmup = await model.executeAsync(dummy);
        if (Array.isArray(warmup)) warmup.forEach(t => t.dispose());
        else warmup.dispose();
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
    // Stop current session cleanly
    streamActive     = false;
    inferenceRunning = false;
    lastBoxes        = [];

    // Cancel old render loop before it can fire again
    if (renderLoopId !== null) {
        cancelAnimationFrame(renderLoopId);
        renderLoopId = null;
    }

    // Stop camera tracks
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    // Drain any queued rAF callbacks from the old loop
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

        // Size canvas BEFORE play() — resets context cleanly while nothing is drawing
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;

        await video.play();

        // isFront: laptop webcam = always front-facing (mirror)
        //          mobile rear   = no mirror
        //          mobile front  = mirror
        sessionIsFront = isMobile ? (currentFacingMode === 'user') : true;

        streamActive = true;
        startRenderLoop();
        runInferenceLoop();

    } catch (err) {
        setStatus('error', '❌ Camera denied — check permissions.');
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
//  RENDER LOOP — draws at ~60fps using last known detections
// ============================================================
function startRenderLoop() {
    function frame() {
        if (!streamActive) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (video.readyState >= 2) {
            // Mirror video horizontally for front-facing cameras
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

        // Draw last known bounding boxes
        drawBoxes(lastBoxes);

        // FPS
        fpsCount++;
        const now = performance.now();
        if (now - fpsLastTime >= 1000) {
            if (fpsDisplay) fpsDisplay.textContent =
                Math.round((fpsCount * 1000) / (now - fpsLastTime));
            fpsCount    = 0;
            fpsLastTime = now;
        }

        renderLoopId = requestAnimationFrame(frame);
    }

    renderLoopId = requestAnimationFrame(frame);
}

// ============================================================
//  INFERENCE LOOP — updates lastBoxes as fast as GPU allows
// ============================================================
async function runInferenceLoop() {
    while (streamActive) {
        if (!model || video.readyState < 2 || inferenceRunning) {
            await waitFrame();
            continue;
        }

        inferenceRunning = true;

        try {
            // Build input tensor
            const inputTensor = tf.tidy(() =>
                tf.browser.fromPixels(video)
                    .resizeBilinear([INPUT_H, INPUT_W])
                    .expandDims(0)
                    .toFloat()
                    .div(255.0)
            );

            // executeAsync is reliable for frozen graph models
            const predictions = await model.executeAsync(inputTensor);
            inputTensor.dispose();

            // Normalize to single tensor
            const tensorOutput = Array.isArray(predictions) ? predictions[0] : predictions;
            const shape        = tensorOutput.shape;   // [1, 84, 8400] or [1, 8400, 84]

            // ── BUG 2 FIX: use .array() exactly like original ──
            // .array() returns a nested JS array:
            //   data[0] = shape [84, 8400] → data[0][row][col]
            const data       = await tensorOutput.array();
            const detections = data[0];  // strip batch dim → [84][8400] or [8400][84]

            // Dispose tensors properly
            if (Array.isArray(predictions)) predictions.forEach(t => t.dispose());
            else predictions.dispose();

            // Parse and run NMS
            const candidates = parseDetections(detections, shape);
            lastBoxes = nms(candidates);

            // Update status
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
//  After .array() and stripping batch:
//   • detections[row][col]  if shape was [1, 84, 8400]
//     row = coord/class index (0-83), col = box index (0-8399)
//   • detections[row][col]  if shape was [1, 8400, 84]
//     row = box index (0-8399), col = coord/class index (0-83)
//
//  Detection is by checking shape[1]:
//    84 (or 80 for nano) → [channels, boxes] layout
//    8400                → [boxes, channels] layout
//
//  BUG 1 FIX: coordinate normalization guard
//    If w <= 2 the coords are in [0,1] → scale to [0,640]
//    If w >  2 the coords are already in [0,640] pixel space
// ============================================================
function parseDetections(detections, shape) {
    const candidates = [];

    // shape is [1, dim1, dim2] — check dim1 to determine layout
    const dim1 = shape[1];
    const dim2 = shape[2];

    if (dim1 === 84 || dim1 === 80) {
        // Layout: [1, 84, 8400] — rows=channels, cols=boxes
        const numClasses = dim1 - 4;
        const numBoxes   = dim2;

        for (let col = 0; col < numBoxes; col++) {
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = detections[4 + cls][col];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf > CONF_THRESHOLD && classId >= 0) {
                let x = detections[0][col];
                let y = detections[1][col];
                let w = detections[2][col];
                let h = detections[3][col];
                // BUG 1 FIX: normalize guard
                if (w <= 2 && h <= 2) {
                    x *= INPUT_W; y *= INPUT_H;
                    w *= INPUT_W; h *= INPUT_H;
                }
                candidates.push({ x, y, w, h, conf: maxConf, classId });
            }
        }

    } else if (dim2 === 84 || dim2 === 80) {
        // Layout: [1, 8400, 84] — rows=boxes, cols=channels
        const numBoxes   = dim1;
        const numClasses = dim2 - 4;

        for (let row = 0; row < numBoxes; row++) {
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = detections[row][4 + cls];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf > CONF_THRESHOLD && classId >= 0) {
                let x = detections[row][0];
                let y = detections[row][1];
                let w = detections[row][2];
                let h = detections[row][3];
                // BUG 1 FIX: normalize guard
                if (w <= 2 && h <= 2) {
                    x *= INPUT_W; y *= INPUT_H;
                    w *= INPUT_W; h *= INPUT_H;
                }
                candidates.push({ x, y, w, h, conf: maxConf, classId });
            }
        }

    } else {
        console.warn('Unknown model output shape:', shape);
    }

    return candidates;
}

// ── Draw bounding boxes ────────────────────────────────────
function drawBoxes(boxes) {
    if (!boxes.length) return;

    const scaleX = canvas.width  / INPUT_W;
    const scaleY = canvas.height / INPUT_H;

    boxes.forEach(box => {
        const color = getColor(box.classId);
        const name  = COCO_CLASSES[box.classId] || 'Object';
        const label = `${name} ${Math.round(box.conf * 100)}%`;

        // box.x/y is center in [0,640] space
        let boxW = box.w * scaleX;
        let boxH = box.h * scaleY;
        let left = box.x * scaleX - boxW / 2;
        let top  = box.y * scaleY - boxH / 2;

        // Mirror X for front-facing cameras (must match video mirror)
        if (sessionIsFront) {
            left = canvas.width - left - boxW;
        }

        ctx.save();

        // Main bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2.5;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 8;
        ctx.strokeRect(left, top, boxW, boxH);

        // Corner bracket accents
        const c = Math.max(8, Math.min(18, boxW * 0.15, boxH * 0.15));
        ctx.lineWidth  = 3.5;
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.moveTo(left,           top + c);       ctx.lineTo(left,           top);
        ctx.lineTo(left + c,       top);
        ctx.moveTo(left + boxW - c, top);           ctx.lineTo(left + boxW,    top);
        ctx.lineTo(left + boxW,    top + c);
        ctx.moveTo(left,           top + boxH - c); ctx.lineTo(left,           top + boxH);
        ctx.lineTo(left + c,       top + boxH);
        ctx.moveTo(left + boxW - c, top + boxH);    ctx.lineTo(left + boxW,    top + boxH);
        ctx.lineTo(left + boxW,    top + boxH - c);
        ctx.stroke();

        // Label background pill
        ctx.font = 'bold 13px "Outfit", sans-serif';
        const tw  = ctx.measureText(label).width;
        const ph  = 22, px = 8;
        const lx  = Math.max(0, Math.min(left, canvas.width - tw - px * 2 - 1));
        const ly  = top > ph + 4 ? top - ph - 4 : top + 4;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + px * 2, ph, 6);
        ctx.fill();

        // Label text
        ctx.fillStyle = '#000000';
        ctx.fillText(label, lx + px, ly + ph - 6);

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
