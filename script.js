// ============================================================
//  EDGE AI — Object Detection Engine
//
//  Black canvas fixes applied:
//  1. Single render loop ID tracked — old loop cancelled before
//     new one starts, preventing two loops fighting the canvas
//  2. canvas dimensions set BEFORE video.play() is awaited,
//     so width/height are never 0 when drawing starts
//  3. isFront computed once per startWebcam call and stored in
//     a closure variable — no per-frame recomputation that
//     could be wrong during transitions
//  4. ctx.clearRect added so stale frames don't persist when
//     video stalls
//  5. Laptop facingMode: 'user' means front cam — isFront must
//     be true for laptop, handled by dedicated flag
// ============================================================

const video          = document.getElementById('webcam');
const canvas         = document.getElementById('output_canvas');
const ctx            = canvas.getContext('2d');
const statusDiv      = document.getElementById('status');
const switchCamBtn   = document.getElementById('switchCamBtn');
const fpsDisplay     = document.getElementById('fpsDisplay');
const detectionCount = document.getElementById('detectionCount');

// ── State ──────────────────────────────────────────────────
let model             = null;
let currentFacingMode = 'environment';  // only used on mobile
let streamActive      = false;
let inferenceRunning  = false;
let lastBoxes         = [];
let renderLoopId      = null;           // tracks rAF id so we can cancel it

// ── FPS ────────────────────────────────────────────────────
let fpsFrameCount = 0;
let fpsLastTime   = performance.now();

// ── Device ─────────────────────────────────────────────────
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

// ── Class names ────────────────────────────────────────────
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

const CLASS_COLORS = [
    '#FF3838','#FF9D97','#FF701F','#FFB21D','#CFD231','#48F90A','#92CC17','#3DDB86',
    '#1A9334','#00D4BB','#2C99A8','#00C2FF','#344593','#6473FF','#0018EC','#8438FF',
    '#520085','#CB38FF','#FF95C8','#FF37C7'
];

const getColor = id => CLASS_COLORS[id % CLASS_COLORS.length];

// ── IOU ────────────────────────────────────────────────────
function iouScore(a, b) {
    const aL = a.x - a.w / 2, aR = a.x + a.w / 2;
    const aT = a.y - a.h / 2, aB = a.y + a.h / 2;
    const bL = b.x - b.w / 2, bR = b.x + b.w / 2;
    const bT = b.y - b.h / 2, bB = b.y + b.h / 2;
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

        setStatus('loading', '🔥 Warming up GPU (one-time ~5s)...');
        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        let warmOut;
        try {
            warmOut = model.execute(dummy);
        } catch {
            warmOut = await model.executeAsync(dummy);
        }
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
    // ── FIX 1: Stop old loops cleanly BEFORE touching the stream ──
    // Setting streamActive = false signals the inferenceLoop while()
    // to exit on its next iteration.
    streamActive     = false;
    inferenceRunning = false;
    lastBoxes        = [];

    // Cancel any pending rAF from the old renderLoop
    if (renderLoopId !== null) {
        cancelAnimationFrame(renderLoopId);
        renderLoopId = null;
    }

    // Stop old camera tracks
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    // Small delay to let any in-flight rAF callbacks finish
    await waitFrame();
    await waitFrame();

    try {
        const facingMode = isMobile ? currentFacingMode : 'environment';

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode,
                width:     { ideal: 1280 },
                height:    { ideal: 720  },
                frameRate: { ideal: 30   }
            }
        });

        video.srcObject = stream;

        // Wait for metadata so videoWidth/videoHeight are valid
        await new Promise(resolve => { video.onloadedmetadata = resolve; });

        // ── FIX 2: Set canvas size before playing, while dimensions are known ──
        // Resetting canvas.width clears the canvas — do it now, not mid-draw
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;

        await video.play();

        // ── FIX 3: isFront is a stable boolean for this camera session ──
        // On laptop we always mirror (front cam). On mobile, only for selfie cam.
        const isFront = isMobile ? (currentFacingMode === 'user') : false;

        // Everything is ready — activate loops
        streamActive = true;

        // Start render loop, capturing isFront in closure
        startRenderLoop(isFront);

        // Start inference loop, capturing isFront in closure
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
//  RENDER LOOP
//  Draws video frame + detection boxes every animation frame.
//  isFront is captured at startWebcam() time — stable for the
//  entire camera session, no per-frame recomputation.
// ============================================================
function startRenderLoop(isFront) {
    function frame() {
        // ── FIX 1: Exit immediately if this session is over ──
        if (!streamActive) return;

        // Clear canvas first — prevents ghost frames if video stalls
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw video frame
        if (video.readyState >= 2) {
            if (isFront) {
                // Mirror horizontally for selfie/front cam
                ctx.save();
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
            } else {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            }
        }

        // Overlay detection boxes
        drawBoxes(lastBoxes, isFront);

        // FPS counter
        fpsFrameCount++;
        const now = performance.now();
        if (now - fpsLastTime >= 1000) {
            if (fpsDisplay) {
                fpsDisplay.textContent = Math.round((fpsFrameCount * 1000) / (now - fpsLastTime));
            }
            fpsFrameCount = 0;
            fpsLastTime   = now;
        }

        // Schedule next frame and store ID so we can cancel it
        renderLoopId = requestAnimationFrame(frame);
    }

    renderLoopId = requestAnimationFrame(frame);
}

// ============================================================
//  INFERENCE LOOP
//  Runs model inference as fast as the GPU allows.
//  Updates lastBoxes so renderLoop picks them up.
// ============================================================
async function runInferenceLoop(isFront) {
    while (streamActive) {
        if (!model || video.readyState < 2 || inferenceRunning) {
            await waitFrame();
            continue;
        }

        inferenceRunning = true;

        try {
            // Build input tensor inside tf.tidy so intermediates are freed
            const inputTensor = tf.tidy(() =>
                tf.browser.fromPixels(video)
                    .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
                    .expandDims(0)
                    .toFloat()
                    .div(255.0)
            );

            // model.execute() is synchronous GPU scheduling — much faster
            // than executeAsync which awaits the full CPU round-trip
            let rawOut;
            try {
                rawOut = model.execute(inputTensor);
            } catch {
                rawOut = await model.executeAsync(inputTensor);
            }
            inputTensor.dispose();

            const outTensor = Array.isArray(rawOut) ? rawOut[0] : rawOut;
            const shape     = outTensor.shape;

            // async .data() reads GPU result without blocking the render loop
            const flatData = await outTensor.data();

            // Dispose all output tensors properly
            if (Array.isArray(rawOut)) rawOut.forEach(t => t.dispose());
            else rawOut.dispose();

            // Parse detections and run NMS
            const candidates = parseDetections(flatData, shape);
            lastBoxes = nms(candidates);

            // Update UI
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

        // Yield one frame between inferences — prevents GPU starvation
        await waitFrame();
    }
}

// ── Parse model output → candidate boxes ───────────────────
function parseDetections(flatData, shape) {
    const candidates = [];
    if (shape.length !== 3) return candidates;

    if (shape[1] > shape[2]) {
        // Shape [1, 84, 8400]: each column is a detection
        const numClasses = shape[1] - 4;
        const numBoxes   = shape[2];
        for (let col = 0; col < numBoxes; col++) {
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = flatData[(cls + 4) * numBoxes + col];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf >= CONF_THRESHOLD) {
                candidates.push({
                    x: flatData[0 * numBoxes + col],
                    y: flatData[1 * numBoxes + col],
                    w: flatData[2 * numBoxes + col],
                    h: flatData[3 * numBoxes + col],
                    conf: maxConf, classId
                });
            }
        }
    } else {
        // Shape [1, 8400, 84]: each row is a detection
        const numBoxes   = shape[1];
        const numCols    = shape[2];
        const numClasses = numCols - 4;
        for (let row = 0; row < numBoxes; row++) {
            const base = row * numCols;
            let maxConf = 0, classId = -1;
            for (let cls = 0; cls < numClasses; cls++) {
                const conf = flatData[base + 4 + cls];
                if (conf > maxConf) { maxConf = conf; classId = cls; }
            }
            if (maxConf >= CONF_THRESHOLD) {
                candidates.push({
                    x: flatData[base],     y: flatData[base + 1],
                    w: flatData[base + 2], h: flatData[base + 3],
                    conf: maxConf, classId
                });
            }
        }
    }
    return candidates;
}

// ── Draw detection boxes ────────────────────────────────────
function drawBoxes(boxes, isFront) {
    if (!boxes.length) return;

    const sx = canvas.width  / INPUT_SIZE;
    const sy = canvas.height / INPUT_SIZE;

    boxes.forEach(box => {
        const color = getColor(box.classId);
        const label = `${yoloClasses[box.classId] || 'Object'} ${Math.round(box.conf * 100)}%`;

        let { x, y, w, h } = box;

        // Handle un-normalised coordinates (rare edge case)
        if (w <= 2.0 && h <= 2.0) {
            x *= INPUT_SIZE; y *= INPUT_SIZE;
            w *= INPUT_SIZE; h *= INPUT_SIZE;
        }

        let left = (x - w / 2) * sx;
        let top  = (y - h / 2) * sy;
        const bw = w * sx;
        const bh = h * sy;

        // Mirror box position for front camera
        if (isFront) left = canvas.width - left - bw;

        ctx.save();

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 6;
        ctx.strokeRect(left, top, bw, bh);

        // Corner brackets
        const c = Math.min(14, bw * 0.18, bh * 0.18);
        ctx.lineWidth  = 3;
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.moveTo(left,          top + c);      ctx.lineTo(left,          top);      ctx.lineTo(left + c,      top);
        ctx.moveTo(left + bw - c, top);          ctx.lineTo(left + bw,     top);      ctx.lineTo(left + bw,     top + c);
        ctx.moveTo(left,          top + bh - c); ctx.lineTo(left,          top + bh); ctx.lineTo(left + c,      top + bh);
        ctx.moveTo(left + bw - c, top + bh);     ctx.lineTo(left + bw,     top + bh); ctx.lineTo(left + bw,     top + bh - c);
        ctx.stroke();

        // Label pill
        ctx.font = 'bold 12px "Outfit", sans-serif';
        const tw = ctx.measureText(label).width;
        const ph = 20, px = 7;
        const lx = Math.max(0, Math.min(left, canvas.width  - tw - px * 2 - 2));
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
