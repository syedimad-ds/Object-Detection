// ============================================================
//  EDGE AI — Object Detection | THE ULTIMATE OPTIMIZED VERSION
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

// ── Device Detection ───────────────────────────────────────
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
if (!isMobile && switchCamBtn) switchCamBtn.style.display = 'none';

// ── DYNAMIC RESOLUTION STATE ───────────────────────────────
let isHDMode = !isMobile; // Mobile pe Nano(320), Laptop pe Small(640)

let INPUT_W = isHDMode ? 640 : 320;
let INPUT_H = isHDMode ? 640 : 320;

const processCanvas = document.createElement('canvas');
processCanvas.width = INPUT_W;
processCanvas.height = INPUT_H;
const processCtx = processCanvas.getContext('2d', { willReadFrequently: true, alpha: false });

let model             = null;
let currentFacingMode = 'environment';
let streamActive      = false;
let inferenceRunning  = false;
let lastBoxes         = [];
let renderLoopId      = null;
let sessionIsFront    = false;
let backendName       = 'unknown';

let fpsCount    = 0;
let fpsLastTime = performance.now();

const modelPathNano  = './yolov8n_web_model/model.json'; 
const modelPathSmall = './yolov8s_web_model/model.json'; 

// ── STRICT AI TUNING (To remove noise & overlapping) ───────
const CONF_THRESHOLD = isMobile ? 0.50 : 0.55; // Sirf 50%+ confident objects
const IOU_THRESHOLD  = 0.35;                   // Overlapping boxes ko sakhti se merge karega
const MAX_DETECTIONS = 10;                     // Screen par ek baar mein max 10 objects

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

// ── WebGL Initialization ───────────────────────────────────
async function initBackend() {
    try {
        await tf.setBackend('webgl');
        if (tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
            tf.env().set('WEBGL_FORCE_F16_PIPELINES', true);
        }
        tf.env().set('WEBGL_PACK', true);
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
        backendName = 'WebGL (GPU)';
    } catch (err) {
        console.warn('WebGL failed. Fallback to WASM...');
        try {
            await tf.setBackend('wasm');
            backendName = 'WASM';
        } catch (err2) {
            await tf.setBackend('cpu');
            backendName = 'CPU';
        }
    }
    await tf.ready();
    console.log(`✅ Backend Active: ${backendName}`);
}

// ── Model Setup ────────────────────────────────────────────
async function loadModel() {
    try {
        setStatus('loading', `⏳ Initializing ${backendName}...`);
        if (model) {
            model.dispose(); 
            model = null;
        }

        const path = isHDMode ? modelPathSmall : modelPathNano;
        setStatus('loading', `⏳ Downloading ${isHDMode ? 'Small (640)' : 'Nano (320)'} model...`);
        model = await tf.loadGraphModel(path);

        setStatus('loading', '🔥 Warming up AI...');
        const dummy = tf.zeros([1, INPUT_H, INPUT_W, 3]);
        const warmup = await model.executeAsync(dummy);
        if (Array.isArray(warmup)) warmup.forEach(t => t.dispose());
        else warmup.dispose();
        dummy.dispose();

        if(modelIndicator) modelIndicator.textContent = `Detection (${isHDMode ? 'Small' : 'Nano'}) [${backendName}]`;
        setStatus('active', `✅ System Active`);
        
        if (!streamActive) await startWebcam();
    } catch (err) {
        setStatus('error', `❌ Load failed: ${err.message}`);
    }
}

// ── Dynamic Resolution Toggle ──────────────────────────────
if (hdModeBtn) {
    if (isHDMode) hdModeBtn.classList.add('active');

    hdModeBtn.addEventListener('click', async () => {
        isHDMode = !isHDMode;
        hdModeBtn.classList.toggle('active', isHDMode);
        
        INPUT_W = isHDMode ? 640 : 320;
        INPUT_H = isHDMode ? 640 : 320;
        processCanvas.width = INPUT_W;
        processCanvas.height = INPUT_H;

        streamActive = false;
        await loadModel();
        streamActive = true;
        runInferenceLoop();
    });
}

// ── Camera Pipeline ────────────────────────────────────────
async function startWebcam() {
    streamActive = false;
    inferenceRunning = false;
    lastBoxes = [];

    if (renderLoopId !== null) {
        cancelAnimationFrame(renderLoopId);
        renderLoopId = null;
    }
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: isMobile ? currentFacingMode : 'user',
                width: isMobile ? { ideal: 480 } : { ideal: 640 }, 
                height: isMobile ? { ideal: 360 } : { ideal: 480 }
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
        setStatus('error', '❌ Camera denied.');
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

// ── Video Render Loop (60 FPS) ─────────────────────────────
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

// ── Heavy AI Inference Engine (GPU Bound) ──────────────────
async function runInferenceLoop() {
    while (streamActive) {
        if (!model || video.readyState < 2 || inferenceRunning) {
            await new Promise(r => requestAnimationFrame(r));
            continue;
        }

        inferenceRunning = true;

        try {
            // STEP 1: Letterboxing (Aspect Ratio Maintain)
            const vidW = video.videoWidth;
            const vidH = video.videoHeight;
            const scale = Math.min(INPUT_W / vidW, INPUT_H / vidH);
            const drawW = vidW * scale;
            const drawH = vidH * scale;
            const padX = (INPUT_W - drawW) / 2;
            const padY = (INPUT_H - drawH) / 2;

            processCtx.fillStyle = '#727272'; 
            processCtx.fillRect(0, 0, INPUT_W, INPUT_H);
            processCtx.drawImage(video, 0, 0, vidW, vidH, padX, padY, drawW, drawH);

            // STEP 2: Tensor Operations
            const { nmsBoxes, maxScores, classIds } = tf.tidy(() => {
                const inputTensor = tf.browser.fromPixels(processCanvas)
                    .expandDims(0)
                    .toFloat()
                    .div(255.0);

                let out = model.execute(inputTensor);
                if (Array.isArray(out)) out = out[0];

                if (out.shape[1] === 84 || out.shape[1] === 80) {
                    out = out.transpose([0, 2, 1]);
                }
                out = out.squeeze([0]); 

                const numBoxes = out.shape[0];
                const numClasses = out.shape[1] - 4;

                const boxes = out.slice([0, 0], [numBoxes, 4]);
                const scores = out.slice([0, 4], [numBoxes, numClasses]);

                const cx = boxes.slice([0, 0], [numBoxes, 1]);
                const cy = boxes.slice([0, 1], [numBoxes, 1]);
                const w  = boxes.slice([0, 2], [numBoxes, 1]);
                const h  = boxes.slice([0, 3], [numBoxes, 1]);

                const halfW = w.div(2);
                const halfH = h.div(2);

                const y1 = cy.sub(halfH);
                const x1 = cx.sub(halfW);
                const y2 = cy.add(halfH);
                const x2 = cx.add(halfW);

                const tfNmsBoxes = tf.concat([y1, x1, y2, x2], 1);
                const tfMaxScores = scores.max(1);
                const tfClassIds = scores.argMax(1);

                return { nmsBoxes: tfNmsBoxes, maxScores: tfMaxScores, classIds: tfClassIds };
            });

            // STEP 3: NMS Filtering on GPU
            const selectedIndicesTensor = await tf.image.nonMaxSuppressionAsync(
                nmsBoxes, maxScores, MAX_DETECTIONS, IOU_THRESHOLD, CONF_THRESHOLD
            );

            // STEP 4: Memory Safe Data Extraction
            const indicesArr = await selectedIndicesTensor.data();
            const boxesFlat = await nmsBoxes.data();
            const scoresFlat = await maxScores.data();
            const classesFlat = await classIds.data();

            tf.dispose([nmsBoxes, maxScores, classIds, selectedIndicesTensor]);

            const processedBoxes = [];
            for (let i = 0; i < indicesArr.length; i++) {
                const idx = indicesArr[i];
                let y1 = boxesFlat[idx * 4 + 0];
                let x1 = boxesFlat[idx * 4 + 1];
                let y2 = boxesFlat[idx * 4 + 2];
                let x2 = boxesFlat[idx * 4 + 3];

                let w = x2 - x1;
                let h = y2 - y1;
                let cx = x1 + w / 2;
                let cy = y1 + h / 2;

                if (w <= 2 && h <= 2) { 
                    cx *= INPUT_W; cy *= INPUT_H; w *= INPUT_W; h *= INPUT_H; 
                }

                // STEP 5: Reverse Letterboxing (Map to original video space)
                cx = (cx - padX) / scale;
                cy = (cy - padY) / scale;
                w = w / scale;
                h = h / scale;

                processedBoxes.push({
                    x: cx, y: cy, w: w, h: h,
                    conf: scoresFlat[idx],
                    classId: classesFlat[idx]
                });
            }
            lastBoxes = processedBoxes;

            if (lastBoxes.length > 0) {
                setStatus('active', `🟢 ${lastBoxes.length} object${lastBoxes.length !== 1 ? 's' : ''} detected`);
                if (detectionCount) detectionCount.textContent = lastBoxes.length;
            } else {
                setStatus('scanning', '📡 Scanning...');
                if (detectionCount) detectionCount.textContent = '0';
            }

        } catch (err) {
            console.warn('Inference error:', err);
        }

        inferenceRunning = false;
        
        // Throttling for thermal control
        const coolingDelay = isMobile ? 60 : 10; 
        await new Promise(resolve => setTimeout(resolve, coolingDelay)); 
    }
}

// ── Canvas Drawing Logic ───────────────────────────────────
function drawBoxes(boxes) {
    if (!boxes.length) return;

    const scaleX = canvas.width  / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    boxes.forEach(box => {
        const color = getColor(box.classId);
        const name  = COCO_CLASSES[box.classId] || 'Object';
        const label = `${name} ${Math.round(box.conf * 100)}%`;

        let boxW = box.w * scaleX;
        let boxH = box.h * scaleY;
        let left = (box.x * scaleX) - (boxW / 2);
        let top  = (box.y * scaleY) - (boxH / 2);

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

function setStatus(type, msg) {
    statusDiv.className = `status-pill status-${type}`;
    statusDiv.innerHTML = msg;
}

// ── Start Engine ───────────────────────────────────────────
initBackend().then(() => loadModel());