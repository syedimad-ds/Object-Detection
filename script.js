const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');
const switchCamBtn = document.getElementById('switchCamBtn'); 

let model;
let currentFacingMode = 'environment'; 
let isDetecting = false; 

// --- DEVICE DETECTION LOGIC ---
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

if (!isMobile && switchCamBtn) {
    switchCamBtn.style.display = 'none';
}

const modelPath = isMobile ? './yolov8n_web_model/model.json' : './yolov8s_web_model/model.json';

// FIX 1: DYNAMIC THRESHOLD (Mobile ke liye kam, Laptop ke liye zyada)
const CONF_THRESHOLD = isMobile ? 0.25 : 0.40; 
const IOU_THRESHOLD = 0.40;

const yoloClasses = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

const classColors = [
    '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
    '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'
];

function getColor(classId) {
    return classColors[classId % classColors.length];
}

function calculateIOU(box1, box2) {
    const b1Left = box1.x - box1.w / 2;
    const b1Right = box1.x + box1.w / 2;
    const b1Top = box1.y - box1.h / 2;
    const b1Bottom = box1.y + box1.h / 2;

    const xA = Math.max(b1Left, b2Left);
    const yA = Math.max(b1Top, b2Top);
    const xB = Math.min(b1Right, b2Right);
    const yB = Math.min(b1Bottom, b2Bottom);

    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const unionArea = (box1.w * box1.h) + (box2.w * box2.h) - intersectionArea;
    return intersectionArea / unionArea;
}

async function loadModel() {
    try {
        // Force WebGL Backend for best performance
        await tf.setBackend('webgl');
        await tf.ready();
        
        statusDiv.innerText = "⏳ Downloading Model Files...";
        model = await tf.loadGraphModel(modelPath);
        
        // FIX 2: THE WARM-UP ENGINE (Prevents the 2-minute laptop freeze)
        statusDiv.innerText = "🔥 Warming up AI Engine... (Please wait)";
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        const warmupStart = Date.now();
        const dummyOutput = await model.executeAsync(dummyInput);
        tf.dispose([dummyInput, dummyOutput]); // Clean memory
        console.log(`Warmup completed in ${Date.now() - warmupStart}ms`);

        const modeText = isMobile ? "Fast Mode (Nano)" : "High Accuracy Mode (Small)";
        statusDiv.className = "alert alert-success d-inline-block shadow-sm";
        statusDiv.innerText = `✅ Model Loaded! System Active. [${modeText}]`;
        
        startWebcam();
    } catch (error) {
        statusDiv.className = "alert alert-danger d-inline-block shadow-sm";
        statusDiv.innerText = `❌ Error: Model Load Failed`;
        console.error(error);
    }
}

async function startWebcam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: isMobile ? currentFacingMode : 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        video.srcObject = stream;
        video.onloadedmetadata = async () => { 
            await video.play(); 
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            if (!isDetecting) {
                isDetecting = true;
                detectFrame(); 
            }
        };
    } catch (error) {
        statusDiv.innerText = "❌ Camera Error: Please allow permissions.";
    }
}

if(switchCamBtn) {
    switchCamBtn.addEventListener('click', () => {
        if (!isMobile) return; 
        currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
        switchCamBtn.innerText = "🔄 Switching...";
        setTimeout(() => { switchCamBtn.innerText = "🔄 Switch Camera"; }, 1000);
        startWebcam(); 
    });
}

async function detectFrame() {
    if (!isDetecting) return;

    try {
        const inputResolution = [640, 640];
        const tfImg = tf.tidy(() => {
            return tf.browser.fromPixels(video)
                .resizeBilinear(inputResolution)
                .expandDims(0)
                .toFloat()
                .div(255.0);
        });

        // Use execute() instead of executeAsync() if possible for faster sync processing
        const predictions = await model.executeAsync(tfImg);
        const tensorOutput = Array.isArray(predictions) ? predictions[0] : predictions;
        
        const shape = tensorOutput.shape; 
        const data = await tensorOutput.array();
        const detections = data[0]; 

        let candidates = [];

        if (shape[1] === 84 || shape[1] === 80) {
            const numClasses = shape[1] - 4; 
            const numBoxes = shape[2];
            for (let col = 0; col < numBoxes; col++) {
                let maxClassConf = 0;
                let classId = -1;
                for (let cls = 0; cls < numClasses; cls++) {
                    if (detections[cls + 4][col] > maxClassConf) {
                        maxClassConf = detections[cls + 4][col];
                        classId = cls;
                    }
                }
                if (maxClassConf > CONF_THRESHOLD) {
                    candidates.push({ x: detections[0][col], y: detections[1][col], w: detections[2][col], h: detections[3][col], conf: maxClassConf, classId: classId });
                }
            }
        } else if (shape[2] === 84 || shape[2] === 80) {
            const numBoxes = shape[1]; 
            const numClasses = shape[2] - 4;
            for (let row = 0; row < numBoxes; row++) {
                let maxClassConf = 0;
                let classId = -1;
                for (let cls = 0; cls < numClasses; cls++) {
                    if (detections[row][cls + 4] > maxClassConf) {
                        maxClassConf = detections[row][cls + 4];
                        classId = cls;
                    }
                }
                if (maxClassConf > CONF_THRESHOLD) {
                    candidates.push({ x: detections[row][0], y: detections[row][1], w: detections[row][2], h: detections[row][3], conf: maxClassConf, classId: classId });
                }
            }
        }

        candidates.sort((a, b) => b.conf - a.conf);
        let finalBoxes = [];
        while (candidates.length > 0) {
            const best = candidates.shift();
            finalBoxes.push(best);
            candidates = candidates.filter(box => {
                if (box.classId !== best.classId) return true;
                return calculateIOU(best, box) < IOU_THRESHOLD; 
            });
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.save();
        if (currentFacingMode === 'user' || !isMobile) {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        if (finalBoxes.length > 0) {
            statusDiv.className = "alert alert-success d-inline-block shadow-sm";
            statusDiv.innerHTML = `🟢 <strong>${finalBoxes.length} OBJECTS DETECTED!</strong>`;

            finalBoxes.forEach(box => {
                const detectedName = yoloClasses[box.classId] || "Object";
                const themeColor = getColor(box.classId);
                
                let {x, y, w, h, conf} = box;

                if (w <= 2 && h <= 2) {
                    x *= inputResolution[0]; y *= inputResolution[1];
                    w *= inputResolution[0]; h *= inputResolution[1];
                }

                const scaleX = canvas.width / inputResolution[0];
                const scaleY = canvas.height / inputResolution[1];
                
                let boxW = w * scaleX;
                let boxH = h * scaleY;
                let left = (x * scaleX) - (boxW / 2);
                let top = (y * scaleY) - (boxH / 2);

                if (currentFacingMode === 'user' || !isMobile) {
                    left = canvas.width - left - boxW;
                }

                ctx.strokeStyle = themeColor; 
                ctx.lineWidth = 3;
                ctx.shadowColor = themeColor;
                ctx.shadowBlur = 8;
                ctx.strokeRect(left, top, boxW, boxH);

                ctx.shadowBlur = 0; 
                ctx.fillStyle = themeColor;
                const labelText = `${detectedName}: ${(conf * 100).toFixed(0)}%`;
                const textWidth = ctx.measureText(labelText).width + 20;
                ctx.fillRect(left, top - 25, textWidth, 25); 

                ctx.fillStyle = '#ffffff'; 
                ctx.font = 'bold 15px "Segoe UI"';
                ctx.fillText(labelText, left + 7, top - 7);
            });
        } else {
            statusDiv.className = "alert alert-warning d-inline-block shadow-sm";
            statusDiv.innerHTML = `📡 Radar Scanning...`;
        }
        
        tfImg.dispose();
        tf.dispose(predictions); 

    } catch (error) {
        // Silent catch
    }
    
    // FIX 3: Frame Pacing - Request next frame only after current is fully drawn
    // Isse browser ko saans lene ka mauka milega aur phone hang nahi hoga
    await tf.nextFrame();
    requestAnimationFrame(detectFrame);
}

loadModel();