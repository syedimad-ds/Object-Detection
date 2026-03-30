const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');

let model;

// --- 1. YOLOv8 Standard 80 Classes ---
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

// --- 2. Color Palette for Different Classes ---
const classColors = [
    '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
    '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'
];

function getColor(classId) {
    return classColors[classId % classColors.length];
}

// --- 3. Helper Function: IOU (Intersection Over Union) for NMS ---
function calculateIOU(box1, box2) {
    const b1Left = box1.x - box1.w / 2;
    const b1Right = box1.x + box1.w / 2;
    const b1Top = box1.y - box1.h / 2;
    const b1Bottom = box1.y + box1.h / 2;

    const b2Left = box2.x - box2.w / 2;
    const b2Right = box2.x + box2.w / 2;
    const b2Top = box2.y - box2.h / 2;
    const b2Bottom = box2.y + box2.h / 2;

    const xA = Math.max(b1Left, b2Left);
    const yA = Math.max(b1Top, b2Top);
    const xB = Math.min(b1Right, b2Right);
    const yB = Math.min(b1Bottom, b2Bottom);

    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const unionArea = (box1.w * box1.h) + (box2.w * box2.h) - intersectionArea;
    
    return intersectionArea / unionArea;
}

// --- 4. Load Model & Camera ---
async function loadModel() {
    try {
        await tf.ready();
        model = await tf.loadGraphModel('./yolov8s_web_model/model.json');
        
        statusDiv.className = "alert alert-success d-inline-block shadow-sm";
        statusDiv.innerText = "✅ Model Loaded! System Active.";
        startWebcam();
    } catch (error) {
        statusDiv.className = "alert alert-danger d-inline-block shadow-sm";
        statusDiv.innerText = `❌ Error: Check Console`;
        console.error(error);
    }
}

async function startWebcam() {
    try {
        // THE FIX: Mobile par Back Camera use karne ke liye facingMode 'environment'
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        video.srcObject = stream;
        video.onloadedmetadata = async () => { 
            await video.play(); 
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            detectFrame(); 
        };
    } catch (error) {
        statusDiv.innerText = "❌ Camera Error: Please allow permissions.";
    }
}

// --- 5. Multi-Object Detection Engine (Mobile Optimized) ---
async function detectFrame() {
    try {
        const inputResolution = [640, 640];
        const tfImg = tf.tidy(() => {
            return tf.browser.fromPixels(video)
                .resizeBilinear(inputResolution)
                .expandDims(0)
                .toFloat()
                .div(255.0);
        });

        // THE FIX: Allow the browser UI to update before heavy AI calculation
        await tf.nextFrame(); 

        const predictions = await model.executeAsync(tfImg);
        const tensorOutput = Array.isArray(predictions) ? predictions[0] : predictions;
        
        const shape = tensorOutput.shape; 
        const data = await tensorOutput.array();
        const detections = data[0]; 

        let candidates = [];
        const CONF_THRESHOLD = 0.35; 
        const IOU_THRESHOLD = 0.45;

        // Process detections based on output shape
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
                    candidates.push({
                        x: detections[0][col], y: detections[1][col], 
                        w: detections[2][col], h: detections[3][col], 
                        conf: maxClassConf, classId: classId
                    });
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
                    candidates.push({
                        x: detections[row][0], y: detections[row][1], 
                        w: detections[row][2], h: detections[row][3], 
                        conf: maxClassConf, classId: classId
                    });
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
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

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
                
                const boxW = w * scaleX;
                const boxH = h * scaleY;
                const left = (x * scaleX) - (boxW / 2);
                const top = (y * scaleY) - (boxH / 2);

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
        // Error catch
    }
    
    requestAnimationFrame(detectFrame);
}

loadModel();