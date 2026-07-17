// 1. DYNAMIC CONNECTION TO ROSBRIDGE INSIDE DOCKER
const ros = new ROSLIB.Ros({
    url : 'ws://' + window.location.hostname + ':9090'
});

// --- DOM REFERENCES (grabbed first, before anything uses them) ---
const leftEye = document.getElementById('left-eye');
const rightEye = document.getElementById('right-eye');
const robotMouth = document.getElementById('robot-mouth');
const canvas = document.getElementById('tracking-canvas');
const ctx = canvas.getContext('2d');
const img = document.getElementById('camera-stream');

ros.on('connection', () => { document.getElementById('status').innerText = 'Connected'; });
ros.on('error', () => { document.getElementById('status').innerText = 'Connection Error'; });
ros.on('close', () => { document.getElementById('status').innerText = 'Connection Closed'; });

// 2. LAYOUT SWITCHING (Face Mode vs Camera Stream Mode)
function connectStream() {
    const streamUrl = "http://" + window.location.hostname +
        ":8080/stream?topic=/camera/image_raw&type=mjpeg&qos_profile=sensor_data";
    img.src = streamUrl + "&_ts=" + Date.now(); // cache-bust
}

img.onerror = () => {
    document.getElementById('status').innerText = 'Stream error, retrying...';
    setTimeout(connectStream, 2000);
};

function setMode(mode) {
    if (mode === 'face') {
        document.getElementById('face-container').classList.remove('hidden');
        document.getElementById('vision-container').classList.add('hidden');
    } else if (mode === 'vision') {
        document.getElementById('face-container').classList.add('hidden');
        document.getElementById('vision-container').classList.remove('hidden');
        connectStream();
    }
}

// 3. FULLSCREEN & LANDSCAPE ORIENTATION LOCK
async function activateRobotDisplay() {
    try {
        const docElm = document.documentElement;
        if (docElm.requestFullscreen) { await docElm.requestFullscreen(); }
        else if (docElm.webkitRequestFullscreen) { await docElm.webkitRequestFullscreen(); }

        if (screen.orientation && screen.orientation.lock) {
            await screen.orientation.lock('landscape');
        }
    } catch (error) {
        console.log("Device interaction required for browser fullscreen mapping.");
    }
}

function toggleMenu() {
    const menu = document.getElementById('menu-overlay');
    menu.classList.toggle('hidden');
    if (menu.classList.contains('hidden')) {
        activateRobotDisplay();
    }
}

// Tap on dark background forces fullscreen bypass
document.addEventListener('click', (e) => {
    if (e.target.id !== 'menu-trigger' && document.getElementById('menu-overlay').classList.contains('hidden')) {
        activateRobotDisplay();
    }
});

// 4. ROS TOPIC SUBSCRIPTION: SYSTEM MODE CONTROL
const uiModeListener = new ROSLIB.Topic({
    ros : ros,
    name : '/robot_ui_mode', // Pass 'face' or 'vision'
    messageType : 'std_msgs/String'
});
uiModeListener.subscribe((msg) => {
    setMode(msg.data.toLowerCase());
});

// 5. ROS TOPIC SUBSCRIPTION: EMOTIONS
const emotionListener = new ROSLIB.Topic({
    ros : ros,
    name : '/robot_emotion', // Pass 'happy', 'sad', 'thinking', 'talking', 'neutral'
    messageType : 'std_msgs/String'
});

emotionListener.subscribe((message) => {
    // Reset all base classes to clear the previous emotion state
    leftEye.className = 'eye';
    rightEye.className = 'eye';
    robotMouth.className = 'mouth';
    robotMouth.style.height = '';   // clear any lip-sync inline override
    robotMouth.style.transform = ''; 
    robotMouth.style.opacity = '';

    const emotion = message.data.toLowerCase();

    if (emotion === 'happy') {
        leftEye.classList.add('happy');
        rightEye.classList.add('happy');
        robotMouth.classList.add('happy');
    } else if (emotion === 'sad') {
        leftEye.classList.add('sad');
        rightEye.classList.add('sad');
        robotMouth.classList.add('sad');
    } else if (emotion === 'thinking') {
        leftEye.classList.add('thinking');
        rightEye.classList.add('thinking');
        robotMouth.classList.add('thinking');
    } else if (emotion === 'talking') {
        // Mouth shape/motion during speech is now driven live by
        // /mouth_level (real amplitude lip-sync) — see mouthLevelListener
        // below. Just make sure no static animation class fights it.
        robotMouth.classList.add('neutral');
    } else {
        // Default Neutral / Rest state
        robotMouth.classList.add('neutral');
    }
});

// 5b. LIP-SYNC: mouth opening driven by real TTS audio loudness
const MOUTH_MIN_HEIGHT = 15;   // px, closed
const MOUTH_MAX_HEIGHT = 55;   // px, wide open

const mouthLevelListener = new ROSLIB.Topic({
    ros : ros,
    name : '/mouth_level',      // 0.0 (silent) - 1.0 (loudest) from tts_node
    messageType : 'std_msgs/Float32'
});

mouthLevelListener.subscribe((msg) => {
    const level = Math.max(0, Math.min(1, msg.data));
    // Scale the mouth's existing shape vertically from the top edge,
    // like a jaw opening/closing, instead of ballooning the smile arc
    // into a rounder shape (which read as a grin/laugh).
    const scale = 1.0 + level * 2.5;   // subtle open range: 1.0x - 3.2x
    const dropPx = level * 40;           // jaw drops down to 22px

    robotMouth.style.transformOrigin = 'top center';
    robotMouth.style.transform = `translateY(${dropPx}px) scaleY(${scale})`;
    robotMouth.style.opacity = 0.75 + level * 0.25;
});

// 6. PROCEDURAL IDLE BLINK CYCLE
setInterval(() => {
    if (Math.random() > 0.3) {
        const currentLeftClass = leftEye.className;
        const currentRightClass = rightEye.className;

        leftEye.className = 'eye blink';
        rightEye.className = 'eye blink';

        setTimeout(() => {
            leftEye.className = currentLeftClass;
            rightEye.className = currentRightClass;
        }, 140);
    }
}, 4000);

// 7. CANVAS / VIDEO OVERLAY SIZING
// Rescale canvas area dynamically to perfectly match the displayed video aspect bounds
function fitCanvasToVideo() {
    const naturalW = img.naturalWidth;
    const naturalH = img.naturalHeight;
    if (!naturalW || !naturalH) return; // stream not loaded yet

    const boxW = img.clientWidth;
    const boxH = img.clientHeight;

    const imgRatio = naturalW / naturalH;
    const boxRatio = boxW / boxH;

    let renderW, renderH, offsetX, offsetY;

    if (imgRatio > boxRatio) {
        renderW = boxW;
        renderH = boxW / imgRatio;
        offsetX = 0;
        offsetY = (boxH - renderH) / 2;
    } else {
        renderH = boxH;
        renderW = boxH * imgRatio;
        offsetY = 0;
        offsetX = (boxW - renderW) / 2;
    }

    canvas.width = renderW;
    canvas.height = renderH;
    canvas.style.left = (img.offsetLeft + offsetX) + "px";
    canvas.style.top = (img.offsetTop + offsetY) + "px";
}
setInterval(fitCanvasToVideo, 1000);
window.onresize = fitCanvasToVideo;

// 8. TRACKING SUBSCRIPTION
const trackingListener = new ROSLIB.Topic({
    ros : ros,
    name : '/tracked_detections',
    messageType : 'vision_msgs/msg/Detection2DArray' // Standard ROS2 package type
});

trackingListener.subscribe((message) => {
    // 1. Clear previous target lines every single frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 2. Halt rendering pipeline if UI panel is switched to face mode
    if (document.getElementById('vision-container').classList.contains('hidden')) return;

    // 3. Loop through your SORT tracks array
    message.detections.forEach(det => {
        // NOTE: Adjust 640 and 480 below to match the native resolution of your /detections node input frame
        const nativeWidth = 640;
        const nativeHeight = 480;

        const scaleX = canvas.width / nativeWidth;
        const scaleY = canvas.height / nativeHeight;

        // Parse absolute coordinates out of standard Vision Message structures
        const cx = det.bbox.center.position.x * scaleX;
        const cy = det.bbox.center.position.y * scaleY;
        const w = det.bbox.size_x * scaleX;
        const h = det.bbox.size_y * scaleY;

        // Extract parameters
        const x = cx - (w / 2);
        const y = cy - (h / 2);
        const trackId = det.id; // Your string converted tracking id
        const className = det.results[0] ? det.results[0].hypothesis.class_id : "unknown";

        // 4. Render Neon Tracking Bounding Box
        ctx.strokeStyle = '#00ffcc'; // Matched cyan glow profile
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // 5. Draw Tracking Tag background pill
        ctx.fillStyle = '#00ffcc';
        ctx.font = 'bold 14px sans-serif';
        const labelText = `ID: ${trackId} | ${className}`;
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillRect(x - 1, y - 22, textWidth + 10, 22);

        // Print text context label string inside the frame container
        ctx.fillStyle = '#000000';
        ctx.fillText(labelText, x + 5, y - 6);
    });
});
