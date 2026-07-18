// ======================================================
// CloudClear AI Dashboard
// ======================================================

const imageInput = document.getElementById("imageInput");
const chooseImage = document.getElementById("chooseImage");

const historicalInput = document.getElementById("historicalInput");
const chooseHistorical = document.getElementById("chooseHistorical");
const dropAreaHistorical = document.getElementById("dropAreaHistorical");

const modeReference = document.getElementById("modeReference");
const modeHybrid = document.getElementById("modeHybrid");

const processBtn = document.getElementById("processBtn");
const resetBtn = document.getElementById("resetBtn");

const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");

const consoleLogs = document.getElementById("consoleLogs");
const loadingOverlay = document.getElementById("loadingOverlay");
const toast = document.getElementById("toast");
const originalPreview = document.getElementById("originalPreview");
const placeholders = document.querySelectorAll(".placeholder");

const beforeCompare = document.getElementById("beforeCompare");
const afterCompare = document.getElementById("afterCompare");
const comparisonPlaceholders = document.querySelectorAll(".comparisonContainer .placeholder");

let currentMode = "reference";

// ======================================================
// Mode Toggle
// ======================================================

modeReference.addEventListener("click", () => {
    currentMode = "reference";
    modeReference.classList.add("modeActive");
    modeHybrid.classList.remove("modeActive");
    dropAreaHistorical.style.display = "block";
});

modeHybrid.addEventListener("click", () => {
    currentMode = "hybrid";
    modeHybrid.classList.add("modeActive");
    modeReference.classList.remove("modeActive");
    dropAreaHistorical.style.display = "none";
    toastMessage("Single-image mode selected — no historical reference needed.");
});

// ======================================================
// Open File Pickers
// ======================================================

chooseImage.addEventListener("click", () => {
    imageInput.click();
});

chooseHistorical.addEventListener("click", () => {
    historicalInput.click();
});

// ======================================================
// Inline thumbnail preview helper
// ======================================================

function showThumbnailInUploadArea(dropAreaEl, dataUrl) {
    let thumb = dropAreaEl.querySelector(".uploadThumb");
    if (!thumb) {
        thumb = document.createElement("img");
        thumb.className = "uploadThumb";
        thumb.style.maxWidth = "100%";
        thumb.style.maxHeight = "180px";
        thumb.style.borderRadius = "14px";
        thumb.style.marginTop = "15px";
        dropAreaEl.appendChild(thumb);
    }
    thumb.src = dataUrl;
    thumb.style.display = "block";
}

// ======================================================
// Image Preview (current)
// ======================================================

imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
        originalPreview.src = e.target.result;
        originalPreview.style.display = "block";
        placeholders[0].style.display = "none";

        showThumbnailInUploadArea(dropArea, e.target.result);

        beforeCompare.src = e.target.result;
        beforeCompare.style.display = "block";
        comparisonPlaceholders[0].style.display = "none";

        toastMessage("Image Loaded Successfully");
    };
    reader.readAsDataURL(file);
});

// ======================================================
// Image Preview (historical)
// ======================================================

historicalInput.addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
        showThumbnailInUploadArea(dropAreaHistorical, e.target.result);
        toastMessage("Historical Reference Loaded");
    };
    reader.readAsDataURL(file);
});

// ======================================================
// Toast
// ======================================================

function toastMessage(message){
    toast.innerHTML = message;
    toast.style.display = "block";
    setTimeout(()=>{ toast.style.display="none"; },3000);
}

// ======================================================
// Console Logger
// ======================================================

function addLog(message){
    consoleLogs.innerHTML += "<br>▶ " + message;
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// ======================================================
// Loading
// ======================================================

function showLoading(){ loadingOverlay.style.display="flex"; }
function hideLoading(){ loadingOverlay.style.display="none"; }

// ======================================================
// PROCESS BUTTON
// ======================================================

processBtn.addEventListener("click", async () => {

    if (!imageInput.files.length) {
        toastMessage("Please choose a cloudy satellite image first.");
        return;
    }
    if (currentMode === "reference" && !historicalInput.files.length) {
        toastMessage("Reference mode needs a historical image — or switch to Single-Image AI mode.");
        return;
    }

    consoleLogs.innerHTML = "";
    showLoading();

    const steps = currentMode === "reference"
        ? ["Uploading images...", "Generating cloud mask...", "Cloning reference pixels...", "Computing metrics..."]
        : ["Uploading image...", "Generating cloud mask...", "Running GAN reconstruction...", "Computing metrics..."];

    let i = 0;
    progressFill.style.width = "0%";
    const logTimer = setInterval(() => {
        if (i < steps.length) {
            addLog(steps[i]);
            progressText.innerHTML = steps[i];
            progressFill.style.width = Math.round(((i + 1) / steps.length) * 90) + "%";
            i++;
        }
    }, 500);

    try {
        const formData = new FormData();
        formData.append("current", imageInput.files[0]);
        formData.append("mode", currentMode);
        if (historicalInput.files.length) {
            formData.append("historical", historicalInput.files[0]);
        }

       const API_URL = "https://isro-hackathon-mqkl.onrender.com";

const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
});

        const result = await response.json();
        clearInterval(logTimer);

        if (result.error) {
            addLog("Error: " + result.error);
            toastMessage("Processing failed — see console log.");
            hideLoading();
            return;
        }

        const maskPreview = document.getElementById("maskPreview");
        const outputPreview = document.getElementById("outputPreview");
        const confidencePreview = document.getElementById("confidencePreview");

        maskPreview.src = result.cloud_mask;
        maskPreview.style.display = "block";

        outputPreview.src = result.reconstructed;
        outputPreview.style.display = "block";

        confidencePreview.src = result.confidence;
        confidencePreview.style.display = "block";

        placeholders[1].style.display = "none";
        placeholders[2].style.display = "none";
        placeholders[3].style.display = "none";

        // Populate Before/After comparison section
        afterCompare.src = result.reconstructed;
        afterCompare.style.display = "block";
        comparisonPlaceholders[1].style.display = "none";

        document.getElementById("psnrValue").innerHTML = result.metrics.psnr;
        document.getElementById("ssimValue").innerHTML = result.metrics.ssim;
        document.getElementById("rmseValue").innerHTML = result.metrics.rmse;
        document.getElementById("samValue").innerHTML = result.metrics.sam;

        addLog("Mode: " + result.mode);
        progressFill.style.width = "100%";
        progressText.innerHTML = "Completed";
        addLog("Done.");
        hideLoading();
        toastMessage("Processing Complete");
    } catch (err) {
        clearInterval(logTimer);
        addLog("Error: " + err.message);
        toastMessage("Could not reach backend — is api_server.py running?");
        hideLoading();
    }
});

// ======================================================
// RESET BUTTON
// ======================================================

resetBtn.addEventListener("click", () => {
    imageInput.value = "";
    historicalInput.value = "";

    originalPreview.src = "";
    originalPreview.style.display = "none";

    document.getElementById("maskPreview").style.display = "none";
    document.getElementById("outputPreview").style.display = "none";
    document.getElementById("confidencePreview").style.display = "none";

    const thumbs = document.querySelectorAll(".uploadThumb");
    thumbs.forEach(t => t.remove());

    beforeCompare.src = "";
    beforeCompare.style.display = "none";
    afterCompare.src = "";
    afterCompare.style.display = "none";

    placeholders.forEach(p => p.style.display = "flex");
    comparisonPlaceholders.forEach(p => p.style.display = "flex");

    progressFill.style.width = "0%";
    progressText.innerHTML = "Waiting for image...";
    consoleLogs.innerHTML = "Ready.";

    document.getElementById("psnrValue").innerHTML = "--";
    document.getElementById("ssimValue").innerHTML = "--";
    document.getElementById("rmseValue").innerHTML = "--";
    document.getElementById("samValue").innerHTML = "--";

    toastMessage("Dashboard Reset");
});

// ======================================================
// DRAG & DROP (current image)
// ======================================================

const dropArea = document.getElementById("dropArea");

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.borderColor = "#00C2FF";
});
dropArea.addEventListener("dragleave", () => {
    dropArea.style.borderColor = "rgba(0,194,255,.35)";
});
dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.style.borderColor = "rgba(0,194,255,.35)";
    if (e.dataTransfer.files.length === 0) return;
    imageInput.files = e.dataTransfer.files;
    imageInput.dispatchEvent(new Event("change"));
});

// ======================================================
// DRAG & DROP (historical image)
// ======================================================

dropAreaHistorical.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropAreaHistorical.style.borderColor = "#00C2FF";
});
dropAreaHistorical.addEventListener("dragleave", () => {
    dropAreaHistorical.style.borderColor = "rgba(0,194,255,.35)";
});
dropAreaHistorical.addEventListener("drop", (e) => {
    e.preventDefault();
    dropAreaHistorical.style.borderColor = "rgba(0,194,255,.35)";
    if (e.dataTransfer.files.length === 0) return;
    historicalInput.files = e.dataTransfer.files;
    historicalInput.dispatchEvent(new Event("change"));
});

// ======================================================
// SCROLL EFFECT
// ======================================================

window.addEventListener("scroll", () => {
    const nav = document.querySelector("nav");
    if (window.scrollY > 60) {
        nav.style.background = "rgba(5,15,30,.92)";
    } else {
        nav.style.background = "rgba(5,15,30,.65)";
    }
});

// ======================================================
// DOWNLOAD BUTTONS
// ======================================================

document.querySelectorAll(".downloadButton").forEach((button, idx) => {
    button.addEventListener("click", () => {
        const map = {
            0: originalPreview,
            1: document.getElementById("maskPreview"),
            2: document.getElementById("outputPreview"),
            3: document.getElementById("confidencePreview"),
        };
        const img = map[idx];
        if (!img || !img.src) {
            toastMessage("Nothing to download yet — process an image first.");
            return;
        }
        const a = document.createElement("a");
        a.href = img.src;
        a.download = ["original", "cloud_mask", "reconstructed", "confidence"][idx] + ".png";
        a.click();
    });
});

// ======================================================
// INITIALIZE
// ======================================================

consoleLogs.innerHTML = "Ready.";
progressText.innerHTML = "Waiting for image...";
