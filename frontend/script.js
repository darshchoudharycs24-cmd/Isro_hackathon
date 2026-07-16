// ======================================================
// CloudClear AI Dashboard
// Part 1
// ======================================================

const imageInput = document.getElementById("imageInput");
const chooseImage = document.getElementById("chooseImage");
const processBtn = document.getElementById("processBtn");
const resetBtn = document.getElementById("resetBtn");

const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");

const consoleLogs = document.getElementById("consoleLogs");

const loadingOverlay = document.getElementById("loadingOverlay");

const toast = document.getElementById("toast");

const originalPreview = document.getElementById("originalPreview");

const placeholders = document.querySelectorAll(".placeholder");

// ======================================================
// Open File Picker
// ======================================================

chooseImage.addEventListener("click", () => {

    imageInput.click();

});

// ======================================================
// Image Preview
// ======================================================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function (e) {

        originalPreview.src = e.target.result;

        originalPreview.style.display = "block";

        placeholders[0].style.display = "none";

        toastMessage("Image Loaded Successfully");

    }

    reader.readAsDataURL(file);

});

// ======================================================
// Toast
// ======================================================

function toastMessage(message){

    toast.innerHTML = message;

    toast.style.display = "block";

    setTimeout(()=>{

        toast.style.display="none";

    },3000);

}

// ======================================================
// Console Logger
// ======================================================

function addLog(message){

    consoleLogs.innerHTML +=

    "<br>▶ " + message;

    consoleLogs.scrollTop = consoleLogs.scrollHeight;

}

// ======================================================
// Loading
// ======================================================

function showLoading(){

    loadingOverlay.style.display="flex";

}

function hideLoading(){

    loadingOverlay.style.display="none";

}

// ======================================================
// Fake Progress Animation
// ======================================================

function animateProgress(){

    progressFill.style.width="0%";

    let progress=0;

    progressText.innerHTML="Initializing...";

    const timer=setInterval(()=>{

        progress++;

        progressFill.style.width=progress+"%";

        if(progress<20){

            progressText.innerHTML="Uploading Image...";

        }

        else if(progress<40){

            progressText.innerHTML="Detecting Clouds...";

        }

        else if(progress<60){

            progressText.innerHTML="Running AI Reconstruction...";

        }

        else if(progress<80){

            progressText.innerHTML="Generating Confidence Map...";

        }

        else{

            progressText.innerHTML="Preparing Results...";

        }

        if(progress>=100){

            clearInterval(timer);

            progressText.innerHTML="Completed";

            hideLoading();

            toastMessage("Processing Complete");

        }

    },35);

}
// ======================================================
// PROCESS BUTTON
// ======================================================

processBtn.addEventListener("click", () => {

    
    if (!imageInput.files.length) {

        toastMessage("Please choose an image first.");

        return;

    }

    consoleLogs.innerHTML = "";

    showLoading();

    animateProgress();

    const steps = [

        "Uploading satellite image...",

        "Performing radiometric preprocessing...",

        "Running cloud detection model...",

        "Generating cloud mask...",

        "Selecting historical cloud-free imagery...",

        "Running AI reconstruction model...",

        "Generating confidence map...",

        "Calculating PSNR, SSIM, RMSE & SAM...",

        "Preparing downloadable outputs..."

    ];

    let delay = 500;

    steps.forEach((step) => {

        setTimeout(() => {

            addLog(step);

        }, delay);

        delay += 900;

    });

});

// ======================================================
// RESET BUTTON
// ======================================================

resetBtn.addEventListener("click", () => {

    imageInput.value = "";

    originalPreview.src = "";

    originalPreview.style.display = "none";

    placeholders[0].style.display = "flex";

    progressFill.style.width = "0%";

    progressText.innerHTML = "Waiting for image...";

    consoleLogs.innerHTML = "Ready.";

    toastMessage("Dashboard Reset");

});

// ======================================================
// DRAG & DROP
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

document.querySelectorAll(".downloadButton").forEach(button => {

    button.addEventListener("click", () => {

        toastMessage("Download will be enabled after backend integration.");

    });

});

// ======================================================
// FASTAPI TEMPLATE
// (Enable later)
// ======================================================

/*

processBtn.addEventListener("click", async () => {

    if (!imageInput.files.length) return;

    const formData = new FormData();

    formData.append("file", imageInput.files[0]);

    const response = await fetch("http://127.0.0.1:8000/predict", {

        method: "POST",

        body: formData

    });

    const result = await response.json();

    document.getElementById("maskPreview").src = result.cloud_mask;

    document.getElementById("outputPreview").src = result.reconstructed;

    document.getElementById("confidencePreview").src = result.confidence;

});

*/

// ======================================================
// INITIALIZE
// ======================================================

consoleLogs.innerHTML = "Ready.";

progressText.innerHTML = "Waiting for image...";