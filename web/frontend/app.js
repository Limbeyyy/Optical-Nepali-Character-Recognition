const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const imageInput = document.getElementById("imageInput");
const plateTypeSelect = document.getElementById("plateType");

const redoBtn = document.getElementById("redoBtn");
const nextBtn = document.getElementById("nextBtn");
const okBtn = document.getElementById("okBtn");
const statusDiv = document.getElementById("status");
const outputPre = document.getElementById("output");

let img = new Image();
let startX, startY, endX, endY;
let drawing = false;

let fields = [
  "Local_Address",
  "kataho_Address",
  "KID_No",
  "Plus_Code",
  "Ward_Address",
  "City",
  "QR_Code"
];

let currentField = 0;
let rois = {};
let drawnBoxes = [];

function updateStatus() {
  statusDiv.innerText = `Draw bounding box for: ${fields[currentField]}`;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  img.src = URL.createObjectURL(file);
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
});

canvas.addEventListener("mousedown", (e) => {
  drawing = true;
  startX = e.offsetX;
  startY = e.offsetY;
});

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  endX = e.offsetX;
  endY = e.offsetY;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  drawPreviousBoxes();

  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, endX - startX, endY - startY);
});

canvas.addEventListener("mouseup", () => {
  drawing = false;

  const x1 = Math.min(startX, endX);
  const y1 = Math.min(startY, endY);
  const x2 = Math.max(startX, endX);
  const y2 = Math.max(startY, endY);

  const normBox = [
    x1 / canvas.width,
    y1 / canvas.height,
    x2 / canvas.width,
    y2 / canvas.height
  ];

  rois[fields[currentField]] = normBox;
  drawnBoxes.push({ field: fields[currentField], box: [x1, y1, x2, y2] });

  redoBtn.hidden = false;
  nextBtn.hidden = false;
});

function drawPreviousBoxes() {
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  drawnBoxes.forEach(b => {
    const [x1, y1, x2, y2] = b.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });
}

redoBtn.onclick = () => {
  delete rois[fields[currentField]];
  drawnBoxes.pop();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  drawPreviousBoxes();
};

nextBtn.onclick = () => {
  currentField++;
  redoBtn.hidden = true;
  nextBtn.hidden = true;

  if (currentField === fields.length) {
    okBtn.hidden = false;
    statusDiv.innerText = "All fields done. Click OK.";
  } else {
    updateStatus();
  }
};

okBtn.onclick = async () => {
  const formData = new FormData();
  formData.append("plate_type", "manual");
  formData.append("image", imageInput.files[0]);
  formData.append("rois", JSON.stringify(rois));

  const res = await fetch("http://localhost:8001/ocr", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  outputPre.textContent = JSON.stringify(data, null, 2);
};

function start() {
  rois = {};
  drawnBoxes = [];
  currentField = 0;
  okBtn.hidden = true;
  outputPre.textContent = "";

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  if (plateTypeSelect.value === "manual") {
    updateStatus();
  } else {
    runDefaultOCR();
  }
}

async function runDefaultOCR() {
  const formData = new FormData();
  formData.append("plate_type", "default");
  formData.append("image", imageInput.files[0]);

  const res = await fetch("http://localhost:8001/ocr", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  outputPre.textContent = JSON.stringify(data, null, 2);
}
