import express from "express";
import multer from "multer";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import { createCanvas, loadImage } from "canvas";
import serverless from "serverless-http";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const modelPromise = (async () => {
  console.log("Loading PoseNet model...");
  const model = await posenet.load({
    architecture: "MobileNetV1",
    outputStride: 16,
    inputResolution: { width: 257, height: 200 },
    multiplier: 0.75,
  });
  console.log("✅ PoseNet model loaded!");
  return model;
})();

app.get("/", (req, res) => {
  res.json({ message: "PoseNet API ready", endpoint: "/pose" });
});

// Helper to calculate Euclidean distance
function euclideanDistance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// Main endpoint
app.post("/pose", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    const net = await modelPromise;
    const image = await loadImage(req.file.buffer);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);

    // Estimate pose
    const pose = await net.estimateSinglePose(canvas, {
      flipHorizontal: false,
    });

    // Get keypoints
    const kp = {};
    for (let point of pose.keypoints) {
      if (["nose", "leftWrist", "rightWrist"].includes(point.part)) {
        kp[point.part] = point.position;
      }
    }

    if (!kp.nose) {
      return res.status(400).json({ error: "Nose not detected" });
    }

    const distLeft = kp.leftWrist
      ? euclideanDistance(kp.leftWrist, kp.nose)
      : Infinity;
    const distRight = kp.rightWrist
      ? euclideanDistance(kp.rightWrist, kp.nose)
      : Infinity;
    const minDist = Math.min(distLeft, distRight);
    const closestHand = minDist === distLeft ? "leftWrist" : "rightWrist";

    const THRESHOLD_DISTANCE = 550;
    const confidenceDecimal = Math.max(0, 1 - minDist / THRESHOLD_DISTANCE);
    const ACCEPTANCE_THRESHOLD = 0.1;

    const result = {
      closestHand,
      distance: parseFloat(minDist.toFixed(2)),
      accepted: confidenceDecimal >= ACCEPTANCE_THRESHOLD,
    };

    return res.json(result);
  } catch (error) {
    console.error("Pose estimation error:", error);
    res.status(500).json({ error: "Pose estimation failed" });
  }
});

export default serverless(app);
