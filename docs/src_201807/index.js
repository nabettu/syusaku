import * as posenet from "@tensorflow-models/posenet";
import * as tf from "@tensorflow/tfjs";
import body_png from "./img/body.png";
import left_arm_png from "./img/left_arm.png";
import leftAnkle_png from "./img/leftAnkle.png";
import leftKnee_png from "./img/leftKnee.png";
import leftShoulder_png from "./img/leftShoulder.png";
import right_arm_png from "./img/right_arm.png";
import rightAnkle_png from "./img/rightAnkle.png";
import rightKnee_png from "./img/rightKnee.png";
import rightShoulder_png from "./img/rightShoulder.png";
import {
  drawKeypoints,
  drawPoint,
  drawSegment,
  drawSkeleton,
  renderImageToCanvas
} from "./util";
const { partIds, poseChain } = posenet;

const canvas = document.querySelector("#output");
const fileInput = document.querySelector("#fileInput");
const outputStride = 16;
let image = null;
let modelOutputs = null;

function drawResults(canvas, poses, minPartConfidence, minPoseConfidence) {
  renderImageToCanvas(image, [image.width, image.height], canvas);
  poses.forEach(pose => {
    if (pose.score >= minPoseConfidence) {
      drawKeypoints(pose.keypoints, minPartConfidence, canvas.getContext("2d"));
      drawSkeleton(pose.keypoints, minPartConfidence, canvas.getContext("2d"));
    }
  });
  setStatusText("");
}

function drawSinglePoseResults(pose) {
  drawResults(canvas, [pose], 0.5, 0.5);
}
function d2p(p1, p2) {
  return Math.sqrt(
    Math.pow(p2.position.x - p1.position.x, 2) +
      Math.pow(p2.position.y - p1.position.y, 2)
  );
}
function r2p(p1, p2) {
  return Math.atan2(
    p2.position.y - p1.position.y,
    p2.position.x - p1.position.x
  );
}
async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = "";
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = imagePath;
  return promise;
}

function drawOffsetVector(
  ctx,
  y,
  x,
  outputStride,
  offsetsVectorY,
  offsetsVectorX
) {
  drawSegment(
    [y * outputStride, x * outputStride],
    [y * outputStride + offsetsVectorY, x * outputStride + offsetsVectorX],
    "red",
    1,
    ctx
  );
}

function drawDisplacementEdgesFrom(
  ctx,
  partId,
  displacements,
  outputStride,
  edges,
  y,
  x,
  offsetsVectorY,
  offsetsVectorX
) {
  const numEdges = displacements.shape[2] / 2;

  const offsetX = x * outputStride + offsetsVectorX;
  const offsetY = y * outputStride + offsetsVectorY;

  const edgeIds = edges[partId] || [];

  if (edgeIds.length > 0) {
    edgeIds.forEach(edgeId => {
      const displacementY = displacements.get(y, x, edgeId);
      const displacementX = displacements.get(y, x, edgeId + numEdges);

      drawSegment(
        [offsetY, offsetX],
        [offsetY + displacementY, offsetX + displacementX],
        "blue",
        1,
        ctx
      );
    });
  }
}

async function decodeSinglePoseAndDrawResults() {
  if (!modelOutputs) {
    return;
  }

  const pose = await posenet.decodeSinglePose(
    modelOutputs.heatmapScores,
    modelOutputs.offsets,
    outputStride
  );

  drawSinglePoseResults(pose);
}

function decodeSingleAndMultiplePoses() {
  decodeSinglePoseAndDrawResults();
}

function setStatusText(text) {
  const resultElement = document.getElementById("statusText");
  resultElement.innerText = text;
}

function disposeModelOutputs() {
  if (modelOutputs) {
    modelOutputs.heatmapScores.dispose();
    modelOutputs.offsets.dispose();
    modelOutputs.displacementFwd.dispose();
    modelOutputs.displacementBwd.dispose();
  }
}

async function testImageAndEstimatePoses(net, img) {
  setStatusText("姿勢検出中...");
  disposeModelOutputs();
  image = await loadImage(img);
  const input = tf.fromPixels(image);
  modelOutputs = await net.predictForMultiPose(input, outputStride);
  await decodeSingleAndMultiplePoses();
  setStatusText("");
  input.dispose();
}

export async function bindPage(img) {
  const net = await posenet.load();
  await testImageAndEstimatePoses(net, img);
  setStatusText("loading...");
}

fileInput.addEventListener("change", evt => {
  console.log(evt.target.files);
  const f = evt.target.files[0];
  // image =
  var reader = new FileReader();
  // Closure to capture the file information.
  reader.onload = () => {
    bindPage(reader.result);
  };
  reader.readAsDataURL(f);
});
