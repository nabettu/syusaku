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
  // renderBodyPartsToCanvas(poses, canvas.getContext("2d"));
}

async function renderBodyPartsToCanvas(poses, ctx) {
  console.log(JSON.stringify(poses[0].keypoints));
  const p = poses[0].keypoints;
  ctx.save();
  //body
  const body = await loadImage(body_png);
  let size = d2p(p[3], p[4]) / 80; // 顔のサイズをもとに
  let rotate = -r2p(p[3], p[4]) * Math.PI / 180;
  // console.log(-r2p(p[3], p[4]) * (180 / Math.PI));
  // ctx.translate(p[0].position.x - 100 * size, p[0].position.y - 100 * size);
  // // ctx.rotate(rotate);
  // ctx.drawImage(body, 0, 0, body.width * size, body.height * size);
  // ctx.translate(
  //   -(p[0].position.x - 100 * size),
  //   -(p[0].position.y - 100 * size)
  // );
  // // ctx.rotate(-rotate);
  // ctx.restore();
  // ctx.save();

  //rightShoulder
  const rightShoulder = await loadImage(rightShoulder_png);
  size = d2p(p[6], p[8]) / 100;
  rotate = r2p(p[6], p[8]) * Math.PI / 180;
  console.log(-r2p(p[6], p[8]) * (180 / Math.PI));
  ctx.translate(p[6].position.x, p[6].position.y);
  ctx.rotate(rotate);
  ctx.drawImage(
    rightShoulder,
    -45 * size,
    -45 * size,
    rightShoulder.width * size,
    rightShoulder.height * size
  );
  ctx.rotate(-rotate);
  ctx.translate(-p[6].position.x, -p[6].position.y);
  ctx.restore();
  ctx.save();
  //right right_arm
  const rightArm = await loadImage(right_arm_png);
  size = d2p(p[8], p[10]) / 100;
  rotate = r2p(p[8], p[10]) * Math.PI / 180;
  console.log(-r2p(p[8], p[10]) * (180 / Math.PI));
  ctx.translate(p[8].position.x, p[8].position.y);
  ctx.rotate(rotate);
  ctx.translate(-p[8].position.x, -p[8].position.y);
  ctx.drawImage(
    rightArm,
    -30 * size,
    -30 * size,
    rightArm.width * size,
    rightArm.height * size
  );
  ctx.rotate(-rotate);
  ctx.restore();
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
