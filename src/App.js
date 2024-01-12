import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { drawKeypoints, drawSkeletonLines, calculateAngle } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [drawSkeleton, setDrawSkeleton] = useState(true); // State to control skeleton drawing

  // Angle state varibles
  const [showLeftKneeAngle, setShowLeftKneeAngle] = useState(false);
  const [showRightKneeAngle, setShowRightKneeAngle] = useState(false);
  const [showLeftHipAngle, setShowLeftHipAngle] = useState(false);
  const [showRightHipAngle, setShowRightHipAngle] = useState(false);
  // Declare these state variables at the top with useState
  const [angleLeftKnee, setLeftKneeAngle] = useState(null);
  const [angleRightKnee, setRightKneeAngle] = useState(null);
  const [angleLeftHip, setLeftHipAngle] = useState(null);
  const [angleRightHip, setRightHipAngle] = useState(null);


  // Load MoveNet model
  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
      const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
      setModel(detector);
    };
    loadModel();
  }, []);

  function drawResults(poses) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    for (const pose of poses) {
      if (drawSkeleton) {
        drawKeypoints(pose.keypoints, 0.5, ctx);
        drawSkeletonLines(pose.keypoints, 0.5, ctx);

        // Left Knee Angle
        const leftHip = pose.keypoints[11];
        const leftKnee = pose.keypoints[13];
        const leftAnkle = pose.keypoints[15];
        if (leftHip.score > 0.5 && leftKnee.score > 0.5 && leftAnkle.score > 0.5) {
          const angleLeftKnee = calculateAngle(leftHip, leftKnee, leftAnkle);
          if (showLeftKneeAngle) {
            ctx.fillText(` ${angleLeftKnee.toFixed(0)}°`, leftKnee.x, leftKnee.y);
          }
        }

        // Right Knee Angle
        const rightHip = pose.keypoints[12];
        const rightKnee = pose.keypoints[14];
        const rightAnkle = pose.keypoints[16];
        if (rightHip.score > 0.5 && rightKnee.score > 0.5 && rightAnkle.score > 0.5) {
          const angleRightKnee = calculateAngle(rightHip, rightKnee, rightAnkle);
          if (showRightKneeAngle) {
            ctx.fillText(` ${angleRightKnee.toFixed(0)}°`, rightKnee.x, rightKnee.y);
          }
        }

        // Left Hip Angle
        const leftShoulder = pose.keypoints[5];
        if (leftShoulder.score > 0.5 && leftHip.score > 0.5 && leftKnee.score > 0.5) {
          const angleLeftHip = calculateAngle(leftShoulder, leftHip, leftKnee);
          if (showLeftHipAngle) {
            ctx.fillText(` ${angleLeftHip.toFixed(0)}°`, leftHip.x, leftHip.y);
          }
        }

        // Right Hip Angle
        const rightShoulder = pose.keypoints[6];
        if (rightShoulder.score > 0.5 && rightHip.score > 0.5 && rightKnee.score > 0.5) {
          const angleRightHip = calculateAngle(rightShoulder, rightHip, rightKnee);
          if (showRightHipAngle) {
            ctx.fillText(` ${angleRightHip.toFixed(0)}°`, rightHip.x, rightHip.y);
          }
        }
        // Inside drawResults function
        setLeftKneeAngle(angleLeftKnee);
        setRightKneeAngle(angleRightKnee);
        setLeftHipAngle(angleLeftHip);
        setRightHipAngle(angleRightHip);

      }
    }
  }

  // Pose detection function - Memoize with useCallback
  const detectPose = useCallback(async () => {
    if (webcamRef.current && model) {
      const video = webcamRef.current.video;
      const poses = await model.estimatePoses(video, { flipHorizontal: false });
      drawResults(poses);
    }
  }, [model, drawResults]);


  // Function to clear the canvas
  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  };

  // Toggle camera feed
  const toggleCamera = () => {
    setIsCameraActive(!isCameraActive);

    if (isCameraActive) {
      // If turning off the camera, clear the canvas
      clearCanvas();
    }
  };

  // Toggle skeleton drawing
  const toggleSkeletonDrawing = () => {
    setDrawSkeleton(!drawSkeleton);
  };

  // Run pose detection only when camera is active
  useEffect(() => {
    if (!isCameraActive) return;

    const video = webcamRef.current && webcamRef.current.video;
    if (video && video.readyState === 4) {
      // Video is ready, set canvas size and start pose detection
      canvasRef.current.width = video.videoWidth;
      canvasRef.current.height = video.videoHeight;

      const interval = setInterval(() => {
        detectPose();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isCameraActive, detectPose]);

  return (
    <div className="App">
      <header className="App-header">
        <div style={{ position: 'relative', width: '640px', height: '480px' }}>
          {isCameraActive && <Webcam ref={webcamRef} style={{ width: '100%', height: '100%' }} />}
          <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
        </div>
        <button onClick={toggleCamera}>
          {isCameraActive ? "Stop Camera" : "Start Camera"}
        </button>
        <button onClick={toggleSkeletonDrawing}>
          {drawSkeleton ? "Hide Skeleton" : "Show Skeleton"}
        </button>
        <div style={{ marginTop: '10px' }}>
          <label>
            <input type="checkbox" checked={showLeftKneeAngle} onChange={() => setShowLeftKneeAngle(!showLeftKneeAngle)} />
            Left Knee
          </label>
          <label>
            <input type="checkbox" checked={showRightKneeAngle} onChange={() => setShowRightKneeAngle(!showRightKneeAngle)} />
            Right Knee
          </label>
          <label>
            <input type="checkbox" checked={showLeftHipAngle} onChange={() => setShowLeftHipAngle(!showLeftHipAngle)} />
            Left Hip
          </label>
          <label>
            <input type="checkbox" checked={showRightHipAngle} onChange={() => setShowRightHipAngle(!showRightHipAngle)} />
            Right Hip
          </label>
        </div>
      </header>
    </div>
  );
}

export default App;