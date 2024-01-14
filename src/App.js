import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
// import '@tensorflow/tfjs-backend-wasm';
import { drawKeypoints, drawSkeletonLines, calculateAngle, colors } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [drawSkeleton, setDrawSkeleton] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(true);


  // Angle state varibles
  const [showLeftKneeAngle, setShowLeftKneeAngle] = useState(false);
  const [showRightKneeAngle, setShowRightKneeAngle] = useState(false);
  const [showLeftHipAngle, setShowLeftHipAngle] = useState(false);
  const [showRightHipAngle, setShowRightHipAngle] = useState(false);
  const [showLeftElbowAngle, setShowLeftElbowAngle] = useState(false);
  const [showRightElbowAngle, setShowRightElbowAngle] = useState(false);
  const [showLeftShoulderAngle, setShowLeftShoulderAngle] = useState(false);
  const [showRightShoulderAngle, setShowRightShoulderAngle] = useState(false);
  // Declare these state variables at the top with useState
  const [angleLeftKnee, setLeftKneeAngle] = useState(null);
  const [angleRightKnee, setRightKneeAngle] = useState(null);
  const [angleLeftHip, setLeftHipAngle] = useState(null);
  const [angleRightHip, setRightHipAngle] = useState(null);
  const [angleLeftElbow, setLeftElbowAngle] = useState(null);
  const [angleRightElbow, setRightElbowAngle] = useState(null);
  const [angleLeftShoulder, setLeftShoulderAngle] = useState(null);
  const [angleRightShoulder, setRightShoulderAngle] = useState(null);
  // Counter 
  const [squatCount, setSquatCount] = useState(0);

  // Load MoveNet model
  useEffect(() => {
    const loadModel = async () => {
      setIsLoadingModel(true);
      // await tf.setBackend('wasm');
      await tf.setBackend('webgl');
      await tf.ready();
      const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
      const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
      setModel(detector);
      setIsLoadingModel(false);
    };
    loadModel();
  }, []);

  const trailLength = 5; // Length of the trail
  const updateInterval = 10; // Save point every 10 frames

  // Initialize state for trails
  const [leftKneeTrail, setLeftKneeTrail] = useState({ positions: [], show: false });
  const [rightKneeTrail, setRightKneeTrail] = useState({ positions: [], show: false });
  const [leftHipTrail, setLeftHipTrail] = useState({ positions: [], show: false });
  const [rightHipTrail, setRightHipTrail] = useState({ positions: [], show: false });
  const [leftElbowTrail, setLeftElbowTrail] = useState({ positions: [], show: false });
  const [rightElbowTrail, setRightElbowTrail] = useState({ positions: [], show: false });
  const [leftShoulderTrail, setLeftShoulderTrail] = useState({ positions: [], show: false });
  const [rightShoulderTrail, setRightShoulderTrail] = useState({ positions: [], show: false });
  let frameCounter = useRef(0);

  let currentlyInSquat = useRef(false);

  function drawResults(poses) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = "100px Roboto Condensed";

    frameCounter.current++;

    poses.forEach(pose => {
      if (frameCounter.current % updateInterval === 0) {
        // Save keypoints every nth frame
        const leftKnee = pose.keypoints[13];
        const rightKnee = pose.keypoints[14];
        const leftHip = pose.keypoints[11];
        const rightHip = pose.keypoints[12];
        const leftElbow = pose.keypoints[7];
        const rightElbow = pose.keypoints[8];
        const leftShoulder = pose.keypoints[5];
        const rightShoulder = pose.keypoints[6];

        const updateTrail = (keypoint, trailState) => {
          if (keypoint.score > 0.5) {
            trailState.setTrail(prevTrail => ({
              ...prevTrail,
              positions: [...prevTrail.positions, keypoint].slice(-trailLength)
            }));
          } else {
            trailState.setTrail(prevTrail => ({ ...prevTrail, positions: [] })); // Clear history if keypoint is not detected
          }
        };
        updateTrail(leftKnee, { setTrail: setLeftKneeTrail, trail: leftKneeTrail });
        updateTrail(rightKnee, { setTrail: setRightKneeTrail, trail: rightKneeTrail });
        updateTrail(leftHip, { setTrail: setLeftHipTrail, trail: leftHipTrail });
        updateTrail(rightHip, { setTrail: setRightHipTrail, trail: rightHipTrail });
        updateTrail(leftElbow, { setTrail: setLeftElbowTrail, trail: leftElbowTrail });
        updateTrail(rightElbow, { setTrail: setRightElbowTrail, trail: rightElbowTrail });
        updateTrail(leftShoulder, { setTrail: setLeftShoulderTrail, trail: leftShoulderTrail });
        updateTrail(rightShoulder, { setTrail: setRightShoulderTrail, trail: rightShoulderTrail });
      }

      const rgbValues = colors.trailColor.match(/\d+/g); // This will extract ["255", "255", "255"]

      // Now convert these string values to numbers
      const [r, g, b] = rgbValues.map(Number);

      // Use these values in your strokeStyle with the desired opacity
      const drawTrail = (trail) => {
        for (let i = 0; i < trail.length - 1; i++) {
          const opacity = 1 - (i / trail.length);
          ctx.beginPath();
          ctx.moveTo(trail[i].x, trail[i].y);
          ctx.lineTo(trail[i + 1].x, trail[i + 1].y);
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
          ctx.stroke();
        }
      };

      if (leftKneeTrail.show) {
        drawTrail(leftKneeTrail.positions);
      }
      if (rightKneeTrail.show) {
        drawTrail(rightKneeTrail.positions);
      }
      if (leftHipTrail.show) {
        drawTrail(leftKneeTrail.positions);
      }
      if (rightHipTrail.show) {
        drawTrail(rightKneeTrail.positions);
      }
      if (leftElbowTrail.show) {
        drawTrail(leftElbowTrail.positions);
      }
      if (rightElbowTrail.show) {
        drawTrail(rightElbowTrail.positions);
      }
      if (leftShoulderTrail.show) {
        drawTrail(leftShoulderTrail.positions);
      }
      if (rightShoulderTrail.show) {
        drawTrail(rightShoulderTrail.positions);
      }


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

        // Left Elbow Angle
        const leftElbow = pose.keypoints[7];
        const leftWrist = pose.keypoints[9];
        if (leftShoulder.score > 0.5 && leftElbow.score > 0.5 && leftWrist.score > 0.5) {
          const angleLeftElbow = calculateAngle(leftShoulder, leftElbow, leftWrist);
          if (showLeftElbowAngle) {
            ctx.fillText(` ${angleLeftElbow.toFixed(0)}°`, leftElbow.x, leftElbow.y);
          }
        }

        // Right Elbow Angle
        const rightElbow = pose.keypoints[8];
        const rightWrist = pose.keypoints[10];
        if (rightShoulder.score > 0.5 && rightElbow.score > 0.5 && rightWrist.score > 0.5) {
          const angleRightElbow = calculateAngle(rightShoulder, rightElbow, rightWrist);
          if (showRightElbowAngle) {
            ctx.fillText(` ${angleRightElbow.toFixed(0)}°`, rightElbow.x, rightElbow.y);
          }
        }

        // Left Shoulder Angle
        if (leftHip.score > 0.5 && leftShoulder.score > 0.5 && leftElbow.score > 0.5) {
          const angleLeftShoulder = calculateAngle(leftHip, leftShoulder, leftElbow);
          if (showLeftShoulderAngle) {
            ctx.fillText(` ${angleLeftShoulder.toFixed(0)}°`, leftShoulder.x, leftShoulder.y);
          }
        }

        // Right Shoulder Angle
        if (rightHip.score > 0.5 && rightShoulder.score > 0.5 && rightElbow.score > 0.5) {
          const angleRightShoulder = calculateAngle(rightHip, rightShoulder, rightElbow);
          if (showRightShoulderAngle) {
            ctx.fillText(` ${angleRightShoulder.toFixed(0)}°`, rightShoulder.x, rightShoulder.y);
          }
        }

        setLeftKneeAngle(angleLeftKnee);
        setRightKneeAngle(angleRightKnee);
        setLeftHipAngle(angleLeftHip);
        setRightHipAngle(angleRightHip);
        setLeftElbowAngle(angleLeftElbow);
        setRightElbowAngle(angleRightElbow);
        setLeftShoulderAngle(angleLeftShoulder);
        setRightShoulderAngle(angleRightShoulder);

        // Calculate squat status
        let isUserInSquat = angleLeftKnee <= 90 && angleRightKnee <= 90;
        if (isUserInSquat && !currentlyInSquat.current) {
          currentlyInSquat.current = true;
        } else if (!isUserInSquat && currentlyInSquat.current) {
          setSquatCount(prevCount => prevCount + 1);
          currentlyInSquat.current = false;
        }
      }
    });
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
      }, 10);
      return () => clearInterval(interval);
    }
  }, [isCameraActive, detectPose]);

  const joints = [
    { name: "Left Knee", showAngle: showLeftKneeAngle, setShowAngle: setShowLeftKneeAngle, trail: leftKneeTrail, setTrail: setLeftKneeTrail },
    { name: "Right Knee", showAngle: showRightKneeAngle, setShowAngle: setShowRightKneeAngle, trail: rightKneeTrail, setTrail: setRightKneeTrail },
    { name: "Left Hip", showAngle: showLeftHipAngle, setShowAngle: setShowLeftHipAngle, trail: leftHipTrail, setTrail: setLeftHipTrail },
    { name: "Right Hip", showAngle: showRightHipAngle, setShowAngle: setShowRightHipAngle, trail: rightHipTrail, setTrail: setRightHipTrail },
    { name: "Left Elbow", showAngle: showLeftElbowAngle, setShowAngle: setShowLeftElbowAngle, trail: leftElbowTrail, setTrail: setLeftElbowTrail },
    { name: "Right Elbow", showAngle: showRightElbowAngle, setShowAngle: setShowRightElbowAngle, trail: rightElbowTrail, setTrail: setRightElbowTrail },
    { name: "Left Shoulder", showAngle: showLeftShoulderAngle, setShowAngle: setShowLeftShoulderAngle, trail: leftShoulderTrail, setTrail: setLeftShoulderTrail },
    { name: "Right Shoulder", showAngle: showRightShoulderAngle, setShowAngle: setShowRightShoulderAngle, trail: rightShoulderTrail, setTrail: setRightShoulderTrail }
  ];

  // Remember to define the corresponding useState hooks for each new joint angle and trail visibility state.



  return (
    <div className="App">
      <header className="App-header">
        {isLoadingModel && <p>Loading model, please wait...</p>}
        <div style={{ position: 'relative', width: '640px', height: '480px' }}>
          {isCameraActive && <Webcam ref={webcamRef} style={{ width: '100%', height: '100%' }} />}
          <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
        </div>
        <button onClick={toggleCamera}>{isCameraActive ? "Stop Camera" : "Start Camera"}</button>
        <button onClick={toggleSkeletonDrawing}>{drawSkeleton ? "Hide Skeleton" : "Show Skeleton"}</button>
        <div style={{ marginTop: '10px' }}>
          {joints.map((joint, index) => (
            <div key={index}>
              <label>
                <input type="checkbox" checked={joint.showAngle} onChange={() => joint.setShowAngle(!joint.showAngle)} />
                {joint.name} Angle
              </label>
              <label>
                <input type="checkbox" checked={joint.trail.show} onChange={() => joint.setTrail({ ...joint.trail, show: !joint.trail.show })} />
                {joint.name} Trail
              </label>
            </div>
          ))}
        </div>
        {/* <p>Squat Count: {squatCount}</p> */}
      </header>
    </div>
  );
}

export default App;