import React, { useState, useRef, useCallback, useEffect } from 'react';
import { queryWithAudio } from '../api';

const MAX_RECORDING_SECONDS = 60; // 1 minute max voice duration

/**
 * Modern Microphone SVG Icon
 */
function MicIcon() {
  return (
    <svg className="orb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

/**
 * Modern Stop SVG Icon
 */
function StopIcon() {
  return (
    <svg className="orb-icon" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="3" />
    </svg>
  );
}

/**
 * Dynamic 60-Second Circular Countdown Ring (Green -> Yellow -> Red)
 */
function CountdownRing({ duration, maxDuration = 60 }) {
  const size = 118;
  const strokeWidth = 4;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(duration / maxDuration, 1);
  const strokeDashoffset = circumference * (1 - progress);

  let ringColor = '#10b981'; // Green (0 - 35s)
  let glowColor = 'rgba(16, 185, 129, 0.5)';
  if (duration >= 50) {
    ringColor = '#ef4444'; // Red (50 - 60s)
    glowColor = 'rgba(239, 68, 68, 0.7)';
  } else if (duration >= 35) {
    ringColor = '#f59e0b'; // Yellow / Amber (35 - 50s)
    glowColor = 'rgba(245, 158, 11, 0.6)';
  }

  return (
    <svg
      className="orb-countdown-ring"
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%) rotate(-90deg)',
        pointerEvents: 'none',
        zIndex: 4,
        filter: `drop-shadow(0 0 8px ${glowColor})`,
      }}
    >
      {/* Background track */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="transparent"
        stroke="rgba(255, 255, 255, 0.08)"
        strokeWidth={strokeWidth}
      />
      {/* Animated countdown progress ring */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="transparent"
        stroke={ringColor}
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={strokeDashoffset}
        strokeLinecap="round"
        style={{
          transition: 'stroke-dashoffset 0.15s linear, stroke 0.3s ease',
        }}
      />
    </svg>
  );
}

/**
 * Real-time Audio Waveform Ring — renders actual microphone amplitude
 */
function WaveformRing({ analyserRef }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    const size = 160;
    canvas.width = size;
    canvas.height = size;

    const BAR_COUNT = 48;
    const CENTER = size / 2;
    const INNER_R = 48;
    const MAX_BAR_H = 22;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const draw = () => {
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, size, size);

      for (let i = 0; i < BAR_COUNT; i++) {
        const dataIdx = Math.floor((i / BAR_COUNT) * dataArray.length);
        const amplitude = dataArray[dataIdx] / 255;
        const barHeight = Math.max(3, amplitude * MAX_BAR_H);
        const angle = (i / BAR_COUNT) * Math.PI * 2 - Math.PI / 2;

        const x1 = CENTER + Math.cos(angle) * INNER_R;
        const y1 = CENTER + Math.sin(angle) * INNER_R;
        const x2 = CENTER + Math.cos(angle) * (INNER_R + barHeight);
        const y2 = CENTER + Math.sin(angle) * (INNER_R + barHeight);

        const alpha = 0.4 + amplitude * 0.6;
        ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [analyserRef]);

  return (
    <canvas
      ref={canvasRef}
      className="waveform-ring-canvas"
      aria-hidden="true"
    />
  );
}

/**
 * Luxury Voice Orb Recorder Component with real-time waveform & 60s countdown ring
 */
export default function Recorder({ onResult, onStatusChange, disabled }) {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);
  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
    setDuration(0);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Set up Web Audio API analyser for waveform
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      });

      chunksRef.current = [];
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop audio tracks
        stream.getTracks().forEach((track) => track.stop());
        if (audioCtxRef.current) {
          audioCtxRef.current.close();
          audioCtxRef.current = null;
        }
        analyserRef.current = null;

        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

        if (audioBlob.size === 0) {
          onStatusChange?.('error', 'No audio recorded');
          return;
        }

        onStatusChange?.('transcribing', 'Transcribing audio via Sarvam STT...');

        try {
          const result = await queryWithAudio(audioBlob);
          onResult?.(result);
          onStatusChange?.('done', 'Query completed');
        } catch (error) {
          onStatusChange?.('error', error.message || 'Voice search failed', error);
        }
      };

      mediaRecorder.start(250);
      setRecording(true);
      startTimeRef.current = Date.now();
      onStatusChange?.('recording', 'Listening... Speak in any of 14 Indic languages or English (max 1 min)');

      timerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setDuration(elapsed);

        // Auto-stop at 60 seconds (1 minute cap)
        if (elapsed >= MAX_RECORDING_SECONDS) {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
          }
          setRecording(false);
          clearInterval(timerRef.current);
        }
      }, 100);
    } catch (error) {
      console.error('Failed to start recording:', error);
      onStatusChange?.('error', 'Microphone access denied or not available');
    }
  }, [onResult, onStatusChange]);

  const handleClick = useCallback(() => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecording]);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const getHudClass = () => {
    if (duration >= 50) return 'critical';
    if (duration >= 35) return 'warning';
    return 'safe';
  };

  return (
    <div className="voice-visualizer-container">
      <div className={`orb-wrapper ${recording ? 'active' : ''}`}>
        <div className="orb-glow-ring" />
        <div className="orb-ripple-1" />
        <div className="orb-ripple-2" />

        {/* Real-time waveform ring (only while recording) */}
        {recording && analyserRef.current && (
          <WaveformRing analyserRef={analyserRef} />
        )}

        {/* 60-Second Countdown Outer Ring with Green/Yellow/Red indicator */}
        {recording && (
          <CountdownRing duration={duration} maxDuration={MAX_RECORDING_SECONDS} />
        )}

        <button
          className={`voice-orb-btn ${recording ? 'recording' : ''}`}
          onClick={handleClick}
          disabled={disabled && !recording}
          aria-label={recording ? 'Stop recording voice' : 'Start speaking'}
          id="record-button"
        >
          {recording ? <StopIcon /> : <MicIcon />}
        </button>
      </div>

      {recording ? (
        <div className={`recording-hud ${getHudClass()}`}>
          <span className="rec-pulse-dot" />
          <span>RECORDING • {formatTime(duration)} / 01:00</span>
          {duration >= 50 && (
            <span className="time-warning-tag">({MAX_RECORDING_SECONDS - duration}s left)</span>
          )}
        </div>
      ) : (
        <span className="voice-hint">
          Click the mic to speak your question (max 1 min)
        </span>
      )}
    </div>
  );
}
