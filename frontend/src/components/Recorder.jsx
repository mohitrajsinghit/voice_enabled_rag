import React, { useState, useRef, useCallback, useEffect } from 'react';
import { queryWithAudio } from '../api';

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
 * Luxury Voice Orb Recorder Component with real-time waveform
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
          onStatusChange?.('error', error.message || 'Voice search failed');
        }
      };

      mediaRecorder.start(250);
      setRecording(true);
      startTimeRef.current = Date.now();
      onStatusChange?.('recording', 'Listening... Speak in any of 14 Indic languages or English');

      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 100);
    } catch (error) {
      console.error('Failed to start recording:', error);
      onStatusChange?.('error', 'Microphone access denied or not available');
    }
  }, [onResult, onStatusChange]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setDuration(0);
      clearInterval(timerRef.current);
    }
  }, [recording]);

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
        <div className="recording-hud">
          <span className="rec-pulse-dot" />
          <span>RECORDING • {formatTime(duration)}</span>
        </div>
      ) : (
        <span className="voice-hint">
          Click the mic to speak your question
        </span>
      )}
    </div>
  );
}
