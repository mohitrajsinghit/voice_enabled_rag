import React, { useState, useRef, useCallback } from 'react';
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
 * Luxury Voice Orb Recorder Component
 */
export default function Recorder({ onResult, onStatusChange, disabled }) {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

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
          Click the glowing orb to speak your question
        </span>
      )}
    </div>
  );
}
