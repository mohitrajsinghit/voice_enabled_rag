import React, { useState, useRef, useCallback } from 'react';
import { queryWithAudio } from '../api';

/**
 * Mic icon SVG component.
 */
function MicIcon({ recording }) {
  return (
    <svg
      className="mic-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke={recording ? '#ef4444' : 'currentColor'}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

/**
 * Stop icon SVG.
 */
function StopIcon() {
  return (
    <svg className="mic-icon" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

/**
 * Recorder component: captures mic audio via MediaRecorder API.
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
        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());

        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

        if (audioBlob.size === 0) {
          onStatusChange?.('error', 'No audio recorded');
          return;
        }

        onStatusChange?.('transcribing', 'Processing audio...');

        try {
          const result = await queryWithAudio(audioBlob);
          onResult?.(result);
          onStatusChange?.('done', 'Complete');
        } catch (error) {
          onStatusChange?.('error', error.message);
        }
      };

      mediaRecorder.start(250); // Collect data every 250ms
      setRecording(true);
      startTimeRef.current = Date.now();
      onStatusChange?.('recording', 'Recording...');

      // Update duration timer
      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 100);
    } catch (error) {
      console.error('Failed to start recording:', error);
      onStatusChange?.('error', 'Microphone access denied');
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
    <div className="recorder-section">
      <button
        className={`record-btn ${recording ? 'recording' : ''}`}
        onClick={handleClick}
        disabled={disabled && !recording}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
        id="record-button"
      >
        {recording ? <StopIcon /> : <MicIcon recording={recording} />}
      </button>

      {recording && (
        <span className="recording-timer">
          🔴 {formatTime(duration)}
        </span>
      )}
    </div>
  );
}
