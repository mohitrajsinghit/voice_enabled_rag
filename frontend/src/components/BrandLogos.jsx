import React, { useState } from 'react';

/**
 * High-definition Vector Brand Logos with fallback safety
 */

export function HHGoaLogo({ className = 'logo-img-hhgoa' }) {
  const [imgError, setImgError] = useState(false);

  if (!imgError) {
    return (
      <img
        src="/assets/hhgoa.webp"
        alt="Hacker House Goa"
        className={className}
        onError={() => setImgError(true)}
      />
    );
  }

  // Fallback high-fidelity SVG crest if file is missing
  return (
    <svg className={className} viewBox="0 0 160 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="hhgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="50%" stopColor="#a855f7" />
          <stop offset="100%" stopColor="#ec4899" />
        </linearGradient>
        <filter id="hhgGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      {/* Background Badge */}
      <rect x="2" y="2" width="156" height="44" rx="10" fill="#0d0d26" stroke="url(#hhgGrad)" strokeWidth="1.5" />
      {/* Palm / Sun icon */}
      <circle cx="26" cy="24" r="14" fill="url(#hhgGrad)" opacity="0.2" />
      <path d="M26 14C23 18 20 22 17 26M26 14C29 18 32 22 35 26M26 14V34M22 20C24 22 26 23 26 23M30 20C28 22 26 23 26 23" stroke="#f43f5e" strokeWidth="2" strokeLinecap="round" />
      {/* Typography */}
      <text x="48" y="22" fill="#ffffff" fontFamily="Outfit, sans-serif" fontSize="13" fontWeight="800" letterSpacing="0.05em">HH GOA</text>
      <text x="48" y="34" fill="#a855f7" fontFamily="JetBrains Mono, monospace" fontSize="9" fontWeight="700" letterSpacing="0.08em">2026 EDITION</text>
      <circle cx="146" cy="14" r="3" fill="#10b981" filter="url(#hhgGlow)" />
    </svg>
  );
}

export function HackerHouseLogo({ className = 'logo-img-hh' }) {
  const [imgError, setImgError] = useState(false);

  if (!imgError) {
    return (
      <img
        src="/assets/hackerhouse.png"
        alt="Hacker House"
        className={className}
        onError={() => setImgError(true)}
      />
    );
  }

  // Fallback high-fidelity SVG cube logo
  return (
    <svg className={className} viewBox="0 0 140 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="hhCubeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#3b82f6" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="136" height="36" rx="8" fill="rgba(6, 182, 212, 0.08)" stroke="rgba(6, 182, 212, 0.3)" strokeWidth="1" />
      {/* 3D Isometric Cube */}
      <path d="M20 12L30 7L40 12L30 17L20 12Z" fill="#06b6d4" opacity="0.9" />
      <path d="M20 12L30 17V29L20 24V12Z" fill="#0284c7" opacity="0.8" />
      <path d="M40 12L30 17V29L40 24V12Z" fill="#38bdf8" opacity="0.6" />
      {/* Text */}
      <text x="48" y="24" fill="#f1f5f9" fontFamily="Outfit, sans-serif" fontSize="12" fontWeight="700" letterSpacing="0.04em">HACKERHOUSE</text>
    </svg>
  );
}
