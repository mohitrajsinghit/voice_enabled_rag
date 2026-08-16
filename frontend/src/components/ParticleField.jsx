import React, { useRef, useEffect, useCallback } from 'react';

/**
 * ParticleField — Interactive canvas particle system with auto-scattering,
 * anti-clumping inter-particle physics, dynamic constellation mesh, and mouse magnetism.
 */
export default function ParticleField() {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: -9999, y: -9999 });
  const particlesRef = useRef([]);
  const animFrameRef = useRef(null);

  const PARTICLE_COUNT = 165;
  const CONNECT_DIST = 140;
  const MOUSE_RADIUS = 180;
  const MOUSE_ATTRACT_FORCE = 0.03;
  const MOUSE_INNER_REPEL = 45;       // Prevents clustering directly on cursor
  const MIN_SEPARATION = 36;          // Anti-clumping distance between particles
  const MAX_SPEED = 2.4;

  const COLORS = [
    'rgba(99, 102, 241, 0.75)',  // electric indigo
    'rgba(139, 92, 246, 0.7)',   // neon purple
    'rgba(6, 182, 212, 0.75)',   // bright cyan
    'rgba(236, 72, 153, 0.6)',   // sunset pink
    'rgba(168, 85, 247, 0.65)',  // violet
    'rgba(56, 189, 248, 0.7)',   // sky blue
  ];

  const createParticle = useCallback((w, h) => {
    // Generate distinct random wander vectors
    const angle = Math.random() * Math.PI * 2;
    const speed = Math.random() * 0.4 + 0.25;
    const bvx = Math.cos(angle) * speed;
    const bvy = Math.sin(angle) * speed;

    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: bvx,
      vy: bvy,
      baseVx: bvx,
      baseVy: bvy,
      radius: Math.random() * 2.0 + 0.9,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      opacity: Math.random() * 0.55 + 0.35,
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let w = window.innerWidth;
    let h = window.innerHeight;

    const setSize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w;
      canvas.height = h;
    };
    setSize();

    // Init particles
    particlesRef.current = Array.from({ length: PARTICLE_COUNT }, () => createParticle(w, h));

    const handleMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    const handleMouseLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 };
    };

    window.addEventListener('resize', setSize);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);

    const animate = () => {
      ctx.clearRect(0, 0, w, h);

      const particles = particlesRef.current;
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;
      const hasMouse = mx > 0 && my > 0;

      // 1. Anti-clumping: repel particles if they get too close to each other
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const a = particles[i];
          const b = particles[j];
          const ddx = a.x - b.x;
          const ddy = a.y - b.y;
          const d = Math.sqrt(ddx * ddx + ddy * ddy);

          if (d < MIN_SEPARATION && d > 0.01) {
            // Repulsive force to scatter clusters apart
            const repelForce = (1 - d / MIN_SEPARATION) * 0.08;
            const fx = (ddx / d) * repelForce;
            const fy = (ddy / d) * repelForce;
            a.vx += fx;
            a.vy += fy;
            b.vx -= fx;
            b.vy -= fy;
          }
        }
      }

      // 2. Update particle positions & render
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        if (hasMouse) {
          const dx = mx - p.x;
          const dy = my - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < MOUSE_RADIUS && dist > 1) {
            if (dist > MOUSE_INNER_REPEL) {
              // Gentle attraction towards cursor
              p.vx += (dx / dist) * MOUSE_ATTRACT_FORCE;
              p.vy += (dy / dist) * MOUSE_ATTRACT_FORCE;
            } else {
              // Soft inner repulsion to prevent pile-up at exact cursor point
              const innerRepel = (1 - dist / MOUSE_INNER_REPEL) * 0.08;
              p.vx -= (dx / dist) * innerRepel;
              p.vy -= (dy / dist) * innerRepel;
            }

            // Direct laser link from particle to mouse cursor
            const mouseAlpha = (1 - dist / MOUSE_RADIUS) * 0.35;
            ctx.strokeStyle = `rgba(6, 182, 212, ${mouseAlpha})`;
            ctx.lineWidth = 0.9;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(mx, my);
            ctx.stroke();
          } else {
            // Outside mouse radius: smoothly restore to natural wandering velocity (scatter back)
            p.vx += (p.baseVx - p.vx) * 0.035;
            p.vy += (p.baseVy - p.vy) * 0.035;
          }
        } else {
          // No mouse on screen: smoothly scatter back to natural wandering drift
          p.vx += (p.baseVx - p.vx) * 0.035;
          p.vy += (p.baseVy - p.vy) * 0.035;
        }

        // Speed limit
        const currentSpeed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        if (currentSpeed > MAX_SPEED) {
          p.vx = (p.vx / currentSpeed) * MAX_SPEED;
          p.vy = (p.vy / currentSpeed) * MAX_SPEED;
        }

        p.x += p.vx;
        p.y += p.vy;

        // Wrap edges smoothly
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;
        if (p.y < -10) p.y = h + 10;
        if (p.y > h + 10) p.y = -10;

        // Draw particle dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.opacity;
        ctx.fill();
      }

      // 3. Draw constellation links between nearby particles
      ctx.globalAlpha = 1;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const a = particles[i];
          const b = particles[j];
          const ddx = a.x - b.x;
          const ddy = a.y - b.y;
          const d = Math.sqrt(ddx * ddx + ddy * ddy);
          if (d < CONNECT_DIST) {
            const alpha = (1 - d / CONNECT_DIST) * 0.22;
            ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
            ctx.lineWidth = 0.7;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // Mouse ambient aura
      if (hasMouse) {
        const gradient = ctx.createRadialGradient(mx, my, 0, mx, my, MOUSE_RADIUS);
        gradient.addColorStop(0, 'rgba(6, 182, 212, 0.09)');
        gradient.addColorStop(0.5, 'rgba(99, 102, 241, 0.04)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(mx - MOUSE_RADIUS, my - MOUSE_RADIUS, MOUSE_RADIUS * 2, MOUSE_RADIUS * 2);
      }

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener('resize', setSize);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [createParticle]);

  return (
    <canvas
      ref={canvasRef}
      className="particle-field-canvas"
      aria-hidden="true"
    />
  );
}
