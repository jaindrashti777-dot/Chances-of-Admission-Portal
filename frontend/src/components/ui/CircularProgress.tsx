'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import styles from './CircularProgress.module.css';

interface CircularProgressProps {
  value: number; // 0 to 100
  size?: number;
  strokeWidth?: number;
  label?: string;
}

export function CircularProgress({ 
  value, 
  size = 200, 
  strokeWidth = 15,
  label = "Admission Chance" 
}: CircularProgressProps) {
  const [currentValue, setCurrentValue] = useState(0);

  useEffect(() => {
    // Small delay for initial animation
    const timer = setTimeout(() => setCurrentValue(value), 100);
    return () => clearTimeout(timer);
  }, [value]);

  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  // Offset for semi-circle (if we want a full circle, just use normal offset. Here let's do a full circle gauge)
  const offset = circumference - (currentValue / 100) * circumference;

  let color = 'var(--status-red)';
  if (value >= 75) color = 'var(--status-emerald)';
  else if (value >= 40) color = 'var(--status-orange)';

  return (
    <div className={styles.container} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className={styles.bgCircle}
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress Circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className={styles.progressCircle}
          strokeWidth={strokeWidth}
          stroke={color}
          fill="none"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className={styles.content}>
        <motion.div 
          className={styles.percentage}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.5 }}
        >
          {Math.round(currentValue)}%
        </motion.div>
        {label && <div className={styles.label}>{label}</div>}
      </div>
    </div>
  );
}
