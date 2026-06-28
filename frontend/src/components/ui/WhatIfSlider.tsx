import React, { useState, useEffect } from 'react';
import styles from './WhatIfSlider.module.css';

interface WhatIfSliderProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (val: number) => void;
  formatValue?: (val: number) => string;
}

export function WhatIfSlider({ 
  label, 
  min, 
  max, 
  step = 1, 
  value, 
  onChange,
  formatValue = (v) => v.toString()
}: WhatIfSliderProps) {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLocalValue(value);
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setLocalValue(val);
    onChange(val); // In a real scenario, this might be debounced
  };

  const percentage = ((localValue - min) / (max - min)) * 100;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <label className={styles.label}>{label}</label>
        <span className={styles.value}>{formatValue(localValue)}</span>
      </div>
      <input 
        type="range" 
        min={min} 
        max={max} 
        step={step}
        value={localValue} 
        onChange={handleChange}
        className={styles.slider}
        style={{ '--progress': `${percentage}%` } as React.CSSProperties}
      />
      <div className={styles.ticks}>
        <span>{formatValue(min)}</span>
        <span>{formatValue(max)}</span>
      </div>
    </div>
  );
}
