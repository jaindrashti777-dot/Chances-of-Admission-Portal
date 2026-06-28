import React from 'react';
import styles from './PremiumInput.module.css';
import { ChevronDown } from 'lucide-react';

interface PremiumSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
  icon?: React.ReactNode;
  options: { value: string | number; label: string }[];
  error?: string;
}

export function PremiumSelect({ 
  label, 
  icon, 
  options,
  error,
  className = '', 
  ...props 
}: PremiumSelectProps) {
  
  return (
    <div className={`${styles.container} ${className}`}>
      <label className={styles.label}>{label}</label>
      <div className={styles.inputWrapper}>
        {icon && <div className={styles.icon}>{icon}</div>}
        <select 
          className={`${styles.input} ${icon ? styles.hasIcon : ''} ${error ? styles.hasError : ''}`}
          style={{ appearance: 'none' }}
          {...props}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <div className={styles.icon} style={{ left: 'auto', right: '1rem', pointerEvents: 'none' }}>
          <ChevronDown size={16} />
        </div>
      </div>
      {error && <span className={styles.errorText}>{error}</span>}
    </div>
  );
}
