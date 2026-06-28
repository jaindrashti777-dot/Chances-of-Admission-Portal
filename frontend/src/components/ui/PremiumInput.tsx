import React from 'react';
import styles from './PremiumInput.module.css';

interface PremiumInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  icon?: React.ReactNode;
  error?: string;
}

export function PremiumInput({ 
  label, 
  icon, 
  error,
  className = '', 
  ...props 
}: PremiumInputProps) {
  
  return (
    <div className={`${styles.container} ${className}`}>
      <label className={styles.label}>{label}</label>
      <div className={styles.inputWrapper}>
        {icon && <div className={styles.icon}>{icon}</div>}
        <input 
          className={`${styles.input} ${icon ? styles.hasIcon : ''} ${error ? styles.hasError : ''}`}
          {...props}
        />
      </div>
      {error && <span className={styles.errorText}>{error}</span>}
    </div>
  );
}
