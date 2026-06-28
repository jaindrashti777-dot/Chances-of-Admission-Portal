import React from 'react';
import styles from './GlassCard.module.css';

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export function GlassCard({ 
  children, 
  className = '', 
  padding = 'md',
  ...props 
}: GlassCardProps) {
  
  return (
    <div 
      className={`glass ${styles.card} ${styles[`padding-${padding}`]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
