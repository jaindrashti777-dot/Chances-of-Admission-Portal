'use client';

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import styles from './FeatureBar.module.css';

interface FeatureBarProps {
  data: { feature: string; contribution: number }[];
}

export function FeatureBar({ data }: FeatureBarProps) {
  // Sort data by absolute contribution magnitude
  const sortedData = [...data].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return (
    <div className={styles.container}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          layout="vertical"
          data={sortedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <XAxis type="number" hide />
          <YAxis 
            dataKey="feature" 
            type="category" 
            axisLine={false} 
            tickLine={false} 
            width={120}
            tick={{ fontSize: 12, fill: 'var(--text-secondary)' }}
          />
          <Tooltip 
            cursor={{ fill: 'transparent' }}
            contentStyle={{ 
              backgroundColor: 'var(--glass-bg)', 
              borderRadius: '8px', 
              border: '1px solid var(--border-color)',
              backdropFilter: 'blur(10px)'
            }}
          />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={20}>
            {sortedData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={entry.contribution > 0 ? 'var(--status-emerald)' : 'var(--status-red)'} 
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
