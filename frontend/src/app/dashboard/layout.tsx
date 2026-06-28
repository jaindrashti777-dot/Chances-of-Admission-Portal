'use client';

import React, { useState, useEffect } from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopBar } from '@/components/layout/TopBar';
import { ChatWidget } from '@/components/ui/ChatWidget';
import styles from './DashboardLayout.module.css';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check initial preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setIsDarkMode(true);
    }
  }, []);

  useEffect(() => {
    // Note: In a real app we'd save this to localStorage and apply a class to the <html> tag.
    // For now we rely on CSS media queries but this state powers the toggle UI.
    if (isDarkMode) {
      document.documentElement.style.colorScheme = 'dark';
      // Ideally we add a .dark class to body here if we weren't relying purely on media queries.
    } else {
      document.documentElement.style.colorScheme = 'light';
    }
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  return (
    <div className={styles.container}>
      <Sidebar isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />
      <div className={styles.mainWrapper}>
        <TopBar />
        <main className={styles.mainContent}>
          {children}
        </main>
      </div>
      <ChatWidget />
    </div>
  );
}
