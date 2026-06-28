'use client';

import React from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopBar } from '@/components/layout/TopBar';
import { ChatWidget } from '@/components/ui/ChatWidget';
import styles from './DashboardLayout.module.css';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className={styles.container}>
      <Sidebar />
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
