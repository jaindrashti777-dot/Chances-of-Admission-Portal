'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  History, 
  LineChart, 
  Lightbulb, 
  GraduationCap, 
  Settings, 
  Info,
  Moon,
  Sun
} from 'lucide-react';
import styles from './Sidebar.module.css';

interface SidebarProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const NAV_ITEMS = [
  { label: 'Dashboard', href: '/dashboard', icon: <LayoutDashboard size={20} /> },
  { label: 'Prediction History', href: '/dashboard/history', icon: <History size={20} /> },
  { label: 'Compare Profiles', href: '/dashboard/compare', icon: <LineChart size={20} /> },
  { label: 'Insights', href: '/dashboard/insights', icon: <Lightbulb size={20} /> },
  { label: 'Colleges Explorer', href: '/dashboard/explorer', icon: <GraduationCap size={20} /> },
];

const BOTTOM_ITEMS = [
  { label: 'About the Model', href: '/dashboard/about', icon: <Info size={20} /> },
  { label: 'Settings', href: '/dashboard/settings', icon: <Settings size={20} /> },
];

export function Sidebar({ isDarkMode, toggleDarkMode }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logoContainer}>
        <div className={styles.logoIcon}>
          <GraduationCap size={24} color="white" />
        </div>
        <span className={styles.logoText}>AdmissionAI</span>
      </div>

      <nav className={styles.nav}>
        <div className={styles.navGroup}>
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link 
                key={item.href} 
                href={item.href}
                className={`${styles.navItem} ${isActive ? styles.active : ''}`}
              >
                <span className={styles.icon}>{item.icon}</span>
                <span className={styles.label}>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>

      <div className={styles.bottomSection}>
        <div className={styles.navGroup}>
          {BOTTOM_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link 
                key={item.href} 
                href={item.href}
                className={`${styles.navItem} ${isActive ? styles.active : ''}`}
              >
                <span className={styles.icon}>{item.icon}</span>
                <span className={styles.label}>{item.label}</span>
              </Link>
            );
          })}
        </div>

        <div className={styles.themeToggle} onClick={toggleDarkMode}>
          <span className={styles.icon}>
            {isDarkMode ? <Moon size={20} /> : <Sun size={20} />}
          </span>
          <span className={styles.label}>{isDarkMode ? 'Dark Mode' : 'Light Mode'}</span>
          <div className={`${styles.switch} ${isDarkMode ? styles.switchActive : ''}`}>
            <div className={styles.switchHandle} />
          </div>
        </div>
      </div>
    </aside>
  );
}
