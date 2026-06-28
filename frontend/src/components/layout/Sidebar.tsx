'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTheme } from 'next-themes';
import { 
  LayoutDashboard, 
  GraduationCap,
  Info,
  Moon,
  Sun
} from 'lucide-react';
import styles from './Sidebar.module.css';

const NAV_ITEMS = [
  { label: 'Predictor', href: '/dashboard', icon: <LayoutDashboard size={20} /> },
  { label: 'About the Model', href: '/dashboard/about', icon: <Info size={20} /> },
];

export function Sidebar() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logoContainer}>
        <div className={styles.logoIcon}>
          <GraduationCap size={24} color="white" />
        </div>
        <span className={styles.logoText}>Admission Predictor</span>
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
        <div 
          className={styles.themeToggle} 
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          <span className={styles.icon}>
            {mounted && theme === 'dark' ? <Moon size={20} /> : <Sun size={20} />}
          </span>
          <span className={styles.label}>{mounted && theme === 'dark' ? 'Dark Mode' : 'Light Mode'}</span>
          <div className={`${styles.switch} ${mounted && theme === 'dark' ? styles.switchActive : ''}`}>
            <div className={styles.switchHandle} />
          </div>
        </div>
      </div>
    </aside>
  );
}
