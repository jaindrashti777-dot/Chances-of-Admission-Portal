import React from 'react';
import { GitBranch } from 'lucide-react';
import styles from './TopBar.module.css';

export function TopBar() {
  return (
    <header className={styles.topbar}>
      <div className={styles.breadcrumb}>
        <span className={styles.breadcrumbText}>Admission Predictor</span>
      </div>

      <div className={styles.actions}>
        <a
          href="https://github.com/jaindrashti777-dot/Chances-of-Admission-Portal"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.githubLink}
          aria-label="View source on GitHub"
        >
          <GitBranch size={20} />
          <span className={styles.githubLabel}>Source</span>
        </a>
      </div>
    </header>
  );
}
