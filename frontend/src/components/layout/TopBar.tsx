import React from 'react';
import { Search, Bell, User } from 'lucide-react';
import styles from './TopBar.module.css';

export function TopBar() {
  return (
    <header className={styles.topbar}>
      <div className={styles.searchContainer}>
        <Search size={18} className={styles.searchIcon} />
        <input 
          type="text" 
          placeholder="Search colleges, branches, or features..." 
          className={styles.searchInput}
          onClick={() => alert('Search functionality coming soon!')}
        />
      </div>

      <div className={styles.actions}>
        <button className={styles.iconButton} onClick={() => alert('Notifications coming soon!')}>
          <Bell size={20} />
          <span className={styles.notificationBadge} />
        </button>
        
        <div className={styles.profileDropdown} onClick={() => alert('Profile settings coming soon!')}>
          <div className={styles.avatar}>
            <User size={18} />
          </div>
          <div className={styles.profileInfo}>
            <span className={styles.profileName}>Student Profile</span>
            <span className={styles.profileRole}>Candidate</span>
          </div>
        </div>
      </div>
    </header>
  );
}
