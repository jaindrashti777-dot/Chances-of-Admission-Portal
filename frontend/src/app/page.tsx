'use client';

import React from 'react';
import Link from 'next/link';
import { GraduationCap, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { motion } from 'framer-motion';
import styles from './LandingPage.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <nav className={styles.nav}>
        <div className={styles.logo}>
          <div className={styles.logoIcon}>
            <GraduationCap size={24} color="white" />
          </div>
          AdmissionAI
        </div>
        <div>
          <Link href="/dashboard">
            <Button variant="ghost">Sign In</Button>
          </Link>
          <Link href="/dashboard">
            <Button variant="primary">Get Started</Button>
          </Link>
        </div>
      </nav>

      <main className={styles.main}>
        <motion.div 
          className={styles.badge}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          v2.0 • Now powered by Advanced Machine Learning
        </motion.div>

        <motion.h1 
          className={styles.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          Predict Your Future.<br/>
          <span className={styles.highlight}>Guarantee Your Admission.</span>
        </motion.h1>

        <motion.p 
          className={styles.description}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Stop guessing your chances. Use our AI model trained on over 500,000 student records to predict your probability of getting into top Indian engineering colleges with 96% accuracy.
        </motion.p>

        <motion.div 
          className={styles.actions}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Link href="/dashboard">
            <Button variant="primary" size="lg" icon={<ArrowRight size={18} />}>
              Try the Predictor
            </Button>
          </Link>
          <Link href="/dashboard/about">
            <Button variant="outline" size="lg">
              View the Methodology
            </Button>
          </Link>
        </motion.div>

        <motion.div 
          className={styles.heroImage}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          <div className={styles.mockupHeader}>
            <div className={styles.dot} />
            <div className={styles.dot} />
            <div className={styles.dot} />
          </div>
          <div className={styles.mockupContent}>
            <div className={styles.mockupGraphic} />
            {/* We could place an actual screenshot image here later using next/image */}
          </div>
        </motion.div>
      </main>
    </div>
  );
}
