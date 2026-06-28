'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowRight, GraduationCap } from 'lucide-react';
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
          Admission Predictor
        </div>
        <Link href="/dashboard">
          <Button variant="primary">Open Dashboard</Button>
        </Link>
      </nav>

      <main className={styles.main}>
        <motion.div
          className={styles.badge}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Portfolio Project | End-to-End ML System
        </motion.div>

        <motion.h1
          className={styles.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          College Admission
          <br />
          <span className={styles.highlight}>Prediction System</span>
        </motion.h1>

        <motion.p
          className={styles.description}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          A full-stack machine learning application demonstrating data pipelines,
          model training, FastAPI inference, validation, SHAP-ready explanations,
          and a typed Next.js dashboard. The model is trained on synthetic data and
          is not intended for real admissions decisions.
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
              View Methodology
            </Button>
          </Link>
        </motion.div>

        <motion.div
          className={styles.highlights}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          <div className={styles.highlightItem}>
            <span className={styles.highlightValue}>16</span>
            <span className={styles.highlightLabel}>Input Features</span>
          </div>
          <div className={styles.highlightDivider} />
          <div className={styles.highlightItem}>
            <span className={styles.highlightValue}>7</span>
            <span className={styles.highlightLabel}>Models Compared</span>
          </div>
          <div className={styles.highlightDivider} />
          <div className={styles.highlightItem}>
            <span className={styles.highlightValue}>Pydantic</span>
            <span className={styles.highlightLabel}>Validation</span>
          </div>
          <div className={styles.highlightDivider} />
          <div className={styles.highlightItem}>
            <span className={styles.highlightValue}>FastAPI</span>
            <span className={styles.highlightLabel}>Inference API</span>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
