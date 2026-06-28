'use client';

import React, { useState } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { PremiumInput } from '@/components/ui/PremiumInput';
import { PremiumSelect } from '@/components/ui/PremiumSelect';
import { WhatIfSlider } from '@/components/ui/WhatIfSlider';
import { Button } from '@/components/ui/Button';
import { CircularProgress } from '@/components/ui/CircularProgress';
import { FeatureBar } from '@/components/charts/FeatureBar';
import { GraduationCap, Award, BrainCircuit, Activity, Download } from 'lucide-react';
import styles from './DashboardPage.module.css';

interface PredictionResult {
  admission_chance: number;
  confidence_level: string;
  resume_score: number;
  top_factors?: { feature: string; contribution: number }[];
}

const FEATURE_LABELS: Record<string, string> = {
  'num__Tenth_Percentage': '10th Percentage',
  'num__Twelfth_Percentage': '12th Percentage',
  'num__JEE_Percentile': 'JEE Percentile',
  'num__CUET_Score': 'CUET Score',
  'num__CGPA': 'CGPA',
  'num__Backlogs': 'Active Backlogs',
  'num__Family_Income': 'Family Income',
  'num__Gap_Year': 'Gap Years',
  'num__Research_Paper': 'Research Papers',
  'num__Internship': 'Internships',
  'cat__Category_General': 'General Category',
  'cat__Gender_Male': 'Male',
  'cat__Gender_Female': 'Female',
};

function formatFeatureName(rawName: string) {
  return FEATURE_LABELS[rawName] || rawName.replace(/^(num__|cat__)/, '').replace(/_/g, ' ');
}

export default function Dashboard() {
  const [formData, setFormData] = useState({
    Tenth_Percentage: 90,
    Twelfth_Percentage: 88,
    JEE_Percentile: 95,
    CUET_Score: 700,
    Category: 'General',
    State: 'Maharashtra',
    Family_Income: 10,
    Gender: 'Male',
    Gap_Year: 0,
    CGPA: 8.5,
    Backlogs: 0,
    Extracurricular: 1,
    Research_Paper: 0,
    Internship: 1,
    Desired_Branch: 'CSE',
    College_Tier: 'Tier_1',
  });

  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    // Auto convert numbers
    const numValue = !isNaN(Number(value)) && value !== '' ? Number(value) : value;
    setFormData(prev => ({ ...prev, [name]: numValue }));
  };

  const handleSliderChange = (name: string, value: number) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Prediction failed. Please check your inputs.');
      }
      
      const data = await res.json();
      
      // Format SHAP feature names
      if (data.top_factors) {
        data.top_factors = data.top_factors.map((f: any) => ({
          ...f,
          feature: formatFeatureName(f.feature)
        }));
      }
      
      setPrediction(data);
    } catch (error: any) {
      console.error("Prediction failed:", error);
      setErrorMsg(error.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Chance of Admission Predictor</h1>
          <p className={styles.subtitle}>Predict your probability of getting admission into Indian engineering colleges.</p>
        </div>
        {prediction && (
          <Button variant="outline" icon={<Download size={18} />} onClick={() => window.print()}>
            Download Report
          </Button>
        )}
      </header>

      <div className={styles.grid}>
        {/* LEFT COLUMN: Profile Form */}
        <div className={styles.formCol}>
          <GlassCard padding="lg">
            <form onSubmit={handlePredict}>
              <div className={styles.sectionHeader}>
                <div className={styles.sectionIcon}><GraduationCap size={20} /></div>
                <h3>Academic Performance</h3>
              </div>
              <div className={styles.inputGrid}>
                <PremiumInput label="10th Percentage (%)" name="Tenth_Percentage" type="number" step="0.1" value={formData.Tenth_Percentage} onChange={handleChange} required />
                <PremiumInput label="12th Percentage (%)" name="Twelfth_Percentage" type="number" step="0.1" value={formData.Twelfth_Percentage} onChange={handleChange} required />
                <WhatIfSlider label="Current CGPA (out of 10)" min={0} max={10} step={0.1} value={formData.CGPA} onChange={(v) => handleSliderChange('CGPA', v)} />
                <PremiumInput label="Active Backlogs" name="Backlogs" type="number" value={formData.Backlogs} onChange={handleChange} required />
              </div>

              <div className={styles.sectionHeader}>
                <div className={styles.sectionIcon}><BrainCircuit size={20} /></div>
                <h3>Entrance Exams</h3>
              </div>
              <div className={styles.inputGrid}>
                <PremiumInput label="JEE Percentile" name="JEE_Percentile" type="number" step="0.1" value={formData.JEE_Percentile} onChange={handleChange} required />
                <PremiumInput label="CUET Score (out of 800)" name="CUET_Score" type="number" value={formData.CUET_Score} onChange={handleChange} required />
              </div>

              <div className={styles.sectionHeader}>
                <div className={styles.sectionIcon}><Award size={20} /></div>
                <h3>Achievements</h3>
              </div>
              <div className={styles.inputGrid}>
                <PremiumSelect 
                  label="Extracurriculars" name="Extracurricular" value={formData.Extracurricular} onChange={handleChange}
                  options={[{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }]} 
                />
                <PremiumInput label="Research Papers" name="Research_Paper" type="number" value={formData.Research_Paper} onChange={handleChange} required />
                <PremiumInput label="Internships" name="Internship" type="number" value={formData.Internship} onChange={handleChange} required />
              </div>

              <div className={styles.sectionHeader}>
                <div className={styles.sectionIcon}><Activity size={20} /></div>
                <h3>Target & Personal</h3>
              </div>
              <div className={styles.inputGrid}>
                <PremiumSelect 
                  label="Desired Branch" name="Desired_Branch" value={formData.Desired_Branch} onChange={handleChange}
                  options={[
                    { value: 'CSE', label: 'Computer Science (CSE)' },
                    { value: 'ECE', label: 'Electronics (ECE)' },
                    { value: 'ME', label: 'Mechanical (ME)' },
                    { value: 'CE', label: 'Civil (CE)' },
                    { value: 'EE', label: 'Electrical (EE)' },
                    { value: 'IT', label: 'Information Tech (IT)' },
                    { value: 'Chemical', label: 'Chemical' },
                    { value: 'Biotech', label: 'Biotech' },
                  ]} 
                />
                <PremiumSelect 
                  label="College Tier" name="College_Tier" value={formData.College_Tier} onChange={handleChange}
                  options={[
                    { value: 'Tier_1', label: 'Tier 1 (IITs/NITs/BITS)' },
                    { value: 'Tier_2', label: 'Tier 2 (Top State/Private)' },
                    { value: 'Tier_3', label: 'Tier 3 (Local/Private)' },
                  ]} 
                />
                <PremiumSelect 
                  label="Category" name="Category" value={formData.Category} onChange={handleChange}
                  options={[
                    { value: 'General', label: 'General' },
                    { value: 'OBC', label: 'OBC' },
                    { value: 'SC', label: 'SC' },
                    { value: 'ST', label: 'ST' },
                    { value: 'EWS', label: 'EWS' },
                  ]} 
                />
                <PremiumSelect 
                  label="Gender" name="Gender" value={formData.Gender} onChange={handleChange}
                  options={[{ value: 'Male', label: 'Male' }, { value: 'Female', label: 'Female' }]} 
                />
                <PremiumInput label="Family Income (LPA)" name="Family_Income" type="number" value={formData.Family_Income} onChange={handleChange} required />
                <PremiumSelect 
                  label="Home State" name="State" value={formData.State} onChange={handleChange}
                  options={[
                    { value: 'Maharashtra', label: 'Maharashtra' },
                    { value: 'Delhi', label: 'Delhi' },
                    { value: 'Karnataka', label: 'Karnataka' },
                    { value: 'Tamil_Nadu', label: 'Tamil Nadu' },
                    { value: 'Uttar_Pradesh', label: 'Uttar Pradesh' },
                    { value: 'West_Bengal', label: 'West Bengal' },
                    { value: 'Rajasthan', label: 'Rajasthan' },
                    { value: 'Gujarat', label: 'Gujarat' },
                    { value: 'Telangana', label: 'Telangana' },
                    { value: 'Kerala', label: 'Kerala' },
                  ]} 
                />
                <PremiumInput label="Gap Years" name="Gap_Year" type="number" value={formData.Gap_Year} onChange={handleChange} required />
              </div>

              <div className={styles.submitArea}>
                <Button type="submit" size="lg" className={styles.submitBtn} disabled={loading}>
                  {loading ? 'Analyzing...' : 'Predict Admission Chance'}
                </Button>
              </div>
            </form>
          </GlassCard>
        </div>

        {/* RIGHT COLUMN: Results */}
        <div className={styles.resultsCol}>
          {errorMsg ? (
            <GlassCard className={styles.emptyState}>
              <Activity size={48} className={styles.emptyIcon} style={{ color: 'var(--error-color)' }} />
              <h3 style={{ color: 'var(--error-color)' }}>Prediction Error</h3>
              <p>{errorMsg}</p>
            </GlassCard>
          ) : !prediction ? (
            <GlassCard className={styles.emptyState}>
              <BrainCircuit size={48} className={styles.emptyIcon} />
              <h3>Awaiting Profile</h3>
              <p>Fill in your details and click predict to see your admission probability and insights.</p>
            </GlassCard>
          ) : (
            <>
              {/* Main Prediction Card */}
              <GlassCard className={styles.mainResultCard}>
                <div className={styles.gaugeHeader}>
                  <h3>Predicted Admission Chance</h3>
                  <span className={`${styles.badge} ${styles['badge' + prediction.confidence_level]}`}>
                    {prediction.confidence_level} Confidence
                  </span>
                </div>
                
                <div className={styles.gaugeWrapper}>
                  <CircularProgress value={prediction.admission_chance * 100} size={240} />
                </div>

                <div className={styles.scoreRow}>
                  <div className={styles.scoreBox}>
                    <span className={styles.scoreLabel}>Resume Readiness</span>
                    <span className={styles.scoreValue}>{prediction.resume_score}/100</span>
                  </div>
                  <div className={styles.scoreBox}>
                    <span className={styles.scoreLabel}>Recommended Tier</span>
                    <span className={styles.scoreValue}>{formData.College_Tier.replace('_', ' ')}</span>
                  </div>
                </div>
              </GlassCard>

              {/* SHAP Factors */}
              {prediction.top_factors && prediction.top_factors.length > 0 && (
                <GlassCard className={styles.shapCard}>
                  <h3>What&apos;s Influencing Your Score?</h3>
                  <p className={styles.shapDesc}>
                    Based on SHAP explainability, here are the top factors affecting your probability.
                  </p>
                  <FeatureBar data={prediction.top_factors} />
                </GlassCard>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
