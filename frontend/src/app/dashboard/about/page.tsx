import { GlassCard } from '@/components/ui/GlassCard';
import styles from './AboutPage.module.css';

export default function AboutPage() {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>About the Model</h1>
        <p className={styles.subtitle}>
          Methodology, architecture, and limitations for the synthetic admission prediction system.
        </p>
      </header>

      <div className={styles.grid}>
        <GlassCard padding="lg">
          <h2 className={styles.sectionTitle}>Overview</h2>
          <p className={styles.text}>
            This project demonstrates an end-to-end ML application: synthetic data generation,
            preprocessing, model training, FastAPI inference, Pydantic validation, SHAP-ready
            feature contributions, and a deployed Next.js dashboard.
          </p>
        </GlassCard>

        <GlassCard padding="lg">
          <h2 className={styles.sectionTitle}>Dataset</h2>
          <ul className={styles.list}>
            <li>
              <strong>Source:</strong> Synthetic records generated from a weighted formula
            </li>
            <li>
              <strong>Size:</strong> 5,000 records by default
            </li>
            <li>
              <strong>Features:</strong> 16 academic, demographic, and profile fields
            </li>
            <li>
              <strong>Target:</strong> Admission_Chance, a continuous value from 0 to 1
            </li>
          </ul>
          <p className={styles.text}>
            High offline scores are expected because the target is generated synthetically.
            They should not be interpreted as real-world admissions accuracy.
          </p>
        </GlassCard>

        <GlassCard padding="lg">
          <h2 className={styles.sectionTitle}>Architecture</h2>
          <ul className={styles.list}>
            <li>
              <strong>Backend:</strong> FastAPI service with structured validation errors
            </li>
            <li>
              <strong>Frontend:</strong> Next.js App Router with typed form state
            </li>
            <li>
              <strong>ML Pipeline:</strong> scikit-learn pipeline serialized with joblib
            </li>
            <li>
              <strong>Deployment:</strong> Vercel frontend and Render backend
            </li>
          </ul>
        </GlassCard>

        <GlassCard padding="lg">
          <h2 className={styles.sectionTitle}>Limitations</h2>
          <ul className={styles.list}>
            <li>The dataset is synthetic; the model learns a generated formula.</li>
            <li>The app is not suitable for real admissions decisions or counseling.</li>
            <li>The resume readiness score is a transparent heuristic, not a trained output.</li>
            <li>The advisor widget is rule-based and does not call an LLM.</li>
          </ul>
        </GlassCard>
      </div>
    </div>
  );
}
