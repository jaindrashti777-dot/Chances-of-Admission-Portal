import { GlassCard } from '@/components/ui/GlassCard';
import { Hammer } from 'lucide-react';

export default function ComingSoonPage() {
  return (
    <div style={{ padding: '2rem', display: 'flex', justifyContent: 'center', width: '100%' }}>
      <GlassCard padding="lg" style={{ textAlign: 'center', maxWidth: '500px', width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
          <Hammer size={48} style={{ color: 'var(--primary-indigo)' }} />
        </div>
        <h2>Under Construction</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
          This feature is currently being developed and will be available in a future update!
        </p>
      </GlassCard>
    </div>
  );
}