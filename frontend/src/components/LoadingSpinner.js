import { Loader2 } from 'lucide-react';
import React from 'react';

const LoadingSpinner = ({ className = 'size-5' }) => (
  <Loader2 aria-hidden="true" className={`animate-spin ${className}`} />
);

export default LoadingSpinner;
