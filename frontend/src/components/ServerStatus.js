import { AlertCircle } from 'lucide-react';
import React from 'react';
import LoadingSpinner from './LoadingSpinner';

// Compact pill for the header; only speaks up when the server isn't ready.
const ServerStatus = ({ state, onRetry }) => {
  if (state === 'waking') {
    return (
      <div role="status" className="flex min-h-[36px] items-center gap-2 rounded-full bg-fill px-3 text-sm text-label-2">
        <LoadingSpinner className="size-4 shrink-0" />
        Server starting, up to a minute
      </div>
    );
  }

  if (state === 'offline') {
    return (
      <div role="alert" className="flex items-center gap-1 rounded-full bg-danger/10 pl-3 text-sm text-label">
        <AlertCircle aria-hidden="true" className="size-4 shrink-0 text-danger" />
        <span className="ml-1">Server unreachable</span>
        <button
          type="button"
          onClick={onRetry}
          className="min-h-[44px] rounded-full px-3 font-semibold text-tint transition-colors hover:bg-tint/10"
        >
          Try again
        </button>
      </div>
    );
  }

  return null;
};

export default ServerStatus;
