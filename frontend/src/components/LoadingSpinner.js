import { Circle } from 'lucide-react';
import React from 'react';

const LoadingSpinner = ({ message = "Loading...", subMessage = "Preparing AI models" }) => {
  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="relative">
        <Circle className="h-16 w-16 text-nba-orange animate-spin" />
        <div className="absolute inset-0 h-16 w-16 border-4 border-transparent border-t-nba-orange rounded-full animate-spin"></div>
      </div>
      <div className="text-center">
        <p className="text-white text-lg font-medium">{message}</p>
        <p className="text-gray-300 text-sm">{subMessage}</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
