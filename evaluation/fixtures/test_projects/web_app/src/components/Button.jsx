import React from 'react';

export const Button = ({ onClick, children, variant = 'primary' }) => {
  return (
    <button 
      className={`btn btn-${variant}`}
      onClick={onClick}
      aria-label={children}
    >
      {children}
    </button>
  );
};
