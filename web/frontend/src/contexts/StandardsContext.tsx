import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { StandardsService } from '../services/standardsService';
import { Standard, Category } from '../types';

interface StandardsContextType {
  standards: Record<string, Standard[]>;
  categories: Category[];
  loading: boolean;
  error: string | null;
  refreshStandards: () => Promise<void>;
  getStandardById: (id: string) => Standard | undefined;
}

const StandardsContext = createContext<StandardsContextType | null>(null);

export const useStandards = () => {
  const context = useContext(StandardsContext);
  if (!context) {
    throw new Error('useStandards must be used within a StandardsProvider');
  }
  return context;
};

interface StandardsProviderProps {
  children: React.ReactNode;
}

export const StandardsProvider: React.FC<StandardsProviderProps> = ({ children }) => {
  const [standards, setStandards] = useState<Record<string, Standard[]>>({});
  const [categories, setCategories] = useState<Category[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshStandards = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [standardsData, categoriesData] = await Promise.all([
        StandardsService.getAllStandards(),
        StandardsService.getCategories()
      ]);
      setStandards(standardsData.standards);
      setCategories(categoriesData.categories);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load standards');
    } finally {
      setLoading(false);
    }
  }, []);

  const getStandardById = useCallback((id: string) => {
    for (const categoryStandards of Object.values(standards)) {
      const standard = categoryStandards.find(s => s.id === id);
      if (standard) return standard;
    }
    return undefined;
  }, [standards]);

  useEffect(() => {
    refreshStandards();
  }, [refreshStandards]);

  return (
    <StandardsContext.Provider
      value={{
        standards,
        categories,
        loading,
        error,
        refreshStandards,
        getStandardById
      }}
    >
      {children}
    </StandardsContext.Provider>
  );
};