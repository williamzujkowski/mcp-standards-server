export interface Standard {
  id: string;
  title: string;
  description: string;
  category: string;
  subcategory: string;
  tags: string[];
  priority: string;
  version: string;
  examples: Example[];
  rules: Record<string, any>;
  created_at?: string;
  updated_at?: string;
  metadata: Record<string, any>;
}

export interface Example {
  title?: string;
  language?: string;
  code: string;
  description?: string;
}

export interface Category {
  name: string;
  description: string;
  count: number;
}

export interface SearchResult {
  standard: Standard;
  score: number;
  highlights: Record<string, string[]>;
}

export interface ProjectContext {
  languages: string[];
  frameworks: string[];
  project_type: string;
  deployment_target: string;
  team_size: string;
  compliance_requirements: string[];
  existing_tools: string[];
  performance_requirements: Record<string, any>;
  security_requirements: Record<string, any>;
  scalability_requirements: Record<string, any>;
}

export interface Recommendation {
  standard: Standard;
  relevance_score: number;
  confidence: number;
  reasoning: string;
  implementation_notes: string;
}