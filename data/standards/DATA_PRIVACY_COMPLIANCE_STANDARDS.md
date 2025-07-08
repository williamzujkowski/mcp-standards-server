# Data Privacy and Compliance Standards

**Version:** v1.0.0  
**Domain:** privacy  
**Type:** Technical  
**Risk Level:** CRITICAL  
**Maturity Level:** Production  
**Author:** MCP Standards Team  
**Created:** 2025-07-08T00:00:00.000000  
**Last Updated:** 2025-07-08T00:00:00.000000  

## Purpose

Comprehensive standards for data privacy and regulatory compliance, including GDPR/CCPA implementation, PII handling, and privacy by design principles

This data privacy standard defines the requirements, guidelines, and best practices for data privacy and compliance standards. It provides comprehensive guidance for personal data protection, regulatory compliance, and privacy engineering while ensuring user rights, data security, and organizational accountability across all systems.

**Privacy Focus Areas:**
- **Regulatory Compliance**: GDPR, CCPA, and global privacy laws
- **Data Classification**: Sensitive data identification and handling
- **PII Protection**: Encryption and anonymization techniques
- **Consent Management**: User permissions and preferences
- **Data Governance**: Retention, deletion, and portability
- **Privacy Engineering**: Privacy by design implementation

## Scope

This privacy standard applies to:
- Personal data collection and processing
- Data storage and transmission systems
- Third-party data sharing
- Cross-border data transfers
- User consent mechanisms
- Data breach procedures
- Privacy impact assessments
- Audit and compliance reporting

## Implementation

### Privacy Compliance Requirements

**NIST Controls:** NIST-AC-1, AC-3, AC-7, AU-1, AU-2, AU-3, AU-12, IA-1, IA-2, IA-8, IP-1, IP-2, IP-3, IP-4, MP-1, MP-6, PE-1, PL-1, PL-4, PL-8, RA-3, RA-5, SA-8, SC-1, SC-7, SC-8, SC-13, SC-28, SI-12

**Privacy Standards:** ISO/IEC 27701, ISO/IEC 29134, NIST Privacy Framework
**Regulatory Standards:** GDPR, CCPA/CPRA, LGPD, PIPEDA, APPI
**Security Standards:** AES-256, TLS 1.3, FIPS 140-2

### Data Classification System

#### Classification Levels

```yaml
data_classification:
  public:
    description: "Information intended for public disclosure"
    controls:
      - no_encryption_required
      - standard_access_logging
    examples:
      - marketing_content
      - public_documentation
      
  internal:
    description: "Internal business information"
    controls:
      - encryption_at_rest
      - access_control_required
      - audit_logging
    examples:
      - internal_emails
      - business_processes
      
  confidential:
    description: "Sensitive business or personal information"
    controls:
      - encryption_everywhere
      - strict_access_control
      - detailed_audit_trail
      - data_loss_prevention
    examples:
      - customer_pii
      - financial_records
      - health_information
      
  restricted:
    description: "Highly sensitive requiring maximum protection"
    controls:
      - end_to_end_encryption
      - multi_factor_authentication
      - privileged_access_management
      - continuous_monitoring
      - data_residency_controls
    examples:
      - payment_card_data
      - government_ids
      - biometric_data
```

#### PII Identification

```typescript
interface PIIClassification {
  directIdentifiers: {
    // Can identify an individual alone
    fullName: boolean;
    emailAddress: boolean;
    phoneNumber: boolean;
    socialSecurityNumber: boolean;
    passportNumber: boolean;
    driversLicense: boolean;
    biometricData: boolean;
  };
  
  indirectIdentifiers: {
    // Can identify when combined
    dateOfBirth: boolean;
    zipCode: boolean;
    gender: boolean;
    ethnicity: boolean;
    jobTitle: boolean;
    salary: boolean;
  };
  
  sensitiveData: {
    // Special category data (GDPR Article 9)
    healthData: boolean;
    geneticData: boolean;
    sexualOrientation: boolean;
    politicalOpinions: boolean;
    religiousBeliefs: boolean;
    tradeUnionMembership: boolean;
  };
}

class PIIScanner {
  async scanData(data: any): Promise<PIIScanResult> {
    const patterns = {
      email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
      ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
      creditCard: /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g,
      phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g,
      ipAddress: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g
    };
    
    const findings: PIIFinding[] = [];
    
    // Scan for patterns
    for (const [type, pattern] of Object.entries(patterns)) {
      const matches = this.findMatches(data, pattern);
      if (matches.length > 0) {
        findings.push({
          type,
          count: matches.length,
          locations: matches.map(m => m.location),
          confidence: this.calculateConfidence(type, matches)
        });
      }
    }
    
    // ML-based detection for context
    const mlFindings = await this.mlDetection(data);
    findings.push(...mlFindings);
    
    return {
      hasPII: findings.length > 0,
      findings,
      riskLevel: this.calculateRiskLevel(findings),
      recommendations: this.generateRecommendations(findings)
    };
  }
}
```

### GDPR Implementation

#### Data Subject Rights

```typescript
interface GDPRRights {
  access: RightToAccess;
  rectification: RightToRectification;
  erasure: RightToErasure;
  portability: RightToPortability;
  restriction: RightToRestriction;
  objection: RightToObjection;
  automatedDecision: RightToNotBeSubjectToAutomatedDecision;
}

class GDPRComplianceEngine {
  async handleDataSubjectRequest(
    request: DataSubjectRequest
  ): Promise<DataSubjectResponse> {
    // Verify identity
    const identity = await this.verifyIdentity(request.userId, request.verification);
    
    if (!identity.verified) {
      return {
        status: 'identity_verification_failed',
        reason: identity.reason
      };
    }
    
    switch (request.type) {
      case 'access':
        return await this.handleAccessRequest(request);
        
      case 'erasure':
        return await this.handleErasureRequest(request);
        
      case 'portability':
        return await this.handlePortabilityRequest(request);
        
      case 'rectification':
        return await this.handleRectificationRequest(request);
        
      case 'restriction':
        return await this.handleRestrictionRequest(request);
        
      case 'objection':
        return await this.handleObjectionRequest(request);
        
      default:
        throw new Error('Unknown request type');
    }
  }
  
  async handleAccessRequest(request: DataSubjectRequest): Promise<DataSubjectResponse> {
    // Collect all personal data
    const personalData = await this.collectPersonalData(request.userId);
    
    // Include processing information
    const processingInfo = await this.getProcessingInformation(request.userId);
    
    // Generate report
    const report = {
      personalData,
      processingPurposes: processingInfo.purposes,
      dataCategories: processingInfo.categories,
      recipients: processingInfo.recipients,
      retentionPeriods: processingInfo.retention,
      dataSource: processingInfo.source,
      automatedDecisionMaking: processingInfo.automatedDecisions
    };
    
    // Encrypt and sign
    const encryptedReport = await this.encryptReport(report, request.userId);
    
    return {
      status: 'completed',
      data: encryptedReport,
      format: 'encrypted_json',
      validUntil: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days
    };
  }
  
  async handleErasureRequest(request: DataSubjectRequest): Promise<DataSubjectResponse> {
    // Check if erasure can be performed
    const erasureCheck = await this.checkErasureEligibility(request.userId);
    
    if (!erasureCheck.eligible) {
      return {
        status: 'rejected',
        reason: erasureCheck.reason,
        legalBasis: erasureCheck.legalBasis
      };
    }
    
    // Perform erasure
    const erasureResult = await this.performErasure(request.userId, {
      cascade: true,
      notifyThirdParties: true,
      generateCertificate: true
    });
    
    return {
      status: 'completed',
      certificate: erasureResult.certificate,
      erasedData: erasureResult.summary,
      thirdPartyNotifications: erasureResult.notifications
    };
  }
}
```

#### Consent Management

```typescript
interface ConsentManagement {
  collection: ConsentCollection;
  storage: ConsentStorage;
  withdrawal: ConsentWithdrawal;
  audit: ConsentAudit;
}

class ConsentManager {
  async collectConsent(
    userId: string,
    purpose: ConsentPurpose,
    options: ConsentOptions
  ): Promise<ConsentRecord> {
    // Validate consent request
    this.validateConsentRequest(purpose, options);
    
    // Create consent record
    const consent: ConsentRecord = {
      id: generateUUID(),
      userId,
      purpose: purpose.id,
      description: purpose.description,
      lawfulBasis: purpose.lawfulBasis,
      dataCategories: purpose.dataCategories,
      retention: purpose.retentionPeriod,
      thirdParties: purpose.thirdParties,
      given: true,
      timestamp: new Date(),
      ipAddress: this.hashIP(options.ipAddress),
      userAgent: options.userAgent,
      version: purpose.version,
      withdrawalMethod: 'api|ui|email',
      expiryDate: this.calculateExpiry(purpose)
    };
    
    // Store with cryptographic proof
    const signature = await this.signConsent(consent);
    consent.signature = signature;
    
    await this.store(consent);
    
    // Update user preferences
    await this.updateUserPreferences(userId, purpose.id, true);
    
    return consent;
  }
  
  async withdrawConsent(
    userId: string,
    purposeId: string,
    reason?: string
  ): Promise<WithdrawalRecord> {
    // Verify current consent
    const currentConsent = await this.getCurrentConsent(userId, purposeId);
    
    if (!currentConsent) {
      throw new Error('No active consent found');
    }
    
    // Create withdrawal record
    const withdrawal: WithdrawalRecord = {
      id: generateUUID(),
      consentId: currentConsent.id,
      userId,
      purposeId,
      timestamp: new Date(),
      reason,
      processingsStopped: [],
      dataDeleted: false
    };
    
    // Stop related processing
    const stoppedProcessings = await this.stopProcessing(userId, purposeId);
    withdrawal.processingsStopped = stoppedProcessings;
    
    // Check if data deletion required
    if (await this.isDeletionRequired(purposeId)) {
      const deletionResult = await this.scheduleDataDeletion(userId, purposeId);
      withdrawal.dataDeleted = deletionResult.scheduled;
      withdrawal.deletionDate = deletionResult.date;
    }
    
    // Store withdrawal
    await this.storeWithdrawal(withdrawal);
    
    // Update systems
    await this.propagateWithdrawal(withdrawal);
    
    return withdrawal;
  }
}
```

### CCPA Implementation

#### Consumer Rights

```typescript
interface CCPARights {
  rightToKnow: RightToKnow;
  rightToDelete: RightToDelete;
  rightToOptOut: RightToOptOut;
  rightToNonDiscrimination: RightToNonDiscrimination;
  rightToCorrect: RightToCorrect; // CPRA addition
  rightToLimit: RightToLimit; // CPRA addition
}

class CCPAComplianceEngine {
  async handleConsumerRequest(
    request: ConsumerRequest
  ): Promise<ConsumerResponse> {
    // Verify California resident
    const isCaliforniaResident = await this.verifyCaliforniaResidency(request);
    
    if (!isCaliforniaResident) {
      return {
        status: 'not_applicable',
        reason: 'CCPA applies to California residents only'
      };
    }
    
    // Rate limiting check (2 requests per 12 months)
    const rateLimitCheck = await this.checkRateLimit(request.consumerId);
    if (!rateLimitCheck.allowed) {
      return {
        status: 'rate_limited',
        nextAllowedDate: rateLimitCheck.nextAllowedDate
      };
    }
    
    switch (request.type) {
      case 'know':
        return await this.handleKnowRequest(request);
      case 'delete':
        return await this.handleDeleteRequest(request);
      case 'opt-out':
        return await this.handleOptOutRequest(request);
      case 'opt-in':
        return await this.handleOptInRequest(request);
      default:
        throw new Error('Invalid request type');
    }
  }
  
  async handleKnowRequest(request: ConsumerRequest): Promise<ConsumerResponse> {
    const lookbackPeriod = 12; // months
    
    // Categories of personal information collected
    const categories = await this.getCollectedCategories(
      request.consumerId,
      lookbackPeriod
    );
    
    // Specific pieces of personal information
    const personalInfo = await this.getPersonalInformation(
      request.consumerId,
      lookbackPeriod
    );
    
    // Business purposes for collection
    const purposes = await this.getBusinessPurposes(request.consumerId);
    
    // Third parties with whom shared
    const thirdParties = await this.getThirdPartySharing(
      request.consumerId,
      lookbackPeriod
    );
    
    return {
      status: 'completed',
      data: {
        categories,
        personalInformation: personalInfo,
        collectionPurposes: purposes,
        thirdPartySharing: thirdParties,
        saleOfData: await this.checkDataSale(request.consumerId)
      },
      generatedDate: new Date(),
      validFor: 45 // days
    };
  }
  
  async implementDoNotSell(): Promise<void> {
    // Add "Do Not Sell My Personal Information" link
    const doNotSellButton = {
      text: 'Do Not Sell or Share My Personal Information',
      placement: ['homepage', 'footer', 'privacy-policy'],
      styling: {
        visibility: 'clear and conspicuous',
        accessibility: 'WCAG AA compliant'
      }
    };
    
    // Implement opt-out mechanism
    const optOutFlow = {
      steps: [
        'Display information about data selling',
        'Collect opt-out preference',
        'Verify request (optional)',
        'Process opt-out within 15 days',
        'Confirm opt-out to consumer'
      ],
      noAccountRequired: true,
      cookieBased: true,
      persistenceMinimum: 12 // months
    };
  }
}
```

### PII Encryption and Protection

#### Encryption Implementation

```typescript
interface EncryptionStrategy {
  atRest: EncryptionAtRest;
  inTransit: EncryptionInTransit;
  inUse: EncryptionInUse;
  keyManagement: KeyManagement;
}

class PIIEncryption {
  private readonly algorithm = 'aes-256-gcm';
  private readonly keyDerivation = 'argon2id';
  
  async encryptPII(
    data: any,
    classification: DataClassification
  ): Promise<EncryptedData> {
    // Determine encryption requirements
    const requirements = this.getEncryptionRequirements(classification);
    
    // Serialize and prepare data
    const serialized = this.serializeData(data);
    
    // Generate or retrieve DEK (Data Encryption Key)
    const dek = await this.getDataEncryptionKey(classification);
    
    // Encrypt data
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv(this.algorithm, dek, iv);
    
    const encrypted = Buffer.concat([
      cipher.update(serialized),
      cipher.final()
    ]);
    
    const authTag = cipher.getAuthTag();
    
    // Encrypt DEK with KEK (Key Encryption Key)
    const encryptedDEK = await this.encryptDEK(dek);
    
    return {
      data: encrypted,
      iv,
      authTag,
      encryptedDEK,
      algorithm: this.algorithm,
      classification,
      timestamp: new Date(),
      keyId: encryptedDEK.keyId
    };
  }
  
  async implementFieldLevelEncryption(): Promise<void> {
    // Encrypt specific fields in database
    const fieldEncryption = {
      strategy: 'deterministic', // or 'randomized'
      fields: [
        'ssn',
        'creditCardNumber',
        'bankAccount',
        'medicalRecordNumber'
      ],
      implementation: `
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        
        -- Deterministic encryption for searchable fields
        CREATE OR REPLACE FUNCTION encrypt_deterministic(
          plaintext TEXT,
          key_id UUID
        ) RETURNS TEXT AS $$
        DECLARE
          encryption_key BYTEA;
        BEGIN
          -- Retrieve key from KMS
          encryption_key := get_encryption_key(key_id);
          
          RETURN encode(
            encrypt(
              plaintext::BYTEA,
              encryption_key,
              'aes'
            ),
            'base64'
          );
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        
        -- Randomized encryption for high-security fields
        CREATE OR REPLACE FUNCTION encrypt_randomized(
          plaintext TEXT,
          key_id UUID
        ) RETURNS JSONB AS $$
        DECLARE
          encryption_key BYTEA;
          iv BYTEA;
          encrypted BYTEA;
        BEGIN
          encryption_key := get_encryption_key(key_id);
          iv := gen_random_bytes(16);
          
          encrypted := encrypt_iv(
            plaintext::BYTEA,
            encryption_key,
            iv,
            'aes'
          );
          
          RETURN jsonb_build_object(
            'data', encode(encrypted, 'base64'),
            'iv', encode(iv, 'base64'),
            'key_id', key_id,
            'algorithm', 'aes-256-cbc'
          );
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
      `
    };
  }
}
```

#### Anonymization and Pseudonymization

```typescript
interface PrivacyTechniques {
  anonymization: AnonymizationMethods;
  pseudonymization: PseudonymizationMethods;
  differential_privacy: DifferentialPrivacy;
  synthetic_data: SyntheticDataGeneration;
}

class PrivacyEngineer {
  async anonymizeDataset(
    dataset: Dataset,
    config: AnonymizationConfig
  ): Promise<AnonymizedDataset> {
    const techniques = {
      suppression: (value: any) => null,
      
      generalization: (value: any, level: number) => {
        // Example: ZIP code 12345 -> 123**
        if (typeof value === 'string' && level > 0) {
          const keepChars = Math.max(1, value.length - level);
          return value.substring(0, keepChars) + '*'.repeat(level);
        }
        return value;
      },
      
      perturbation: (value: number, noise: number) => {
        // Add random noise
        const randomNoise = (Math.random() - 0.5) * 2 * noise;
        return value + randomNoise;
      },
      
      swapping: (dataset: any[], column: string) => {
        // Randomly swap values within column
        const values = dataset.map(row => row[column]);
        const shuffled = this.shuffle([...values]);
        dataset.forEach((row, i) => {
          row[column] = shuffled[i];
        });
        return dataset;
      },
      
      aggregation: (values: number[], k: number) => {
        // k-anonymity through aggregation
        const groups = this.createGroups(values, k);
        return groups.map(group => ({
          min: Math.min(...group),
          max: Math.max(...group),
          avg: group.reduce((a, b) => a + b) / group.length,
          count: group.length
        }));
      }
    };
    
    // Apply k-anonymity
    const kAnonymized = await this.applyKAnonymity(dataset, config.k);
    
    // Apply l-diversity
    const lDiversified = await this.applyLDiversity(kAnonymized, config.l);
    
    // Apply t-closeness
    const tClose = await this.applyTCloseness(lDiversified, config.t);
    
    return {
      data: tClose,
      privacyMetrics: {
        kAnonymity: config.k,
        lDiversity: config.l,
        tCloseness: config.t,
        utilityLoss: this.calculateUtilityLoss(dataset, tClose)
      }
    };
  }
  
  async implementDifferentialPrivacy(
    query: Query,
    epsilon: number
  ): Promise<DPResult> {
    // Laplace mechanism for numeric queries
    const sensitivity = await this.calculateSensitivity(query);
    const scale = sensitivity / epsilon;
    
    const trueResult = await this.executeQuery(query);
    const noise = this.laplacianNoise(scale);
    
    return {
      result: trueResult + noise,
      epsilon,
      delta: 0, // Pure differential privacy
      mechanism: 'laplace',
      accuracy: this.calculateAccuracy(noise, trueResult)
    };
  }
}
```

### Data Retention and Deletion

#### Retention Policies

```yaml
data_retention_policies:
  default_retention:
    active_user_data: "As long as account is active"
    inactive_user_data: "2 years after last activity"
    
  data_categories:
    transactional:
      retention: "7 years"
      basis: "Tax and accounting requirements"
      
    behavioral:
      retention: "13 months"
      basis: "Analytics and improvement"
      
    marketing:
      retention: "Until consent withdrawn"
      basis: "Consent-based processing"
      
    security_logs:
      retention: "1 year"
      basis: "Security and fraud prevention"
      
    legal_hold:
      retention: "Indefinite"
      basis: "Litigation or investigation"
```

#### Automated Deletion

```typescript
class DataLifecycleManager {
  async setupRetentionPolicies(): Promise<void> {
    // Schedule periodic retention checks
    cron.schedule('0 2 * * *', async () => {
      await this.processRetentionPolicies();
    });
  }
  
  async processRetentionPolicies(): Promise<RetentionReport> {
    const policies = await this.getActiveRetentionPolicies();
    const report: RetentionReport = {
      processed: 0,
      deleted: 0,
      errors: [],
      timestamp: new Date()
    };
    
    for (const policy of policies) {
      try {
        // Identify data past retention
        const expiredData = await this.identifyExpiredData(policy);
        
        // Check for legal holds
        const filteredData = await this.filterLegalHolds(expiredData);
        
        // Perform deletion
        for (const data of filteredData) {
          await this.deleteData(data, {
            reason: 'retention_policy',
            policy: policy.id,
            cascade: true,
            backup: policy.requireBackup
          });
          
          report.deleted++;
        }
        
        report.processed += expiredData.length;
        
      } catch (error) {
        report.errors.push({
          policy: policy.id,
          error: error.message
        });
      }
    }
    
    // Generate compliance report
    await this.generateComplianceReport(report);
    
    return report;
  }
  
  async implementRightToErasure(): Promise<void> {
    // Comprehensive deletion across all systems
    const erasureWorkflow = {
      steps: [
        'Verify erasure request',
        'Check legal obligations',
        'Identify all data locations',
        'Delete from primary systems',
        'Delete from backups (or flag)',
        'Delete from third parties',
        'Remove from caches',
        'Update search indices',
        'Generate deletion certificate'
      ],
      
      implementation: async (userId: string) => {
        // Primary database deletion
        await this.deletePrimaryData(userId);
        
        // Archive and backup handling
        await this.handleBackupDeletion(userId);
        
        // Third-party notification
        await this.notifyThirdParties(userId);
        
        // Cache invalidation
        await this.invalidateCaches(userId);
        
        // Search index update
        await this.updateSearchIndices(userId);
        
        // Audit trail (keep minimal record)
        await this.createDeletionRecord(userId);
        
        return this.generateDeletionCertificate(userId);
      }
    };
  }
}
```

### Privacy Impact Assessments

#### PIA Framework

```typescript
interface PrivacyImpactAssessment {
  project: ProjectDetails;
  dataFlow: DataFlowAnalysis;
  risks: PrivacyRisk[];
  mitigations: Mitigation[];
  approval: ApprovalProcess;
}

class PIAEngine {
  async conductPIA(
    project: Project
  ): Promise<PIAResult> {
    const assessment: PrivacyImpactAssessment = {
      project: {
        name: project.name,
        description: project.description,
        dataTypes: await this.identifyDataTypes(project),
        processingActivities: await this.identifyProcessing(project),
        stakeholders: project.stakeholders
      },
      
      dataFlow: await this.analyzeDataFlow(project),
      
      risks: await this.identifyRisks(project),
      
      mitigations: [],
      
      approval: {
        required: false,
        approvers: [],
        status: 'pending'
      }
    };
    
    // Risk assessment
    for (const risk of assessment.risks) {
      const mitigation = await this.proposeMitigation(risk);
      assessment.mitigations.push(mitigation);
    }
    
    // Determine if high-risk processing
    const isHighRisk = this.isHighRiskProcessing(assessment);
    
    if (isHighRisk) {
      assessment.approval.required = true;
      assessment.approval.approvers = ['DPO', 'Legal', 'Security'];
    }
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(assessment);
    
    return {
      assessment,
      recommendations,
      riskScore: this.calculateRiskScore(assessment),
      complianceStatus: this.evaluateCompliance(assessment)
    };
  }
  
  isHighRiskProcessing(assessment: PrivacyImpactAssessment): boolean {
    const highRiskCriteria = [
      () => assessment.dataTypes.includes('biometric'),
      () => assessment.dataTypes.includes('genetic'),
      () => assessment.processingActivities.includes('profiling'),
      () => assessment.processingActivities.includes('automated-decision'),
      () => assessment.project.scale === 'large-scale',
      () => assessment.dataTypes.includes('criminal-conviction'),
      () => assessment.project.subjects.includes('children'),
      () => assessment.processingActivities.includes('surveillance'),
      () => assessment.project.technology.includes('new-tech')
    ];
    
    // If 2 or more criteria met, it's high risk (GDPR Article 35)
    const criteriaM et = highRiskCriteria.filter(criterion => criterion()).length;
    return criteriaMet >= 2;
  }
}
```

### Audit Trail Implementation

#### Comprehensive Logging

```typescript
interface AuditLog {
  id: string;
  timestamp: Date;
  actor: Actor;
  action: Action;
  resource: Resource;
  result: Result;
  metadata: Metadata;
  signature: string;
}

class PrivacyAuditLogger {
  async logDataAccess(
    access: DataAccess
  ): Promise<void> {
    const auditEntry: AuditLog = {
      id: generateUUID(),
      timestamp: new Date(),
      actor: {
        userId: access.userId,
        role: access.userRole,
        ipAddress: this.hashIP(access.ipAddress),
        sessionId: access.sessionId
      },
      action: {
        type: 'data_access',
        purpose: access.purpose,
        lawfulBasis: access.lawfulBasis
      },
      resource: {
        type: access.dataType,
        id: access.resourceId,
        classification: access.classification
      },
      result: {
        status: access.success ? 'success' : 'failure',
        recordsAccessed: access.recordCount,
        fieldsAccessed: access.fields
      },
      metadata: {
        application: access.application,
        apiVersion: access.apiVersion,
        userConsent: access.consentId
      },
      signature: ''
    };
    
    // Sign the entry for tamper-proof logging
    auditEntry.signature = await this.signAuditEntry(auditEntry);
    
    // Store in immutable audit log
    await this.storeAuditLog(auditEntry);
    
    // Real-time alerting for suspicious activity
    await this.checkSuspiciousActivity(auditEntry);
  }
  
  async generateComplianceReport(
    period: DateRange
  ): Promise<ComplianceReport> {
    const logs = await this.getAuditLogs(period);
    
    return {
      period,
      summary: {
        totalAccesses: logs.length,
        uniqueUsers: new Set(logs.map(l => l.actor.userId)).size,
        dataCategories: this.categorizeAccesses(logs),
        purposes: this.summarizePurposes(logs)
      },
      compliance: {
        unauthorizedAccesses: logs.filter(l => l.result.status === 'failure'),
        excessiveAccesses: await this.detectExcessiveAccess(logs),
        unusualPatterns: await this.detectAnomalies(logs)
      },
      recommendations: await this.generateRecommendations(logs)
    };
  }
}
```

## Best Practices

### Privacy by Design

1. **Proactive not Reactive**
   - Anticipate privacy issues
   - Prevent privacy invasions
   - Build privacy into system design

2. **Privacy as Default**
   - Maximum privacy without user action
   - Opt-in for data collection
   - Minimal data collection

3. **Full Functionality**
   - Accommodate all legitimate interests
   - Win-win not zero-sum
   - Creative solutions

4. **End-to-End Security**
   - Secure throughout lifecycle
   - Secure destruction
   - Continuous monitoring

5. **Visibility and Transparency**
   - Open about practices
   - Clear privacy notices
   - User control and choice

6. **Respect for User Privacy**
   - User-centric design
   - Strong privacy defaults
   - Clear consent mechanisms

7. **Privacy Embedded**
   - Integral part of system
   - Not bolt-on
   - Core functionality

### Common Compliance Pitfalls

- Inadequate consent mechanisms
- Poor data inventory management
- Insufficient security measures
- Lack of vendor management
- Missing privacy notices
- No retention policies
- Inadequate breach procedures
- Ignoring cross-border transfers

## Tools and Resources

### Compliance Tools
- **Privacy Management**: OneTrust, TrustArc, BigID
- **Consent Management**: Cookiebot, Usercentrics, CookieYes
- **Data Discovery**: Spirion, Varonis, Microsoft Purview

### Security Tools
- **Encryption**: HashiCorp Vault, AWS KMS, Azure Key Vault
- **DLP**: Symantec DLP, Forcepoint, Digital Guardian
- **Anonymization**: ARX, Privitar, Aircloak

### Assessment Tools
- **PIA Tools**: DPIA templates, assessment frameworks
- **Risk Assessment**: FAIR model, ISO 27005
- **Audit Tools**: Splunk, ELK Stack, Datadog

## Monitoring and Metrics

```yaml
privacy_metrics:
  compliance_metrics:
    - consent_rate: Percentage of users providing consent
    - opt_out_rate: Percentage using privacy rights
    - request_response_time: Days to fulfill requests
    - breach_notification_time: Hours to notify
    
  operational_metrics:
    - data_minimization_score: Reduction in data collected
    - encryption_coverage: Percentage of PII encrypted
    - third_party_assessments: Vendors assessed
    - training_completion: Staff privacy training
    
  risk_metrics:
    - privacy_incidents: Number and severity
    - audit_findings: Critical findings count
    - remediation_time: Average fix time
    - vulnerability_score: Privacy risk score
```

## Future Considerations

- AI governance and privacy
- Biometric data regulations
- Cross-border data frameworks
- Privacy-preserving analytics
- Homomorphic encryption adoption
- Decentralized identity systems
- Quantum-resistant encryption