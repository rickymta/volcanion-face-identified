// MongoDB initialization script for Face Verification API
// This script will be executed when the container starts for the first time

// Switch to the face_verification database
db = db.getSiblingDB('face_verification');

// Create application user
db.createUser({
  user: 'face_verification_user',
  pwd: 'face_verification_password',
  roles: [
    {
      role: 'readWrite',
      db: 'face_verification'
    }
  ]
});

// Create collections with validation schemas
db.createCollection('document_detections', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['detection_id', 'document_type', 'confidence_score', 'timestamp'],
      properties: {
        detection_id: {
          bsonType: 'string',
          description: 'Unique detection identifier'
        },
        document_type: {
          bsonType: 'string',
          enum: ['NATIONAL_ID', 'PASSPORT', 'DRIVER_LICENSE', 'OTHER'],
          description: 'Type of detected document'
        },
        confidence_score: {
          bsonType: 'double',
          minimum: 0,
          maximum: 1,
          description: 'Detection confidence score'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Detection timestamp'
        }
      }
    }
  }
});

db.createCollection('quality_analyses', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['quality_id', 'overall_score', 'timestamp'],
      properties: {
        quality_id: {
          bsonType: 'string',
          description: 'Unique quality analysis identifier'
        },
        overall_score: {
          bsonType: 'double',
          minimum: 0,
          maximum: 1,
          description: 'Overall quality score'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Analysis timestamp'
        }
      }
    }
  }
});

db.createCollection('face_detections', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['detection_id', 'faces_count', 'timestamp'],
      properties: {
        detection_id: {
          bsonType: 'string',
          description: 'Unique face detection identifier'
        },
        faces_count: {
          bsonType: 'int',
          minimum: 0,
          description: 'Number of faces detected'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Detection timestamp'
        }
      }
    }
  }
});

db.createCollection('face_verifications', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['verification_id', 'is_same_person', 'similarity_score', 'timestamp'],
      properties: {
        verification_id: {
          bsonType: 'string',
          description: 'Unique verification identifier'
        },
        is_same_person: {
          bsonType: 'bool',
          description: 'Whether faces belong to same person'
        },
        similarity_score: {
          bsonType: 'double',
          minimum: 0,
          maximum: 1,
          description: 'Face similarity score'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Verification timestamp'
        }
      }
    }
  }
});

db.createCollection('liveness_detections', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['liveness_id', 'is_live', 'confidence_score', 'timestamp'],
      properties: {
        liveness_id: {
          bsonType: 'string',
          description: 'Unique liveness detection identifier'
        },
        is_live: {
          bsonType: 'bool',
          description: 'Whether face is live'
        },
        confidence_score: {
          bsonType: 'double',
          minimum: 0,
          maximum: 1,
          description: 'Liveness confidence score'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Detection timestamp'
        }
      }
    }
  }
});

db.createCollection('ocr_results', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['result_id', 'extracted_text', 'timestamp'],
      properties: {
        result_id: {
          bsonType: 'string',
          description: 'Unique OCR result identifier'
        },
        extracted_text: {
          bsonType: 'string',
          description: 'Extracted text content'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Extraction timestamp'
        }
      }
    }
  }
});

db.createCollection('performance_metrics', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'endpoint', 'response_time_ms', 'status_code'],
      properties: {
        timestamp: {
          bsonType: 'date',
          description: 'Metric timestamp'
        },
        endpoint: {
          bsonType: 'string',
          description: 'API endpoint'
        },
        response_time_ms: {
          bsonType: 'double',
          minimum: 0,
          description: 'Response time in milliseconds'
        },
        status_code: {
          bsonType: 'int',
          description: 'HTTP status code'
        }
      }
    }
  }
});

// Create indexes for better performance
db.document_detections.createIndex({ 'detection_id': 1 }, { unique: true });
db.document_detections.createIndex({ 'timestamp': -1 });
db.document_detections.createIndex({ 'document_type': 1 });

db.quality_analyses.createIndex({ 'quality_id': 1 }, { unique: true });
db.quality_analyses.createIndex({ 'timestamp': -1 });

db.face_detections.createIndex({ 'detection_id': 1 }, { unique: true });
db.face_detections.createIndex({ 'timestamp': -1 });

db.face_verifications.createIndex({ 'verification_id': 1 }, { unique: true });
db.face_verifications.createIndex({ 'timestamp': -1 });

db.liveness_detections.createIndex({ 'liveness_id': 1 }, { unique: true });
db.liveness_detections.createIndex({ 'timestamp': -1 });

db.ocr_results.createIndex({ 'result_id': 1 }, { unique: true });
db.ocr_results.createIndex({ 'timestamp': -1 });

db.performance_metrics.createIndex({ 'timestamp': -1 });
db.performance_metrics.createIndex({ 'endpoint': 1, 'timestamp': -1 });

// Create TTL index for performance metrics (keep for 30 days)
db.performance_metrics.createIndex({ 'timestamp': 1 }, { expireAfterSeconds: 2592000 });

print('MongoDB initialization completed successfully!');
print('Created collections: document_detections, quality_analyses, face_detections, face_verifications, liveness_detections, ocr_results, performance_metrics');
print('Created indexes for better performance');
print('Created application user: face_verification_user');
