//! User service module

use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;
use tracing::{info, error};

/// Errors that can occur in the user service
#[derive(Error, Debug)]
pub enum UserError {
    #[error("User not found: {0}")]
    NotFound(String),
    
    #[error("Invalid user ID")]
    InvalidId,
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
}

/// User representation
#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    password_hash: String, // private field
}

/// Service for user operations
pub struct UserService {
    repository: Arc<dyn UserRepository>,
}

/// Trait for user repository operations
#[async_trait::async_trait]
pub trait UserRepository: Send + Sync {
    async fn find_by_id(&self, id: &str) -> Result<Option<User>, sqlx::Error>;
    async fn save(&self, user: &User) -> Result<(), sqlx::Error>;
}

impl UserService {
    /// Creates a new user service
    pub fn new(repository: Arc<dyn UserRepository>) -> Self {
        Self { repository }
    }
    
    /// Gets a user by ID
    pub async fn get_user(&self, user_id: &str) -> Result<User, UserError> {
        if user_id.is_empty() {
            return Err(UserError::InvalidId);
        }
        
        info!("Fetching user: {}", user_id);
        
        match self.repository.find_by_id(user_id).await? {
            Some(user) => Ok(user),
            None => {
                error!("User not found: {}", user_id);
                Err(UserError::NotFound(user_id.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_get_user_empty_id() {
        // Test implementation
    }
}
