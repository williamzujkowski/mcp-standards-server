// Package user provides user management functionality
package user

import (
    "context"
    "errors"
    "fmt"
    "log"
)

// ErrUserNotFound is returned when a user cannot be found
var ErrUserNotFound = errors.New("user not found")

// Service handles user operations
type Service struct {
    repo   Repository
    logger *log.Logger
}

// Repository defines the user storage interface
type Repository interface {
    GetByID(ctx context.Context, id string) (*User, error)
    Update(ctx context.Context, user *User) error
}

// User represents a user in the system
type User struct {
    ID       string `json:"id"`
    Email    string `json:"email"`
    Name     string `json:"name"`
    password string // unexported field
}

// NewService creates a new user service
func NewService(repo Repository, logger *log.Logger) *Service {
    return &Service{
        repo:   repo,
        logger: logger,
    }
}

// GetUser retrieves a user by ID
func (s *Service) GetUser(ctx context.Context, userID string) (*User, error) {
    if userID == "" {
        return nil, errors.New("user ID cannot be empty")
    }

    s.logger.Printf("Getting user: %s", userID)
    
    user, err := s.repo.GetByID(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    
    if user == nil {
        return nil, ErrUserNotFound
    }
    
    return user, nil
}
